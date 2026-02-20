#!/usr/bin/env python3
"""
Concept Learning Experiments for Hamner
=======================================
Tests what kinds of concepts a 164M param transformer can learn
through focused fine-tuning. Results inform curriculum design.

Runs 10 experiments + chained/mixed experiments, logs everything.
Each experiment starts from the same base checkpoint, trains for
N steps on one concept type, then tests accuracy.

Usage: python concept_experiments.py
"""

import os, sys, time, random, datetime, signal
import torch
import torch.nn.functional as F
from pathlib import Path

from model import HamnerModel, HamnerConfig
from variants import emotional_param_groups


# ─── Config ───────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE = "logs/experiments.log"
CKPT_DIR = "checkpoints/experiments"
BASE_CKPT = "checkpoints/curriculum/latest.pt"

BATCH_SIZE = 32
SEQ_LEN = 128
DEFAULT_LR = 5e-5
EMOTIONAL_LAYERS = 6
EMOTIONAL_LR_SCALE = 0.2
N_TESTS = 30

shutdown = False
def _sighandler(sig, frame):
    global shutdown
    log("Shutdown requested, finishing current experiment...")
    shutdown = True
signal.signal(signal.SIGINT, _sighandler)
signal.signal(signal.SIGTERM, _sighandler)


# ─── Logging ──────────────────────────────────────────────────────────────────

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ─── Model utilities ─────────────────────────────────────────────────────────

def load_base():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"Loading checkpoint: {BASE_CKPT}")
    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)

    config = HamnerConfig(**ckpt["config"])
    config.emotional_layers = EMOTIONAL_LAYERS
    config.emotional_lr_scale = EMOTIONAL_LR_SCALE
    config.gradient_checkpointing = False
    config.vocab_size = tokenizer.vocab_size

    model = HamnerModel(config).to(DEVICE)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd, strict=True)

    # Clone base weights for resetting between experiments
    base_sd = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    total_p, _ = model.count_parameters()
    log(f"Model: {total_p:,} params | {config.hidden_size}h x {config.num_layers}L")

    return model, tokenizer, config, base_sd


def reset_model(model, base_sd):
    model.load_state_dict({k: v.to(DEVICE) for k, v in base_sd.items()}, strict=True)


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new=20):
    model.eval()
    toks = tokenizer.encode(prompt, add_special_tokens=False)
    ids = torch.tensor([toks], dtype=torch.long, device=DEVICE)
    out = model.generate(
        ids, max_new_tokens=max_new, temperature=0.01,
        top_k=1, top_p=1.0, repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id or 0,
    )
    gen_ids = out[0][len(toks):].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    model.train()
    return text.strip(), gen_ids


# ─── Training loop ────────────────────────────────────────────────────────────

def train_steps(model, compiled, tokenizer, data_fn, steps, lr=DEFAULT_LR):
    """Train for N steps. Returns loss history."""
    param_groups = emotional_param_groups(model, lr)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    losses = []
    log_interval = max(steps // 10, 1)

    for step in range(1, steps + 1):
        if shutdown:
            break
        # Pack batch
        input_ids_list, labels_list = [], []
        for _ in range(BATCH_SIZE):
            tokens = []
            while len(tokens) < SEQ_LEN:
                text = data_fn() + "\n"
                tokens.extend(tokenizer.encode(text, add_special_tokens=False))
            tokens = tokens[:SEQ_LEN]
            # NO pre-shifting — model.forward() handles shift internally
            input_ids_list.append(torch.tensor(tokens, dtype=torch.long))
            labels_list.append(torch.tensor(tokens, dtype=torch.long))

        input_ids = torch.stack(input_ids_list).to(DEVICE)
        labels = torch.stack(labels_list).to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = compiled(input_ids, labels=labels)
            loss = out["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        if step % log_interval == 0 or step == steps:
            avg = sum(losses[-100:]) / len(losses[-100:])
            log(f"    step {step}/{steps} | loss {avg:.4f}")

    return losses


# ─── Test utilities ───────────────────────────────────────────────────────────

def test_experiment(model, tokenizer, test_pairs, answer_type="word"):
    """Test on (prompt, expected) pairs. Returns (accuracy, details)."""
    model.eval()
    correct = 0
    details = []

    for prompt, expected in test_pairs:
        gen_text, gen_ids = generate_text(model, tokenizer, prompt, max_new=15)

        if answer_type == "number":
            exp_digits = "".join(c for c in expected.strip() if c.isdigit())
            gen_digits = ""
            for tid in gen_ids:
                tok = tokenizer.decode([tid]).strip()
                if tok.isdigit() and len(gen_digits) < len(exp_digits):
                    gen_digits += tok
                elif gen_digits:
                    break
            match = gen_digits == exp_digits
        elif answer_type == "yesno":
            first = gen_text.split()[0].rstrip(".,!?").lower() if gen_text.split() else ""
            match = first == expected.strip().lower()
        else:  # word
            match = gen_text.lower().startswith(expected.strip().lower())

        if match:
            correct += 1
        details.append((prompt, expected, gen_text[:50], match))

    model.train()
    return correct / len(test_pairs) if test_pairs else 0, details


# ═══════════════════════════════════════════════════════════════════════════════
# DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

# --- Exp 1: Memorize fixed sentences ---
FIXED_SENTENCES = [
    "The cat sat on the mat.",
    "The dog ran in the park.",
    "A bird flew over the tree.",
    "The fish swam in the pond.",
    "The sun is bright in the sky.",
    "A frog jumped into the water.",
    "The boy kicked the red ball.",
    "A girl picked a yellow flower.",
    "The moon shines at night.",
    "Rain falls from the clouds.",
]

def gen_memorize():
    return random.choice(FIXED_SENTENCES)

def tests_memorize():
    pairs = []
    for s in FIXED_SENTENCES:
        words = s.split()
        mid = len(words) // 2
        prompt = " ".join(words[:mid])
        expected = words[mid].rstrip(".,!").lower()
        pairs.append((prompt, expected))
    return pairs


# --- Exp 2: Echo single digit ---
def gen_echo_digit():
    d = random.randint(0, 9)
    return random.choice([
        f"Number: {d}. Answer: {d}.",
        f"Say {d}. {d}.",
        f"Copy: {d} -> {d}.",
        f"Echo {d} = {d}.",
        f"The digit is {d}. It is {d}.",
    ])

def tests_echo_digit():
    pairs = []
    for d in range(10):
        pairs.append((f"Number: {d}. Answer:", str(d)))
        pairs.append((f"Copy: {d} ->", str(d)))
        pairs.append((f"Echo {d} =", str(d)))
    return pairs


# --- Exp 3: Yes/No comparisons ---
def gen_yes_no():
    a = random.randint(1, 20)
    b = random.randint(1, 20)
    while a == b:
        b = random.randint(1, 20)
    if random.random() < 0.5:
        answer = "Yes" if a > b else "No"
        return random.choice([
            f"Is {a} bigger than {b}? {answer}.",
            f"Is {a} greater than {b}? {answer}.",
            f"Is {a} more than {b}? {answer}.",
        ])
    else:
        answer = "Yes" if a < b else "No"
        return random.choice([
            f"Is {a} smaller than {b}? {answer}.",
            f"Is {a} less than {b}? {answer}.",
        ])

def tests_yes_no():
    pairs = []
    rng = random.Random(12345)
    for _ in range(N_TESTS):
        a = rng.randint(1, 20)
        b = rng.randint(1, 20)
        while a == b:
            b = rng.randint(1, 20)
        answer = "yes" if a > b else "no"
        pairs.append((f"Is {a} bigger than {b}?", answer))
    return pairs


# --- Exp 4: Opposites ---
OPPOSITES = [
    ("hot", "cold"), ("big", "small"), ("fast", "slow"),
    ("up", "down"), ("left", "right"), ("happy", "sad"),
    ("light", "dark"), ("old", "new"), ("good", "bad"),
    ("hard", "soft"), ("wet", "dry"), ("tall", "short"),
    ("loud", "quiet"), ("rich", "poor"), ("open", "closed"),
    ("full", "empty"), ("near", "far"), ("thick", "thin"),
    ("wide", "narrow"), ("deep", "shallow"),
    ("strong", "weak"), ("clean", "dirty"), ("safe", "dangerous"),
    ("early", "late"), ("true", "false"), ("high", "low"),
    ("rough", "smooth"), ("heavy", "light"), ("sharp", "dull"),
    ("sweet", "bitter"), ("bright", "dim"),
]

def gen_opposites():
    a, b = random.choice(OPPOSITES)
    if random.random() < 0.5:
        a, b = b, a
    return random.choice([
        f"The opposite of {a} is {b}.",
        f"{a.capitalize()} is the opposite of {b}.",
        f"Opposite: {a} -> {b}.",
        f"If not {a}, then {b}.",
    ])

def tests_opposites():
    pairs = []
    for a, b in OPPOSITES[:N_TESTS]:
        pairs.append((f"The opposite of {a} is", b))
    return pairs


# --- Exp 5: Single-digit addition (answer 0-9 only) ---
def gen_single_add():
    while True:
        a = random.randint(0, 9)
        b = random.randint(0, 9)
        if a + b <= 9:
            break
    c = a + b
    return random.choice([
        f"{a} + {b} = {c}",
        f"{a} plus {b} equals {c}.",
        f"Add {a} and {b} to get {c}.",
        f"What is {a} + {b}? {c}.",
        f"The sum of {a} and {b} is {c}.",
    ])

def tests_single_add():
    pairs = []
    rng = random.Random(54321)
    for _ in range(N_TESTS):
        while True:
            a = rng.randint(0, 9)
            b = rng.randint(0, 9)
            if a + b <= 9:
                break
        pairs.append((f"{a} + {b} =", str(a + b)))
    return pairs


# --- Exp 6: Q&A facts ---
QA_PAIRS_TRAIN = [
    ("What color is the sky?", "Blue"),
    ("What color is grass?", "Green"),
    ("What sound does a cat make?", "Meow"),
    ("What sound does a dog make?", "Woof"),
    ("How many legs does a dog have?", "Four"),
    ("How many legs does a spider have?", "Eight"),
    ("What do fish live in?", "Water"),
    ("What is frozen water called?", "Ice"),
    ("What comes after Monday?", "Tuesday"),
    ("Is the sun hot or cold?", "Hot"),
    ("What do birds do in the sky?", "Fly"),
    ("What color are bananas?", "Yellow"),
    ("What do you drink when thirsty?", "Water"),
    ("What falls from clouds?", "Rain"),
    ("What color is snow?", "White"),
    ("What do cows give us?", "Milk"),
    ("How many days in a week?", "Seven"),
    ("What color is a lemon?", "Yellow"),
    ("What animal says moo?", "Cow"),
    ("What do bees make?", "Honey"),
    ("What season comes after winter?", "Spring"),
    ("How many months in a year?", "Twelve"),
    ("What planet do we live on?", "Earth"),
    ("What color is the sun?", "Yellow"),
    ("What do we breathe?", "Air"),
    ("What animal has a trunk?", "Elephant"),
    ("What is the opposite of day?", "Night"),
    ("What do chickens lay?", "Eggs"),
    ("Where does a fish live?", "Water"),
    ("What is a baby cat called?", "Kitten"),
]

def gen_qa():
    q, a = random.choice(QA_PAIRS_TRAIN)
    return f"Q: {q}\nA: {a}."

def tests_qa():
    pairs = []
    for q, a in QA_PAIRS_TRAIN[:N_TESTS]:
        pairs.append((f"Q: {q}\nA:", a.lower()))
    return pairs


# --- Exp 7: Counting single digits ---
def gen_count_1digit():
    start = random.randint(0, 5)
    end = random.randint(start + 2, min(start + 6, 9))
    seq = ", ".join(str(i) for i in range(start, end + 1))
    return random.choice([
        f"Count: {seq}.",
        f"{seq}.",
        f"Sequence: {seq}.",
    ])

def tests_count_1digit():
    pairs = []
    rng = random.Random(11111)
    for _ in range(N_TESTS):
        start = rng.randint(0, 4)
        end = rng.randint(start + 3, min(start + 5, 9))
        prompt_seq = ", ".join(str(i) for i in range(start, end))
        expected = str(end)
        pairs.append((f"Count: {prompt_seq},", expected))
    return pairs


# --- Exp 8: Echo two-digit numbers ---
def gen_echo_2digit():
    n = random.randint(10, 99)
    return random.choice([
        f"Number: {n}. Answer: {n}.",
        f"Copy: {n} -> {n}.",
        f"The number is {n}. It is {n}.",
        f"Say {n}. {n}.",
        f"Echo {n} = {n}.",
    ])

def tests_echo_2digit():
    pairs = []
    rng = random.Random(22222)
    for _ in range(N_TESTS):
        n = rng.randint(10, 99)
        pairs.append((f"Number: {n}. Answer:", str(n)))
    return pairs


# --- Exp 9: Number words ---
NUM_WORDS = {
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen", 20: "twenty",
}

def gen_number_words():
    n, word = random.choice(list(NUM_WORDS.items()))
    return random.choice([
        f"{n} in words is {word}.",
        f"The number {n} is called {word}.",
        f"{word} equals {n}.",
        f"Write {n} as a word: {word}.",
    ])

def tests_number_words():
    pairs = []
    for n, word in NUM_WORDS.items():
        pairs.append((f"{n} in words is", word))
    return pairs


# --- Exp 10: Two-digit addition (answers 10-18) ---
def gen_two_digit_add():
    while True:
        a = random.randint(2, 9)
        b = random.randint(2, 9)
        if 10 <= a + b <= 18:
            break
    c = a + b
    return random.choice([
        f"{a} + {b} = {c}",
        f"{a} plus {b} equals {c}.",
        f"Add {a} and {b} to get {c}.",
        f"What is {a} + {b}? {c}.",
    ])

def tests_two_digit_add():
    pairs = []
    rng = random.Random(99999)
    for _ in range(N_TESTS):
        while True:
            a = rng.randint(2, 9)
            b = rng.randint(2, 9)
            if 10 <= a + b <= 18:
                break
        pairs.append((f"{a} + {b} =", str(a + b)))
    return pairs


# --- Mixed: all concepts ---
ALL_GENERATORS = [
    gen_memorize, gen_echo_digit, gen_yes_no, gen_opposites,
    gen_single_add, gen_qa, gen_count_1digit, gen_echo_2digit,
]

def gen_mixed():
    return random.choice(ALL_GENERATORS)()


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    {"name": "memorize",       "data_fn": gen_memorize,      "test_fn": tests_memorize,
     "answer_type": "word",    "steps": 300,   "lr": 5e-5},
    {"name": "echo_digit",     "data_fn": gen_echo_digit,    "test_fn": tests_echo_digit,
     "answer_type": "number",  "steps": 500,   "lr": 5e-5},
    {"name": "yes_no",         "data_fn": gen_yes_no,        "test_fn": tests_yes_no,
     "answer_type": "yesno",   "steps": 2000,  "lr": 5e-5},
    {"name": "opposites",      "data_fn": gen_opposites,     "test_fn": tests_opposites,
     "answer_type": "word",    "steps": 2000,  "lr": 5e-5},
    {"name": "single_add",     "data_fn": gen_single_add,    "test_fn": tests_single_add,
     "answer_type": "number",  "steps": 2000,  "lr": 5e-5},
    {"name": "qa_facts",       "data_fn": gen_qa,            "test_fn": tests_qa,
     "answer_type": "word",    "steps": 2000,  "lr": 5e-5},
    {"name": "count_1digit",   "data_fn": gen_count_1digit,  "test_fn": tests_count_1digit,
     "answer_type": "number",  "steps": 1500,  "lr": 5e-5},
    {"name": "echo_2digit",    "data_fn": gen_echo_2digit,   "test_fn": tests_echo_2digit,
     "answer_type": "number",  "steps": 5000,  "lr": 5e-5},
    {"name": "number_words",   "data_fn": gen_number_words,  "test_fn": tests_number_words,
     "answer_type": "word",    "steps": 2000,  "lr": 5e-5},
    {"name": "two_digit_add",  "data_fn": gen_two_digit_add, "test_fn": tests_two_digit_add,
     "answer_type": "number",  "steps": 5000,  "lr": 5e-5},
]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    random.seed(42)
    torch.manual_seed(42)

    log("=" * 70)
    log("CONCEPT LEARNING EXPERIMENTS")
    log("=" * 70)
    log(f"Testing {len(EXPERIMENTS)} concept types + chained/extended runs")

    model, tokenizer, config, base_sd = load_base()

    # Compile for speed (only compiles forward, generate() stays uncompiled)
    log("Compiling model with torch.compile...")
    compiled = torch.compile(model)

    # ══ Phase 1: Baseline (no training) ══════════════════════════════════════
    log("\n" + "=" * 70)
    log("PHASE 1: BASELINE (zero-shot, no fine-tuning)")
    log("=" * 70)

    baselines = {}
    for exp in EXPERIMENTS:
        test_pairs = exp["test_fn"]()
        acc, details = test_experiment(model, tokenizer, test_pairs, exp["answer_type"])
        baselines[exp["name"]] = acc
        log(f"  {exp['name']:20s}: {acc:.0%} ({sum(1 for _,_,_,m in details if m)}/{len(details)})")
        for prompt, expected, generated, match in details[:3]:
            status = "OK" if match else "XX"
            log(f"    [{status}] '{prompt[:40]}' -> '{generated[:30]}' (exp: '{expected}')")

    # ══ Phase 2: Individual experiments ══════════════════════════════════════
    log("\n" + "=" * 70)
    log("PHASE 2: INDIVIDUAL CONCEPT TRAINING")
    log("=" * 70)

    results = {}
    saved_states = {}

    for i, exp in enumerate(EXPERIMENTS):
        if shutdown:
            break

        log(f"\n{'─'*70}")
        log(f"EXPERIMENT {i+1}/{len(EXPERIMENTS)}: {exp['name']}")
        log(f"Steps: {exp['steps']} | LR: {exp['lr']} | Answer type: {exp['answer_type']}")
        log(f"{'─'*70}")

        # Reset to base weights
        reset_model(model, base_sd)

        # Train
        t0 = time.time()
        losses = train_steps(model, compiled, tokenizer, exp["data_fn"],
                             exp["steps"], exp["lr"])
        elapsed = time.time() - t0

        # Test
        test_pairs = exp["test_fn"]()
        acc, details = test_experiment(model, tokenizer, test_pairs, exp["answer_type"])

        results[exp["name"]] = {
            "accuracy": acc,
            "baseline": baselines[exp["name"]],
            "improvement": acc - baselines[exp["name"]],
            "final_loss": sum(losses[-50:]) / max(len(losses[-50:]), 1) if losses else 0,
            "steps": exp["steps"],
            "time": elapsed,
        }

        log(f"\n  RESULT: {acc:.0%} accuracy (baseline: {baselines[exp['name']]:.0%}, "
            f"delta: {acc - baselines[exp['name']]:+.0%})")
        log(f"  Loss: {results[exp['name']]['final_loss']:.4f} | Time: {elapsed:.1f}s")
        log(f"  Examples:")
        for prompt, expected, generated, match in details[:10]:
            status = "OK" if match else "XX"
            log(f"    [{status}] '{prompt[:45]}' -> '{generated[:30]}' (exp: '{expected}')")

        # Save state for chaining
        saved_states[exp["name"]] = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        # Save checkpoint
        os.makedirs(CKPT_DIR, exist_ok=True)
        ckpt_data = {
            "model_state_dict": model.state_dict(),
            "experiment": exp["name"],
            "accuracy": acc,
            "config": config.__dict__,
        }
        torch.save(ckpt_data, os.path.join(CKPT_DIR, f"{exp['name']}.pt"))

    if shutdown:
        _print_summary(results, baselines)
        return

    # ══ Phase 3: Chained experiments ═════════════════════════════════════════
    log("\n" + "=" * 70)
    log("PHASE 3: CHAINED LEARNING (does concept A help learn concept B?)")
    log("=" * 70)

    chains = [
        # (name, prerequisite_exp, data_fn, test_fn, answer_type, steps)
        ("echo1d->echo2d", "echo_digit", gen_echo_2digit, tests_echo_2digit, "number", 5000),
        ("single->two_add", "single_add", gen_two_digit_add, tests_two_digit_add, "number", 5000),
        ("count1d->echo2d", "count_1digit", gen_echo_2digit, tests_echo_2digit, "number", 5000),
        ("opposites->qa", "opposites", gen_qa, tests_qa, "word", 2000),
    ]

    for chain_name, prereq, data_fn, test_fn, atype, steps in chains:
        if shutdown:
            break
        if prereq not in saved_states:
            log(f"  Skipping chain {chain_name}: prerequisite {prereq} not available")
            continue

        log(f"\n{'─'*70}")
        log(f"CHAIN: {chain_name} ({steps} steps)")
        log(f"{'─'*70}")

        # Load prerequisite state
        model.load_state_dict({k: v.to(DEVICE) for k, v in saved_states[prereq].items()})

        # Test BEFORE chain training
        test_pairs = test_fn()
        acc_before, _ = test_experiment(model, tokenizer, test_pairs, atype)
        log(f"  Before chain training: {acc_before:.0%}")

        # Train
        t0 = time.time()
        losses = train_steps(model, compiled, tokenizer, data_fn, steps, DEFAULT_LR)
        elapsed = time.time() - t0

        # Test AFTER
        acc_after, details = test_experiment(model, tokenizer, test_pairs, atype)
        log(f"  After chain training: {acc_after:.0%} (time: {elapsed:.1f}s)")

        # Compare: train from scratch
        reset_model(model, base_sd)
        losses_scratch = train_steps(model, compiled, tokenizer, data_fn, steps, DEFAULT_LR)
        acc_scratch, details_scratch = test_experiment(model, tokenizer, test_pairs, atype)
        log(f"  From scratch (same steps): {acc_scratch:.0%}")
        log(f"  Transfer benefit: {acc_after - acc_scratch:+.0%}")

        log(f"  Chained examples:")
        for prompt, expected, generated, match in details[:5]:
            status = "OK" if match else "XX"
            log(f"    [{status}] '{prompt[:45]}' -> '{generated[:30]}' (exp: '{expected}')")

        results[f"chain_{chain_name}"] = {
            "chained_accuracy": acc_after,
            "scratch_accuracy": acc_scratch,
            "transfer_benefit": acc_after - acc_scratch,
            "pre_chain_accuracy": acc_before,
        }

    if shutdown:
        _print_summary(results, baselines)
        return

    # ══ Phase 4: Extended training for promising concepts ════════════════════
    log("\n" + "=" * 70)
    log("PHASE 4: EXTENDED TRAINING (3x steps for concepts with >20% accuracy)")
    log("=" * 70)

    for exp in EXPERIMENTS:
        if shutdown:
            break
        r = results.get(exp["name"], {})
        acc = r.get("accuracy", 0)
        if acc < 0.2 or acc >= 0.95:  # skip failures and already-saturated
            continue

        ext_steps = exp["steps"] * 3
        log(f"\n  Extended: {exp['name']} ({acc:.0%} at {exp['steps']} steps -> trying {ext_steps} steps)")

        reset_model(model, base_sd)
        t0 = time.time()
        losses = train_steps(model, compiled, tokenizer, exp["data_fn"], ext_steps, exp["lr"])
        elapsed = time.time() - t0

        test_pairs = exp["test_fn"]()
        acc_ext, details = test_experiment(model, tokenizer, test_pairs, exp["answer_type"])
        log(f"  Extended: {acc_ext:.0%} (was {acc:.0%}) | {elapsed:.1f}s")

        for prompt, expected, generated, match in details[:5]:
            status = "OK" if match else "XX"
            log(f"    [{status}] '{prompt[:45]}' -> '{generated[:30]}' (exp: '{expected}')")

        results[exp["name"]]["extended_accuracy"] = acc_ext
        results[exp["name"]]["extended_steps"] = ext_steps

        # Save if improved
        if acc_ext > acc:
            torch.save(
                {"model_state_dict": model.state_dict(), "config": config.__dict__,
                 "experiment": exp["name"], "accuracy": acc_ext},
                os.path.join(CKPT_DIR, f"{exp['name']}_extended.pt"),
            )

    if shutdown:
        _print_summary(results, baselines)
        return

    # ══ Phase 5: Mixed training (all concepts at once) ═══════════════════════
    log("\n" + "=" * 70)
    log("PHASE 5: MIXED TRAINING (all concepts simultaneously)")
    log("=" * 70)

    reset_model(model, base_sd)
    t0 = time.time()
    losses = train_steps(model, compiled, tokenizer, gen_mixed, 5000, DEFAULT_LR)
    elapsed = time.time() - t0

    log(f"  Trained 5000 steps on mixed concepts | {elapsed:.1f}s")
    log(f"  Testing each concept after mixed training:")

    mixed_results = {}
    for exp in EXPERIMENTS:
        test_pairs = exp["test_fn"]()
        acc, details = test_experiment(model, tokenizer, test_pairs, exp["answer_type"])
        mixed_results[exp["name"]] = acc
        ind_acc = results.get(exp["name"], {}).get("accuracy", 0)
        delta = acc - ind_acc
        log(f"    {exp['name']:20s}: {acc:.0%} (individual: {ind_acc:.0%}, delta: {delta:+.0%})")

    results["mixed"] = mixed_results

    # ══ Final Summary ════════════════════════════════════════════════════════
    _print_summary(results, baselines)


def _print_summary(results, baselines):
    log(f"\n{'='*70}")
    log("FINAL SUMMARY")
    log(f"{'='*70}")
    log(f"{'Experiment':<20} {'Baseline':>10} {'Trained':>10} {'Improve':>10} {'Extended':>10}")
    log("─" * 70)

    for exp in EXPERIMENTS:
        r = results.get(exp["name"], {})
        if not r:
            continue
        base = f"{r.get('baseline', 0):.0%}"
        trained = f"{r.get('accuracy', 0):.0%}"
        improve = f"{r.get('improvement', 0):+.0%}"
        ext = f"{r.get('extended_accuracy', 0):.0%}" if "extended_accuracy" in r else "---"
        log(f"{exp['name']:<20} {base:>10} {trained:>10} {improve:>10} {ext:>10}")

    # Chain results
    chain_keys = [k for k in results if k.startswith("chain_")]
    if chain_keys:
        log(f"\n{'─'*70}")
        log("CHAIN EXPERIMENTS:")
        for k in chain_keys:
            r = results[k]
            log(f"  {k}: chained={r['chained_accuracy']:.0%} scratch={r['scratch_accuracy']:.0%} "
                f"transfer={r['transfer_benefit']:+.0%}")

    # Mixed results
    if "mixed" in results:
        log(f"\n{'─'*70}")
        log("MIXED TRAINING (5000 steps all concepts):")
        for name, acc in results["mixed"].items():
            ind_acc = results.get(name, {}).get("accuracy", 0)
            log(f"  {name:20s}: {acc:.0%} (individual: {ind_acc:.0%})")

    log(f"\n{'='*70}")
    log("EXPERIMENTS COMPLETE")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
