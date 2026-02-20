#!/usr/bin/env python3
"""
Concept Learning Experiments v2 — SFT-style with answer masking
===============================================================
Round 1 showed: packed format works for numbers (93-100%) but fails for
word/classification tasks (0%). The problem: answer tokens are ~5% of the
sequence, so gradients barely touch them.

Fix: SFT-style training — mask prompt tokens, only compute loss on the
answer/completion tokens. This focuses 100% of the gradient signal on
learning the right answers.

Also tests: shorter sequences, higher LR, different concept types.

Usage: python concept_experiments_v2.py
"""

import os, sys, time, random, datetime, signal
import torch
import torch.nn.functional as F
from pathlib import Path

from model import HamnerModel, HamnerConfig
from variants import emotional_param_groups


# ─── Config ───────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE = "logs/experiments_v2.log"
CKPT_DIR = "checkpoints/experiments"
BASE_CKPT = "checkpoints/curriculum/latest.pt"

BATCH_SIZE = 32
DEFAULT_LR = 5e-5
EMOTIONAL_LAYERS = 6
EMOTIONAL_LR_SCALE = 0.2
N_TESTS = 30

shutdown = False
def _sighandler(sig, frame):
    global shutdown
    log("Shutdown requested...")
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

_base_model = None
_compiled = None
_tokenizer = None
_config = None
_base_sd = None


def load_base():
    global _base_model, _compiled, _tokenizer, _config, _base_sd

    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    log(f"Loading checkpoint: {BASE_CKPT}")
    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)

    _config = HamnerConfig(**ckpt["config"])
    _config.emotional_layers = EMOTIONAL_LAYERS
    _config.emotional_lr_scale = EMOTIONAL_LR_SCALE
    _config.gradient_checkpointing = False
    _config.vocab_size = _tokenizer.vocab_size

    _base_model = HamnerModel(_config).to(DEVICE)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    _base_model.load_state_dict(sd, strict=True)

    _base_sd = {k: v.clone().cpu() for k, v in _base_model.state_dict().items()}

    log("Compiling model...")
    _compiled = torch.compile(_base_model)

    total_p, _ = _base_model.count_parameters()
    log(f"Model: {total_p:,} params | {_config.hidden_size}h x {_config.num_layers}L")


def reset_model():
    _base_model.load_state_dict(
        {k: v.to(DEVICE) for k, v in _base_sd.items()}, strict=True
    )


@torch.no_grad()
def generate_text(prompt, max_new=20):
    _base_model.eval()
    toks = _tokenizer.encode(prompt, add_special_tokens=False)
    ids = torch.tensor([toks], dtype=torch.long, device=DEVICE)
    out = _base_model.generate(
        ids, max_new_tokens=max_new, temperature=0.01,
        top_k=1, top_p=1.0, repetition_penalty=1.0,
        eos_token_id=_tokenizer.eos_token_id or 0,
    )
    gen_ids = out[0][len(toks):].tolist()
    text = _tokenizer.decode(gen_ids, skip_special_tokens=True)
    _base_model.train()
    return text.strip(), gen_ids


# ─── SFT-style training with answer masking ──────────────────────────────────

def make_sft_batch(data_fn, batch_size, seq_len):
    """Create batch with prompt masking. data_fn returns (prompt, completion).
    Loss is ONLY computed on completion tokens.

    IMPORTANT: model.forward() already shifts labels internally:
      loss = CE(logits[:-1], labels[1:])
    So we pass input_ids=tokens and labels=tokens (UNSHIFTED).
    The model handles the offset. We just mask prompt tokens with -100.
    """
    input_ids_list, labels_list = [], []

    for _ in range(batch_size):
        prompt, completion = data_fn()

        # Tokenize separately to avoid BPE boundary issues
        prompt_tokens = _tokenizer.encode(prompt, add_special_tokens=False)
        completion_tokens = _tokenizer.encode(completion, add_special_tokens=False)
        full_tokens = prompt_tokens + completion_tokens
        n_prompt = len(prompt_tokens)

        # Pad or truncate to seq_len
        pad_id = _tokenizer.pad_token_id or 0
        while len(full_tokens) < seq_len:
            full_tokens.append(pad_id)
        full_tokens = full_tokens[:seq_len]

        # Labels: same as full_tokens but with -100 for prompt and padding
        labels = []
        for i, tok in enumerate(full_tokens):
            if i < n_prompt:
                labels.append(-100)  # mask prompt
            elif tok == pad_id:
                labels.append(-100)  # mask padding
            else:
                labels.append(tok)

        # NO pre-shifting — model.forward() handles the shift
        input_ids_list.append(torch.tensor(full_tokens, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))

    return torch.stack(input_ids_list).to(DEVICE), torch.stack(labels_list).to(DEVICE)


def make_packed_batch(data_fn, batch_size, seq_len):
    """Original packed format (for comparison). data_fn returns (prompt, completion).
    NO pre-shifting — model.forward() handles the shift internally.
    """
    input_ids_list, labels_list = [], []

    for _ in range(batch_size):
        tokens = []
        while len(tokens) < seq_len:
            prompt, completion = data_fn()
            text = prompt + completion + "\n"
            tokens.extend(_tokenizer.encode(text, add_special_tokens=False))
        tokens = tokens[:seq_len]

        # Pass same tokens as both input and labels (model shifts internally)
        input_ids_list.append(torch.tensor(tokens, dtype=torch.long))
        labels_list.append(torch.tensor(tokens, dtype=torch.long))

    return torch.stack(input_ids_list).to(DEVICE), torch.stack(labels_list).to(DEVICE)


def train_steps(data_fn, steps, lr, seq_len, use_masking=True):
    """Train with SFT masking or packed format."""
    param_groups = emotional_param_groups(_base_model, lr)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    _base_model.train()
    losses = []
    log_interval = max(steps // 10, 1)

    for step in range(1, steps + 1):
        if shutdown:
            break

        if use_masking:
            input_ids, labels = make_sft_batch(data_fn, BATCH_SIZE, seq_len)
        else:
            input_ids, labels = make_packed_batch(data_fn, BATCH_SIZE, seq_len)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = _compiled(input_ids, labels=labels)
            loss = out["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(_base_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        if step % log_interval == 0 or step == steps:
            avg = sum(losses[-100:]) / len(losses[-100:])
            log(f"    step {step}/{steps} | loss {avg:.4f}")

    return losses


# ─── Test utilities ───────────────────────────────────────────────────────────

def test_accuracy(test_pairs, answer_type="word"):
    """Test on (prompt, expected) pairs."""
    _base_model.eval()
    correct = 0
    details = []

    for prompt, expected in test_pairs:
        gen_text, gen_ids = generate_text(prompt, max_new=15)

        if answer_type == "number":
            exp_digits = "".join(c for c in expected.strip() if c.isdigit())
            gen_digits = ""
            for tid in gen_ids:
                tok = _tokenizer.decode([tid]).strip()
                if tok.isdigit() and len(gen_digits) < len(exp_digits):
                    gen_digits += tok
                elif gen_digits:
                    break
            match = gen_digits == exp_digits
        elif answer_type == "yesno":
            first = gen_text.split()[0].rstrip(".,!?").lower() if gen_text.split() else ""
            match = first == expected.strip().lower()
        else:
            match = gen_text.lower().startswith(expected.strip().lower())

        if match:
            correct += 1
        details.append((prompt, expected, gen_text[:50], match))

    _base_model.train()
    return correct / len(test_pairs) if test_pairs else 0, details


# ═══════════════════════════════════════════════════════════════════════════════
# DATA GENERATORS — return (prompt, completion) tuples
# ═══════════════════════════════════════════════════════════════════════════════

# --- Yes/No ---
def gen_yes_no():
    a = random.randint(1, 20)
    b = random.randint(1, 20)
    while a == b:
        b = random.randint(1, 20)
    if random.random() < 0.5:
        answer = "Yes" if a > b else "No"
        prompt = random.choice([
            f"Is {a} bigger than {b}?",
            f"Is {a} greater than {b}?",
            f"Is {a} more than {b}?",
        ])
    else:
        answer = "Yes" if a < b else "No"
        prompt = random.choice([
            f"Is {a} smaller than {b}?",
            f"Is {a} less than {b}?",
        ])
    return prompt, f" {answer}."

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


# --- Opposites ---
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
    prompt = random.choice([
        f"The opposite of {a} is",
        f"{a.capitalize()} is the opposite of",
        f"Opposite: {a} ->",
    ])
    return prompt, f" {b}."

def tests_opposites():
    pairs = []
    for a, b in OPPOSITES[:N_TESTS]:
        pairs.append((f"The opposite of {a} is", b))
    return pairs


# --- Q&A ---
QA_PAIRS = [
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
    q, a = random.choice(QA_PAIRS)
    return f"Q: {q}\nA:", f" {a}."

def tests_qa():
    pairs = []
    for q, a in QA_PAIRS[:N_TESTS]:
        pairs.append((f"Q: {q}\nA:", a.lower()))
    return pairs


# --- Echo two-digit ---
def gen_echo_2digit():
    n = random.randint(10, 99)
    prompt = random.choice([
        f"Number: {n}. Answer:",
        f"Copy: {n} ->",
        f"The number is {n}. It is",
        f"Say {n}.",
        f"Echo {n} =",
    ])
    return prompt, f" {n}."

def tests_echo_2digit():
    pairs = []
    rng = random.Random(22222)
    for _ in range(N_TESTS):
        n = rng.randint(10, 99)
        pairs.append((f"Number: {n}. Answer:", str(n)))
    return pairs


# --- Number words ---
NUM_WORDS = {
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen", 20: "twenty",
}

def gen_number_words():
    n, word = random.choice(list(NUM_WORDS.items()))
    prompt = random.choice([
        f"{n} in words is",
        f"The number {n} is called",
        f"Write {n} as a word:",
    ])
    return prompt, f" {word}."

def tests_number_words():
    pairs = []
    for n, word in NUM_WORDS.items():
        pairs.append((f"{n} in words is", word))
    return pairs


# --- Single-digit addition ---
def gen_single_add():
    while True:
        a = random.randint(0, 9)
        b = random.randint(0, 9)
        if a + b <= 9:
            break
    c = a + b
    prompt = random.choice([
        f"{a} + {b} =",
        f"{a} plus {b} equals",
        f"Add {a} and {b} to get",
        f"What is {a} + {b}? The answer is",
    ])
    return prompt, f" {c}."

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


# --- Two-digit addition (answers 10-18) ---
def gen_two_digit_add():
    while True:
        a = random.randint(2, 9)
        b = random.randint(2, 9)
        if 10 <= a + b <= 18:
            break
    c = a + b
    prompt = random.choice([
        f"{a} + {b} =",
        f"{a} plus {b} equals",
        f"Add {a} and {b} to get",
    ])
    return prompt, f" {c}."

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


# --- Simple chat responses ---
CHAT_PAIRS = [
    ("Hello!", "Hi there!"),
    ("How are you?", "I am doing well, thank you."),
    ("What is your name?", "My name is Hamner."),
    ("Good morning!", "Good morning to you too!"),
    ("Thank you.", "You are welcome."),
    ("Goodbye!", "Goodbye, see you later!"),
    ("Tell me a fact.", "The sky is blue."),
    ("What can you do?", "I can answer questions and chat with you."),
    ("Are you a robot?", "Yes, I am an AI assistant."),
    ("Say something nice.", "You are doing a great job!"),
    ("What day is it?", "I am not sure, but I hope it is a good day."),
    ("Do you like cats?", "Yes, cats are wonderful animals."),
    ("Help me.", "Of course, what do you need help with?"),
    ("I am happy.", "That is wonderful to hear!"),
    ("I am sad.", "I am sorry to hear that. I hope things get better."),
]

def gen_chat():
    user_msg, response = random.choice(CHAT_PAIRS)
    return f"User: {user_msg}\nAssistant:", f" {response}"

def tests_chat():
    pairs = []
    for user_msg, response in CHAT_PAIRS:
        # Check first word only
        first_word = response.split()[0].lower().rstrip(".,!?")
        pairs.append((f"User: {user_msg}\nAssistant:", first_word))
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(name, data_fn, test_fn, answer_type, steps, lr, seq_len,
                   use_masking=True, from_state=None):
    """Run one experiment. Returns (accuracy, details, state_dict)."""
    mode = "SFT-masked" if use_masking else "packed"
    log(f"\n{'─'*70}")
    log(f"EXP: {name} | {steps} steps | LR={lr} | seq={seq_len} | {mode}")
    log(f"{'─'*70}")

    if from_state:
        _base_model.load_state_dict(
            {k: v.to(DEVICE) for k, v in from_state.items()}, strict=True
        )
    else:
        reset_model()

    # Baseline test
    test_pairs = test_fn()
    base_acc, _ = test_accuracy(test_pairs, answer_type)
    log(f"  Baseline: {base_acc:.0%}")

    # Train
    t0 = time.time()
    losses = train_steps(data_fn, steps, lr, seq_len, use_masking)
    elapsed = time.time() - t0

    # Test
    acc, details = test_accuracy(test_pairs, answer_type)
    final_loss = sum(losses[-50:]) / max(len(losses[-50:]), 1) if losses else 0

    log(f"  RESULT: {acc:.0%} (baseline: {base_acc:.0%}, delta: {acc - base_acc:+.0%})")
    log(f"  Loss: {final_loss:.4f} | Time: {elapsed:.1f}s")
    for prompt, expected, generated, match in details[:8]:
        status = "OK" if match else "XX"
        log(f"    [{status}] '{prompt[:45]}' -> '{generated[:30]}' (exp: '{expected}')")

    state = {k: v.clone().cpu() for k, v in _base_model.state_dict().items()}
    return acc, details, state, final_loss


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    random.seed(42)
    torch.manual_seed(42)

    log("=" * 70)
    log("CONCEPT EXPERIMENTS v2 — SFT-style with answer masking")
    log("=" * 70)
    log("Hypothesis: v1 failed on word tasks because answer tokens were ~5%")
    log("of packed sequences. SFT masking focuses 100% of gradient on answers.")
    log("")

    load_base()

    results = {}
    saved = {}

    # ══ Round 1: SFT masking vs packed (control) ═════════════════════════════
    log("\n" + "=" * 70)
    log("ROUND 1: SFT masking vs packed format")
    log("Tests: yes_no, opposites, qa_facts, echo_2digit, number_words")
    log("=" * 70)

    concepts = [
        ("yes_no",       gen_yes_no,       tests_yes_no,       "yesno",  2000),
        ("opposites",    gen_opposites,     tests_opposites,    "word",   2000),
        ("qa_facts",     gen_qa,            tests_qa,           "word",   2000),
        ("echo_2digit",  gen_echo_2digit,   tests_echo_2digit,  "number", 3000),
        ("number_words", gen_number_words,  tests_number_words, "word",   2000),
        ("single_add",   gen_single_add,    tests_single_add,   "number", 2000),
    ]

    for name, data_fn, test_fn, atype, steps in concepts:
        if shutdown:
            break

        # SFT-masked version
        acc_sft, det_sft, state_sft, loss_sft = run_experiment(
            f"{name}_sft", data_fn, test_fn, atype, steps,
            lr=DEFAULT_LR, seq_len=48, use_masking=True,
        )

        # Packed version (control) — shorter run
        acc_pack, det_pack, state_pack, loss_pack = run_experiment(
            f"{name}_packed", data_fn, test_fn, atype, steps,
            lr=DEFAULT_LR, seq_len=128, use_masking=False,
        )

        results[name] = {
            "sft_accuracy": acc_sft,
            "packed_accuracy": acc_pack,
            "sft_loss": loss_sft,
            "packed_loss": loss_pack,
            "benefit": acc_sft - acc_pack,
        }
        saved[f"{name}_sft"] = state_sft

        log(f"\n  >>> {name}: SFT={acc_sft:.0%} vs Packed={acc_pack:.0%} "
            f"(SFT benefit: {acc_sft - acc_pack:+.0%})")

    if shutdown:
        _print_summary(results)
        return

    # ══ Round 2: LR sweep on best performing format ══════════════════════════
    log("\n" + "=" * 70)
    log("ROUND 2: LR sweep (1e-5, 5e-5, 2e-4, 5e-4) on SFT format")
    log("=" * 70)

    # Pick 3 concepts: one easy, one medium, one hard
    lr_sweep_concepts = [
        ("yes_no",       gen_yes_no,       tests_yes_no,       "yesno",  2000),
        ("opposites",    gen_opposites,     tests_opposites,    "word",   2000),
        ("echo_2digit",  gen_echo_2digit,   tests_echo_2digit,  "number", 3000),
    ]

    for lr in [1e-5, 2e-4, 5e-4]:
        if shutdown:
            break
        log(f"\n  --- LR = {lr} ---")
        for name, data_fn, test_fn, atype, steps in lr_sweep_concepts:
            if shutdown:
                break
            acc, _, state, loss = run_experiment(
                f"{name}_lr{lr}", data_fn, test_fn, atype, steps,
                lr=lr, seq_len=48, use_masking=True,
            )
            key = f"{name}_lr{lr}"
            results[key] = {"accuracy": acc, "loss": loss, "lr": lr}
            if acc > results.get(name, {}).get("sft_accuracy", 0):
                saved[f"{name}_best"] = state
                log(f"  *** New best for {name}: {acc:.0%} at LR={lr}")

    if shutdown:
        _print_summary(results)
        return

    # ══ Round 3: Extended training on promising concepts ═════════════════════
    log("\n" + "=" * 70)
    log("ROUND 3: Extended training (10k steps) for concepts showing learning")
    log("=" * 70)

    # Find which concepts showed >10% accuracy in any configuration
    promising = set()
    for key, r in results.items():
        for acc_key in ["sft_accuracy", "accuracy"]:
            if r.get(acc_key, 0) >= 0.1:
                base_name = key.split("_lr")[0].replace("_sft", "").replace("_packed", "")
                promising.add(base_name)

    for name, data_fn, test_fn, atype, _ in concepts:
        if shutdown:
            break
        if name not in promising:
            log(f"  Skipping {name} (no signal)")
            continue

        # Find best LR for this concept
        best_lr = DEFAULT_LR
        best_acc = results.get(name, {}).get("sft_accuracy", 0)
        for lr in [1e-5, 2e-4, 5e-4]:
            key = f"{name}_lr{lr}"
            if results.get(key, {}).get("accuracy", 0) > best_acc:
                best_acc = results[key]["accuracy"]
                best_lr = lr

        log(f"\n  Extended: {name} with best LR={best_lr} for 10k steps")
        acc, det, state, loss = run_experiment(
            f"{name}_extended", data_fn, test_fn, atype, 10000,
            lr=best_lr, seq_len=48, use_masking=True,
        )
        results[f"{name}_extended"] = {"accuracy": acc, "loss": loss, "lr": best_lr}
        saved[f"{name}_extended"] = state

        # Save checkpoint if good
        if acc >= 0.3:
            os.makedirs(CKPT_DIR, exist_ok=True)
            torch.save(
                {"model_state_dict": _base_model.state_dict(),
                 "config": _config.__dict__, "experiment": name, "accuracy": acc},
                os.path.join(CKPT_DIR, f"{name}_sft_extended.pt"),
            )

    if shutdown:
        _print_summary(results)
        return

    # ══ Round 4: Chained curriculum ══════════════════════════════════════════
    log("\n" + "=" * 70)
    log("ROUND 4: CHAINED CURRICULUM (build concepts on top of each other)")
    log("=" * 70)

    # Chain: single_add → two_digit_add
    log("\n  Chain: single_add → two_digit_add")
    best_single_state = saved.get("single_add_sft") or saved.get("single_add_best")
    if best_single_state:
        # From single_add state
        acc_chain, _, state_chain, _ = run_experiment(
            "chain_single→two", gen_two_digit_add, tests_two_digit_add, "number",
            5000, lr=DEFAULT_LR, seq_len=48, use_masking=True,
            from_state=best_single_state,
        )
        # From scratch
        acc_scratch, _, _, _ = run_experiment(
            "scratch_two_digit", gen_two_digit_add, tests_two_digit_add, "number",
            5000, lr=DEFAULT_LR, seq_len=48, use_masking=True,
        )
        log(f"\n  Chain transfer: {acc_chain:.0%} vs scratch: {acc_scratch:.0%} "
            f"(benefit: {acc_chain - acc_scratch:+.0%})")
        results["chain_single_to_two"] = {
            "chained": acc_chain, "scratch": acc_scratch,
            "benefit": acc_chain - acc_scratch,
        }

    # Chain: echo_2digit → two_digit_add
    log("\n  Chain: echo_2digit → two_digit_add")
    best_echo_state = saved.get("echo_2digit_sft") or saved.get("echo_2digit_best")
    if best_echo_state:
        acc_chain2, _, _, _ = run_experiment(
            "chain_echo→two", gen_two_digit_add, tests_two_digit_add, "number",
            5000, lr=DEFAULT_LR, seq_len=48, use_masking=True,
            from_state=best_echo_state,
        )
        results["chain_echo_to_two"] = {"chained": acc_chain2}
        log(f"  Echo→Two: {acc_chain2:.0%}")

    # Chain: opposites → qa_facts
    log("\n  Chain: opposites → qa_facts")
    best_opp_state = saved.get("opposites_sft") or saved.get("opposites_best")
    if best_opp_state:
        acc_chain3, _, _, _ = run_experiment(
            "chain_opp→qa", gen_qa, tests_qa, "word",
            3000, lr=DEFAULT_LR, seq_len=64, use_masking=True,
            from_state=best_opp_state,
        )
        results["chain_opp_to_qa"] = {"chained": acc_chain3}

    if shutdown:
        _print_summary(results)
        return

    # ══ Round 5: Chat format training ════════════════════════════════════════
    log("\n" + "=" * 70)
    log("ROUND 5: CHAT FORMAT TRAINING (User/Assistant pairs)")
    log("=" * 70)

    acc_chat, det_chat, state_chat, _ = run_experiment(
        "chat_basic", gen_chat, tests_chat, "word",
        3000, lr=DEFAULT_LR, seq_len=64, use_masking=True,
    )
    results["chat_basic"] = {"accuracy": acc_chat}

    # Extended chat
    if acc_chat >= 0.1:
        acc_chat_ext, _, _, _ = run_experiment(
            "chat_extended", gen_chat, tests_chat, "word",
            10000, lr=DEFAULT_LR, seq_len=64, use_masking=True,
        )
        results["chat_extended"] = {"accuracy": acc_chat_ext}

    if shutdown:
        _print_summary(results)
        return

    # ══ Round 6: Multi-task SFT (all concepts together) ═════════════════════
    log("\n" + "=" * 70)
    log("ROUND 6: MULTI-TASK SFT (all concepts simultaneously)")
    log("=" * 70)

    all_generators = [
        gen_yes_no, gen_opposites, gen_qa, gen_echo_2digit,
        gen_number_words, gen_single_add, gen_chat,
    ]

    def gen_multi():
        return random.choice(all_generators)()

    reset_model()
    t0 = time.time()
    losses = train_steps(gen_multi, 10000, DEFAULT_LR, 64, use_masking=True)
    elapsed = time.time() - t0
    log(f"  Multi-task training: 10k steps, {elapsed:.1f}s")

    log(f"  Per-concept accuracy after multi-task SFT:")
    multi_results = {}
    for name, _, test_fn, atype, _ in concepts:
        test_pairs = test_fn()
        acc, details = test_accuracy(test_pairs, atype)
        multi_results[name] = acc
        ind = results.get(name, {}).get("sft_accuracy", 0)
        log(f"    {name:20s}: {acc:.0%} (individual SFT: {ind:.0%})")

    # Also test chat
    chat_test = tests_chat()
    acc_chat_multi, _ = test_accuracy(chat_test, "word")
    multi_results["chat"] = acc_chat_multi
    log(f"    {'chat':20s}: {acc_chat_multi:.0%}")

    results["multi_task"] = multi_results

    # ══ Summary ══════════════════════════════════════════════════════════════
    _print_summary(results)


def _print_summary(results):
    log(f"\n{'='*70}")
    log("FINAL SUMMARY — v2 SFT Experiments")
    log(f"{'='*70}")

    # Round 1: SFT vs Packed
    log(f"\n{'─'*70}")
    log(f"{'Concept':<20} {'SFT':>10} {'Packed':>10} {'Benefit':>10}")
    log(f"{'─'*70}")
    for key in ["yes_no", "opposites", "qa_facts", "echo_2digit", "number_words", "single_add"]:
        r = results.get(key, {})
        if "sft_accuracy" in r:
            log(f"{key:<20} {r['sft_accuracy']:>10.0%} {r['packed_accuracy']:>10.0%} "
                f"{r['benefit']:>+10.0%}")

    # LR sweep
    log(f"\n{'─'*70}")
    log("LR SWEEP (SFT format):")
    for key, r in sorted(results.items()):
        if "_lr" in key and "accuracy" in r:
            log(f"  {key:<30} {r['accuracy']:.0%} (LR={r['lr']})")

    # Extended
    log(f"\n{'─'*70}")
    log("EXTENDED (10k steps):")
    for key, r in sorted(results.items()):
        if "_extended" in key and "accuracy" in r:
            log(f"  {key:<30} {r['accuracy']:.0%}")

    # Chains
    log(f"\n{'─'*70}")
    log("CHAINS:")
    for key, r in sorted(results.items()):
        if "chain_" in key:
            log(f"  {key}: {r}")

    # Multi-task
    if "multi_task" in results:
        log(f"\n{'─'*70}")
        log("MULTI-TASK SFT (10k steps all concepts):")
        for name, acc in results["multi_task"].items():
            log(f"  {name:20s}: {acc:.0%}")

    log(f"\n{'='*70}")
    log("v2 EXPERIMENTS COMPLETE")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
