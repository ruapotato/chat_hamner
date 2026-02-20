#!/usr/bin/env python3
"""
Concept Experiments v3 — Scaffolding, Generalization & Composition
==================================================================
v2 showed ALL concepts train to 100% individually and in multi-task.
Now we test the harder questions:

1. GENERALIZATION: Does the model learn the skill, or just memorize examples?
   - Hold out test examples, see if format generalizes
2. COMPOSITION: Can independently learned skills combine?
   - Learn "opposites" + "Q&A" → answer "What is the opposite of hot?"
3. RETENTION: Does concept A survive when we train concept B?
4. SCALING: How many distinct facts can coexist?
5. CORRECTED PRETRAINING: Fresh pretrain pass with fixed labels

Usage: python concept_experiments_v3.py
"""

import os, sys, time, random, datetime, signal, json
import torch
import torch.nn.functional as F
from pathlib import Path

from model import HamnerModel, HamnerConfig
from variants import emotional_param_groups


# ─── Config ───────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE = "logs/experiments_v3.log"
BASE_CKPT = "checkpoints/curriculum/latest.pt"

BATCH_SIZE = 32
DEFAULT_LR = 5e-5
EMOTIONAL_LAYERS = 6
EMOTIONAL_LR_SCALE = 0.2

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


def save_state():
    return {k: v.clone().cpu() for k, v in _base_model.state_dict().items()}


def load_state(state):
    _base_model.load_state_dict(
        {k: v.to(DEVICE) for k, v in state.items()}, strict=True
    )


@torch.no_grad()
def generate_text(prompt, max_new=30):
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


# ─── Training ────────────────────────────────────────────────────────────────

def make_sft_batch(data_fn, batch_size, seq_len):
    """SFT batch with prompt masking. model.forward() shifts labels internally."""
    input_ids_list, labels_list = [], []
    for _ in range(batch_size):
        prompt, completion = data_fn()
        prompt_tokens = _tokenizer.encode(prompt, add_special_tokens=False)
        completion_tokens = _tokenizer.encode(completion, add_special_tokens=False)
        full_tokens = prompt_tokens + completion_tokens
        n_prompt = len(prompt_tokens)
        pad_id = _tokenizer.pad_token_id or 0
        while len(full_tokens) < seq_len:
            full_tokens.append(pad_id)
        full_tokens = full_tokens[:seq_len]
        labels = []
        for i, tok in enumerate(full_tokens):
            if i < n_prompt:
                labels.append(-100)
            elif tok == pad_id:
                labels.append(-100)
            else:
                labels.append(tok)
        input_ids_list.append(torch.tensor(full_tokens, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))
    return torch.stack(input_ids_list).to(DEVICE), torch.stack(labels_list).to(DEVICE)


def train_steps(data_fn, steps, lr, seq_len=64, log_interval=None):
    """Train with SFT masking."""
    param_groups = emotional_param_groups(_base_model, lr)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    _base_model.train()
    losses = []
    if log_interval is None:
        log_interval = max(steps // 10, 1)

    for step in range(1, steps + 1):
        if shutdown:
            break
        input_ids, labels = make_sft_batch(data_fn, BATCH_SIZE, seq_len)
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


# ─── Testing ──────────────────────────────────────────────────────────────────

def test_pairs_accuracy(test_pairs, answer_type="word"):
    """Test on (prompt, expected_answer) pairs."""
    _base_model.eval()
    correct = 0
    details = []

    for prompt, expected in test_pairs:
        gen_text, gen_ids = generate_text(prompt, max_new=20)

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
        elif answer_type == "exact":
            # Full string match (lowercased, stripped)
            match = gen_text.lower().strip().startswith(expected.strip().lower())
        else:
            match = gen_text.lower().startswith(expected.strip().lower())

        if match:
            correct += 1
        details.append((prompt, expected, gen_text[:60], match))

    _base_model.train()
    acc = correct / len(test_pairs) if test_pairs else 0
    return acc, details


def print_test_results(name, acc, details, show=8):
    log(f"  {name}: {acc:.0%} ({sum(1 for _,_,_,m in details if m)}/{len(details)})")
    for prompt, expected, generated, match in details[:show]:
        status = "OK" if match else "XX"
        log(f"    [{status}] '{prompt[:40]}' -> '{generated[:35]}' (exp: '{expected}')")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA: Expanded datasets with train/test splits
# ═══════════════════════════════════════════════════════════════════════════════

# --- Q&A Facts (60 total: 30 train, 30 test) ---
ALL_QA = [
    # Train set (30)
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
    # Test set (30) - HELD OUT, never seen during training
    ("What color is a fire truck?", "Red"),
    ("What sound does a cow make?", "Moo"),
    ("How many wheels does a bicycle have?", "Two"),
    ("What do we see with?", "Eyes"),
    ("What season is the coldest?", "Winter"),
    ("What animal barks?", "Dog"),
    ("What is a baby dog called?", "Puppy"),
    ("What color is the ocean?", "Blue"),
    ("How many fingers do humans have?", "Ten"),
    ("What comes after Wednesday?", "Thursday"),
    ("What do trees have?", "Leaves"),
    ("What animal has stripes?", "Zebra"),
    ("What do we hear with?", "Ears"),
    ("What planet is closest to the sun?", "Mercury"),
    ("What is the tallest animal?", "Giraffe"),
    ("What color is a stop sign?", "Red"),
    ("How many eyes do humans have?", "Two"),
    ("What do caterpillars turn into?", "Butterflies"),
    ("What is the largest ocean?", "Pacific"),
    ("What do rabbits eat?", "Carrots"),
    ("What month comes after January?", "February"),
    ("What animal has a shell?", "Turtle"),
    ("What do plants need to grow?", "Water"),
    ("What is the hottest season?", "Summer"),
    ("How many sides does a triangle have?", "Three"),
    ("What animal can fly?", "Bird"),
    ("What color is chocolate?", "Brown"),
    ("What do we smell with?", "Nose"),
    ("What comes after Friday?", "Saturday"),
    ("What is a group of fish called?", "School"),
]

QA_TRAIN = ALL_QA[:30]
QA_TEST = ALL_QA[30:]

def gen_qa_train():
    q, a = random.choice(QA_TRAIN)
    return f"Q: {q}\nA:", f" {a}."

def gen_qa_test_pairs():
    return [(f"Q: {q}\nA:", a.lower()) for q, a in QA_TEST]

def gen_qa_train_pairs():
    return [(f"Q: {q}\nA:", a.lower()) for q, a in QA_TRAIN]


# --- Opposites (31 total: 16 train, 15 test) ---
ALL_OPPOSITES = [
    # Train (16)
    ("hot", "cold"), ("big", "small"), ("fast", "slow"),
    ("up", "down"), ("left", "right"), ("happy", "sad"),
    ("light", "dark"), ("old", "new"), ("good", "bad"),
    ("hard", "soft"), ("wet", "dry"), ("tall", "short"),
    ("loud", "quiet"), ("rich", "poor"), ("open", "closed"),
    ("full", "empty"),
    # Test (15) - HELD OUT
    ("near", "far"), ("thick", "thin"), ("wide", "narrow"),
    ("deep", "shallow"), ("strong", "weak"), ("clean", "dirty"),
    ("safe", "dangerous"), ("early", "late"), ("true", "false"),
    ("high", "low"), ("rough", "smooth"), ("heavy", "light"),
    ("sharp", "dull"), ("sweet", "bitter"), ("bright", "dim"),
]

OPP_TRAIN = ALL_OPPOSITES[:16]
OPP_TEST = ALL_OPPOSITES[16:]

def gen_opp_train():
    a, b = random.choice(OPP_TRAIN)
    if random.random() < 0.5:
        a, b = b, a
    return f"The opposite of {a} is", f" {b}."

def gen_opp_test_pairs():
    pairs = []
    for a, b in OPP_TEST:
        pairs.append((f"The opposite of {a} is", b))
    return pairs

def gen_opp_train_pairs():
    pairs = []
    for a, b in OPP_TRAIN:
        pairs.append((f"The opposite of {a} is", b))
    return pairs


# --- Chat pairs (30 total: 15 train, 15 test) ---
ALL_CHAT = [
    # Train (15)
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
    # Test (15) - HELD OUT
    ("Good evening!", "Good evening! How can I help you?"),
    ("What is 2 plus 2?", "That equals 4."),
    ("Tell me a joke.", "Why did the chicken cross the road? To get to the other side!"),
    ("Do you like music?", "Yes, music is a wonderful thing."),
    ("I need advice.", "I would be happy to help. What is on your mind?"),
    ("What is your favorite color?", "I do not have a favorite color, but blue is nice."),
    ("Can you sing?", "I cannot sing, but I can chat with you!"),
    ("I am tired.", "You should rest. Sleep is important for health."),
    ("What is love?", "Love is a deep feeling of care for someone."),
    ("Do you dream?", "I do not dream, but I find the concept fascinating."),
    ("I like you.", "Thank you, that is very kind of you to say!"),
    ("What time is it?", "I am not sure of the exact time right now."),
    ("Surprise me.", "Did you know honey never spoils?"),
    ("You are smart.", "Thank you! I try my best to be helpful."),
    ("Good night!", "Good night! Sleep well and sweet dreams."),
]

CHAT_TRAIN = ALL_CHAT[:15]
CHAT_TEST = ALL_CHAT[15:]

def gen_chat_train():
    user_msg, response = random.choice(CHAT_TRAIN)
    return f"User: {user_msg}\nAssistant:", f" {response}"

def gen_chat_test_pairs():
    pairs = []
    for user_msg, response in CHAT_TEST:
        first_word = response.split()[0].lower().rstrip(".,!?")
        pairs.append((f"User: {user_msg}\nAssistant:", first_word))
    return pairs

def gen_chat_train_pairs():
    pairs = []
    for user_msg, response in CHAT_TRAIN:
        first_word = response.split()[0].lower().rstrip(".,!?")
        pairs.append((f"User: {user_msg}\nAssistant:", first_word))
    return pairs


# --- Math ---
def gen_single_add():
    while True:
        a = random.randint(0, 9)
        b = random.randint(0, 9)
        if a + b <= 9:
            break
    return f"{a} + {b} =", f" {a + b}."

def gen_two_digit_add():
    while True:
        a = random.randint(2, 9)
        b = random.randint(2, 9)
        if 10 <= a + b <= 18:
            break
    return f"{a} + {b} =", f" {a + b}."

def gen_subtraction():
    """Single-digit subtraction with non-negative results."""
    a = random.randint(1, 9)
    b = random.randint(0, a)  # b <= a so result >= 0
    return f"{a} - {b} =", f" {a - b}."

def gen_multiplication():
    """Small multiplication: single digits, answer <= 81."""
    a = random.randint(1, 9)
    b = random.randint(1, 9)
    return f"{a} x {b} =", f" {a * b}."

def tests_single_add():
    rng = random.Random(54321)
    pairs = []
    for _ in range(30):
        while True:
            a = rng.randint(0, 9)
            b = rng.randint(0, 9)
            if a + b <= 9:
                break
        pairs.append((f"{a} + {b} =", str(a + b)))
    return pairs

def tests_two_digit_add():
    rng = random.Random(99999)
    pairs = []
    for _ in range(30):
        while True:
            a = rng.randint(2, 9)
            b = rng.randint(2, 9)
            if 10 <= a + b <= 18:
                break
        pairs.append((f"{a} + {b} =", str(a + b)))
    return pairs

def tests_subtraction():
    rng = random.Random(77777)
    pairs = []
    for _ in range(30):
        a = rng.randint(1, 9)
        b = rng.randint(0, a)
        pairs.append((f"{a} - {b} =", str(a - b)))
    return pairs

def tests_multiplication():
    rng = random.Random(88888)
    pairs = []
    for _ in range(30):
        a = rng.randint(1, 9)
        b = rng.randint(1, 9)
        pairs.append((f"{a} x {b} =", str(a * b)))
    return pairs


# --- Composition test data ---
def gen_qa_opposite():
    """Q&A format asking about opposites (never trained on this exact combo)."""
    a, b = random.choice(ALL_OPPOSITES)
    if random.random() < 0.5:
        return f"Q: What is the opposite of {a}?\nA:", f" {b.capitalize()}."
    else:
        return f"Q: What is the opposite of {b}?\nA:", f" {a.capitalize()}."

def tests_qa_opposite():
    """Test if model can combine Q&A format with opposites knowledge."""
    pairs = []
    for a, b in ALL_OPPOSITES[:20]:
        pairs.append((f"Q: What is the opposite of {a}?\nA:", b))
    return pairs


# --- Scaled facts (100 facts for scaling test) ---
SCALED_FACTS = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"),
    ("What is the capital of England?", "London"),
    ("What is the capital of Spain?", "Madrid"),
    ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of Russia?", "Moscow"),
    ("What is the capital of China?", "Beijing"),
    ("What is the capital of India?", "Delhi"),
    ("What is the capital of Brazil?", "Brasilia"),
    ("What is the capital of Australia?", "Canberra"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the capital of Egypt?", "Cairo"),
    ("What is the capital of Mexico?", "Mexico City"),
    ("What is the capital of Thailand?", "Bangkok"),
    ("What is the capital of Turkey?", "Ankara"),
    ("What is the capital of Greece?", "Athens"),
    ("What is the capital of Sweden?", "Stockholm"),
    ("What is the capital of Norway?", "Oslo"),
    ("What is the capital of Portugal?", "Lisbon"),
    ("What is the biggest planet?", "Jupiter"),
    ("What is the smallest planet?", "Mercury"),
    ("What is the closest star?", "Sun"),
    ("What is the fastest land animal?", "Cheetah"),
    ("What is the largest animal?", "Blue whale"),
    ("What is the tallest mountain?", "Everest"),
    ("What is the longest river?", "Nile"),
    ("What is the largest desert?", "Sahara"),
    ("What is the largest continent?", "Asia"),
    ("What is the smallest continent?", "Australia"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("Who painted the Mona Lisa?", "Da Vinci"),
    ("What is H2O?", "Water"),
    ("How many sides does a square have?", "Four"),
    ("How many sides does a hexagon have?", "Six"),
    ("What color do you get mixing red and blue?", "Purple"),
    ("What color do you get mixing red and yellow?", "Orange"),
    ("What color do you get mixing blue and yellow?", "Green"),
    ("How many hours in a day?", "Twenty four"),
    ("How many minutes in an hour?", "Sixty"),
    ("How many seconds in a minute?", "Sixty"),
    ("How many weeks in a year?", "Fifty two"),
    ("How many continents are there?", "Seven"),
    ("How many oceans are there?", "Five"),
    ("What do plants use for energy?", "Sunlight"),
    ("What gas do plants release?", "Oxygen"),
    ("What gas do humans exhale?", "Carbon dioxide"),
    ("What organ pumps blood?", "Heart"),
    ("What is the hardest natural substance?", "Diamond"),
    ("What metal is liquid at room temperature?", "Mercury"),
]

def gen_scaled_facts(n_facts=50):
    """Return a generator that samples from the first n_facts."""
    facts = SCALED_FACTS[:n_facts]
    def _gen():
        q, a = random.choice(facts)
        return f"Q: {q}\nA:", f" {a}."
    return _gen

def tests_scaled_facts(n_facts=50):
    pairs = []
    for q, a in SCALED_FACTS[:n_facts]:
        pairs.append((f"Q: {q}\nA:", a.lower()))
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_exp(name, data_fn, steps, lr=DEFAULT_LR, seq_len=64, from_state=None):
    """Train and return state dict."""
    log(f"\n{'─'*70}")
    log(f"  TRAIN: {name} | {steps} steps | LR={lr}")
    log(f"{'─'*70}")

    if from_state:
        load_state(from_state)
    else:
        reset_model()

    t0 = time.time()
    losses = train_steps(data_fn, steps, lr, seq_len)
    elapsed = time.time() - t0
    final_loss = sum(losses[-50:]) / max(len(losses[-50:]), 1) if losses else 0
    log(f"  Done: {elapsed:.0f}s | final loss: {final_loss:.4f}")
    return save_state(), final_loss


def test_exp(name, test_pairs, answer_type="word"):
    """Test current model state."""
    acc, details = test_pairs_accuracy(test_pairs, answer_type)
    print_test_results(name, acc, details)
    return acc, details


def main():
    random.seed(42)
    torch.manual_seed(42)

    log("=" * 70)
    log("CONCEPT EXPERIMENTS v3 — Scaffolding, Generalization & Composition")
    log("=" * 70)

    load_base()

    results = {}

    # ══════════════════════════════════════════════════════════════════════
    # ROUND 1: GENERALIZATION — Does the model learn the skill or memorize?
    # ══════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("ROUND 1: GENERALIZATION TEST")
    log("Train on subset, test on held-out examples")
    log("=" * 70)

    # 1a: Q&A generalization (train on 30 QA, test on 30 new QA)
    log("\n  --- Q&A Generalization ---")
    state_qa, _ = run_exp("qa_train30", gen_qa_train, 5000)
    acc_train, _ = test_exp("qa_train_set", gen_qa_train_pairs(), "word")
    acc_test, det_test = test_exp("qa_held_out", gen_qa_test_pairs(), "word")
    results["qa_generalization"] = {
        "train_acc": acc_train, "test_acc": acc_test,
        "generalization_rate": acc_test / max(acc_train, 0.01),
    }
    log(f"  >>> Q&A: train={acc_train:.0%}, held-out={acc_test:.0%}, "
        f"generalization={acc_test/max(acc_train,0.01):.0%}")

    if shutdown: return _summary(results)

    # 1b: Opposites generalization (train on 16, test on 15 new)
    log("\n  --- Opposites Generalization ---")
    state_opp, _ = run_exp("opp_train16", gen_opp_train, 5000)
    acc_train, _ = test_exp("opp_train_set", gen_opp_train_pairs(), "word")
    acc_test, det_test = test_exp("opp_held_out", gen_opp_test_pairs(), "word")
    results["opp_generalization"] = {
        "train_acc": acc_train, "test_acc": acc_test,
        "generalization_rate": acc_test / max(acc_train, 0.01),
    }
    log(f"  >>> Opposites: train={acc_train:.0%}, held-out={acc_test:.0%}, "
        f"generalization={acc_test/max(acc_train,0.01):.0%}")

    if shutdown: return _summary(results)

    # 1c: Chat generalization (train on 15, test on 15 new)
    log("\n  --- Chat Generalization ---")
    state_chat, _ = run_exp("chat_train15", gen_chat_train, 5000)
    acc_train, _ = test_exp("chat_train_set", gen_chat_train_pairs(), "word")
    acc_test, det_test = test_exp("chat_held_out", gen_chat_test_pairs(), "word")
    results["chat_generalization"] = {
        "train_acc": acc_train, "test_acc": acc_test,
        "generalization_rate": acc_test / max(acc_train, 0.01),
    }
    log(f"  >>> Chat: train={acc_train:.0%}, held-out={acc_test:.0%}, "
        f"generalization={acc_test/max(acc_train,0.01):.0%}")

    if shutdown: return _summary(results)

    # 1d: Math generalization - train on addition, test on new random examples
    log("\n  --- Math Generalization (trained: addition, tested: subtraction, multiplication) ---")
    state_math, _ = run_exp("math_add", gen_single_add, 3000)
    acc_add, _ = test_exp("add_trained", tests_single_add(), "number")
    acc_sub, det_sub = test_exp("sub_zero_shot", tests_subtraction(), "number")
    acc_mul, det_mul = test_exp("mul_zero_shot", tests_multiplication(), "number")
    results["math_transfer"] = {
        "add_trained": acc_add,
        "sub_zero_shot": acc_sub,
        "mul_zero_shot": acc_mul,
    }
    log(f"  >>> Math: add(trained)={acc_add:.0%}, sub(zero-shot)={acc_sub:.0%}, "
        f"mul(zero-shot)={acc_mul:.0%}")

    if shutdown: return _summary(results)

    # ══════════════════════════════════════════════════════════════════════
    # ROUND 2: CONCEPT COMPOSITION — Can skills combine?
    # ══════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("ROUND 2: CONCEPT COMPOSITION")
    log("Train skills separately, test if they combine")
    log("=" * 70)

    # 2a: Train Q&A format + opposites separately, then test Q&A-about-opposites
    log("\n  --- Composition: Q&A format + Opposites → Q&A-about-opposites ---")

    # First: train Q&A format
    state_qa, _ = run_exp("comp_qa_format", gen_qa_train, 3000)
    # Then: train opposites ON TOP of Q&A state
    state_qa_opp, _ = run_exp("comp_opp_on_qa", gen_opp_train, 3000, from_state=state_qa)
    # Test composition: Q&A format about opposites (never trained on this combo!)
    acc_comp, det_comp = test_exp("qa_about_opposites", tests_qa_opposite(), "word")
    # Also test if original skills survived
    acc_qa_after, _ = test_exp("qa_after_opp", gen_qa_train_pairs(), "word")
    acc_opp_after, _ = test_exp("opp_after_comp", gen_opp_train_pairs(), "word")

    results["composition_qa_opp"] = {
        "composed_acc": acc_comp,
        "qa_retained": acc_qa_after,
        "opp_retained": acc_opp_after,
    }
    log(f"  >>> Composition: Q&A-about-opposites={acc_comp:.0%}, "
        f"QA retained={acc_qa_after:.0%}, Opp retained={acc_opp_after:.0%}")

    if shutdown: return _summary(results)

    # 2b: Alternative — train both together, then test composition
    log("\n  --- Composition v2: Train Q&A + Opposites JOINTLY, test combination ---")
    def gen_qa_and_opp():
        if random.random() < 0.5:
            return gen_qa_train()
        return gen_opp_train()
    state_joint, _ = run_exp("comp_joint", gen_qa_and_opp, 5000)
    acc_comp2, det_comp2 = test_exp("qa_about_opp_joint", tests_qa_opposite(), "word")
    acc_qa2, _ = test_exp("qa_after_joint", gen_qa_train_pairs(), "word")
    acc_opp2, _ = test_exp("opp_after_joint", gen_opp_train_pairs(), "word")

    results["composition_joint"] = {
        "composed_acc": acc_comp2,
        "qa_retained": acc_qa2,
        "opp_retained": acc_opp2,
    }
    log(f"  >>> Joint composition: Q&A-about-opp={acc_comp2:.0%}, "
        f"QA={acc_qa2:.0%}, Opp={acc_opp2:.0%}")

    if shutdown: return _summary(results)

    # 2c: Train explicitly on the composition task
    log("\n  --- Composition v3: Train DIRECTLY on Q&A-about-opposites ---")
    state_direct, _ = run_exp("comp_direct", gen_qa_opposite, 3000)
    acc_direct, det_direct = test_exp("qa_opp_direct", tests_qa_opposite(), "word")
    results["composition_direct"] = {"accuracy": acc_direct}
    log(f"  >>> Direct training on composition: {acc_direct:.0%}")

    if shutdown: return _summary(results)

    # ══════════════════════════════════════════════════════════════════════
    # ROUND 3: RETENTION — Does concept A survive training on concept B?
    # ══════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("ROUND 3: RETENTION / CATASTROPHIC FORGETTING")
    log("Train A, then train B, re-test A")
    log("=" * 70)

    # 3a: Chat → QA → re-test Chat
    log("\n  --- Retention: Chat → QA → re-test Chat ---")
    state_a, _ = run_exp("ret_chat", gen_chat_train, 3000)
    acc_a_before, _ = test_exp("chat_after_training", gen_chat_train_pairs(), "word")

    state_b, _ = run_exp("ret_qa_after_chat", gen_qa_train, 3000, from_state=state_a)
    acc_a_after, _ = test_exp("chat_after_qa", gen_chat_train_pairs(), "word")
    acc_b_after, _ = test_exp("qa_after_qa", gen_qa_train_pairs(), "word")

    results["retention_chat_qa"] = {
        "chat_before": acc_a_before,
        "chat_after_qa": acc_a_after,
        "qa_accuracy": acc_b_after,
        "chat_retained": acc_a_after / max(acc_a_before, 0.01),
    }
    log(f"  >>> Chat before: {acc_a_before:.0%}, Chat after QA: {acc_a_after:.0%} "
        f"(retained: {acc_a_after/max(acc_a_before,0.01):.0%}), QA: {acc_b_after:.0%}")

    if shutdown: return _summary(results)

    # 3b: QA → Opposites → Math → re-test all
    log("\n  --- Retention chain: QA → Opp → Math → test all ---")
    state1, _ = run_exp("chain_qa", gen_qa_train, 3000)
    acc_qa1, _ = test_exp("qa_step1", gen_qa_train_pairs(), "word")

    state2, _ = run_exp("chain_opp", gen_opp_train, 3000, from_state=state1)
    acc_qa2, _ = test_exp("qa_step2", gen_qa_train_pairs(), "word")
    acc_opp2, _ = test_exp("opp_step2", gen_opp_train_pairs(), "word")

    state3, _ = run_exp("chain_math", gen_single_add, 3000, from_state=state2)
    acc_qa3, _ = test_exp("qa_step3", gen_qa_train_pairs(), "word")
    acc_opp3, _ = test_exp("opp_step3", gen_opp_train_pairs(), "word")
    acc_math3, _ = test_exp("math_step3", tests_single_add(), "number")

    results["retention_chain"] = {
        "qa_after_qa": acc_qa1, "qa_after_opp": acc_qa2, "qa_after_math": acc_qa3,
        "opp_after_opp": acc_opp2, "opp_after_math": acc_opp3,
        "math_after_math": acc_math3,
    }
    log(f"  >>> QA: {acc_qa1:.0%} → {acc_qa2:.0%} → {acc_qa3:.0%}")
    log(f"  >>> Opp: {acc_opp2:.0%} → {acc_opp3:.0%}")
    log(f"  >>> Math: {acc_math3:.0%}")

    if shutdown: return _summary(results)

    # 3c: Interleaved replay to prevent forgetting
    log("\n  --- Retention with replay: QA + replay while training Opp ---")
    def gen_opp_with_qa_replay():
        """70% opposites, 30% QA replay"""
        if random.random() < 0.3:
            return gen_qa_train()
        return gen_opp_train()

    state_replay, _ = run_exp("replay_qa_opp", gen_opp_with_qa_replay, 5000,
                               from_state=state1)  # start from QA-trained
    acc_qa_replay, _ = test_exp("qa_with_replay", gen_qa_train_pairs(), "word")
    acc_opp_replay, _ = test_exp("opp_with_replay", gen_opp_train_pairs(), "word")

    results["retention_replay"] = {
        "qa_with_replay": acc_qa_replay,
        "opp_with_replay": acc_opp_replay,
        "qa_without_replay": acc_qa2,  # from sequential training above
    }
    log(f"  >>> With replay: QA={acc_qa_replay:.0%}, Opp={acc_opp_replay:.0%}")
    log(f"  >>> Without replay: QA={acc_qa2:.0%}")
    log(f"  >>> Replay benefit for QA: {acc_qa_replay - acc_qa2:+.0%}")

    if shutdown: return _summary(results)

    # ══════════════════════════════════════════════════════════════════════
    # ROUND 4: SCALING — How many facts can coexist?
    # ══════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("ROUND 4: FACT SCALING")
    log("How many distinct Q&A facts can the model learn simultaneously?")
    log("=" * 70)

    for n_facts in [10, 20, 30, 50]:
        if shutdown:
            break
        log(f"\n  --- {n_facts} facts ---")
        data_fn = gen_scaled_facts(n_facts)
        state, _ = run_exp(f"scale_{n_facts}", data_fn, max(3000, n_facts * 100))
        test_pairs = tests_scaled_facts(n_facts)
        acc, det = test_exp(f"scale_{n_facts}_test", test_pairs, "word")
        results[f"scale_{n_facts}"] = {"accuracy": acc, "n_facts": n_facts}
        log(f"  >>> {n_facts} facts: {acc:.0%} ({sum(1 for _,_,_,m in det if m)}/{n_facts})")

    if shutdown: return _summary(results)

    # ══════════════════════════════════════════════════════════════════════
    # ROUND 5: MEGA MULTI-TASK — Everything together
    # ══════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("ROUND 5: MEGA MULTI-TASK (all skills + facts together)")
    log("=" * 70)

    all_generators = [
        gen_qa_train, gen_opp_train, gen_chat_train,
        gen_single_add, gen_two_digit_add,
        gen_scaled_facts(50),  # 50 world facts
    ]

    def gen_mega():
        return random.choice(all_generators)()

    state_mega, _ = run_exp("mega_multitask", gen_mega, 15000)

    # Test everything
    mega_results = {}
    log("\n  Per-skill results after mega multi-task:")

    acc, _ = test_exp("mega_qa_train", gen_qa_train_pairs(), "word")
    mega_results["qa_train"] = acc
    acc, _ = test_exp("mega_qa_held_out", gen_qa_test_pairs(), "word")
    mega_results["qa_held_out"] = acc
    acc, _ = test_exp("mega_opp_train", gen_opp_train_pairs(), "word")
    mega_results["opp_train"] = acc
    acc, _ = test_exp("mega_opp_held_out", gen_opp_test_pairs(), "word")
    mega_results["opp_held_out"] = acc
    acc, _ = test_exp("mega_chat_train", gen_chat_train_pairs(), "word")
    mega_results["chat_train"] = acc
    acc, _ = test_exp("mega_chat_held_out", gen_chat_test_pairs(), "word")
    mega_results["chat_held_out"] = acc
    acc, _ = test_exp("mega_add", tests_single_add(), "number")
    mega_results["add"] = acc
    acc, _ = test_exp("mega_2digit_add", tests_two_digit_add(), "number")
    mega_results["2digit_add"] = acc
    acc, _ = test_exp("mega_facts", tests_scaled_facts(50), "word")
    mega_results["world_facts"] = acc

    results["mega_multitask"] = mega_results

    # Test composition after mega training
    acc_comp_mega, _ = test_exp("mega_qa_opposites", tests_qa_opposite(), "word")
    mega_results["composition"] = acc_comp_mega
    log(f"  >>> Composition (Q&A about opposites): {acc_comp_mega:.0%}")

    # Also test zero-shot on completely new tasks
    acc_sub_mega, _ = test_exp("mega_sub_zeroshot", tests_subtraction(), "number")
    mega_results["sub_zeroshot"] = acc_sub_mega
    acc_mul_mega, _ = test_exp("mega_mul_zeroshot", tests_multiplication(), "number")
    mega_results["mul_zeroshot"] = acc_mul_mega

    if shutdown: return _summary(results)

    # ══════════════════════════════════════════════════════════════════════
    # ROUND 6: CURRICULUM BUILD-UP
    # ══════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("ROUND 6: INCREMENTAL CURRICULUM (add skills one by one with replay)")
    log("=" * 70)

    curriculum_stages = [
        ("stage1_qa", gen_qa_train, gen_qa_train_pairs, "word", 3000),
        ("stage2_opp", gen_opp_train, gen_opp_train_pairs, "word", 3000),
        ("stage3_chat", gen_chat_train, gen_chat_train_pairs, "word", 3000),
        ("stage4_math", gen_single_add, tests_single_add, "number", 3000),
        ("stage5_facts", gen_scaled_facts(30), lambda: tests_scaled_facts(30), "word", 5000),
    ]

    # Keep track of all learned generators for replay
    learned_generators = []
    curriculum_state = None  # Will hold the state as we build up
    curriculum_results = {}

    for stage_name, data_fn, test_fn, atype, steps in curriculum_stages:
        if shutdown:
            break

        learned_generators.append(data_fn)

        # Create mixed generator: 60% new skill, 40% replay of all previous
        all_gens = list(learned_generators)  # copy
        def make_mixed(new_fn, all_fns):
            def _gen():
                if len(all_fns) > 1 and random.random() < 0.4:
                    # Replay from any previous skill
                    return random.choice(all_fns[:-1])()
                return new_fn()
            return _gen

        mixed_fn = make_mixed(data_fn, all_gens)
        curriculum_state, _ = run_exp(stage_name, mixed_fn, steps,
                                       from_state=curriculum_state)

        # Test ALL learned skills so far
        log(f"\n  After {stage_name} — testing all learned skills:")
        stage_results = {}
        for prev_name, _, prev_test, prev_atype, _ in curriculum_stages:
            if prev_name > stage_name:  # haven't learned this yet
                break
            test_pairs = prev_test() if callable(prev_test) else prev_test
            acc, _ = test_exp(f"  {prev_name}", test_pairs, prev_atype)
            stage_results[prev_name] = acc

        curriculum_results[stage_name] = stage_results

    results["curriculum"] = curriculum_results

    # Final curriculum accuracy
    log("\n  Final curriculum state — all skills:")
    if curriculum_state:
        load_state(curriculum_state)
        final_curriculum = {}
        for name, _, test_fn, atype, _ in curriculum_stages:
            test_pairs = test_fn() if callable(test_fn) else test_fn
            acc, _ = test_exp(f"final_{name}", test_pairs, atype)
            final_curriculum[name] = acc
        # Test generalization
        acc, _ = test_exp("final_qa_held_out", gen_qa_test_pairs(), "word")
        final_curriculum["qa_held_out"] = acc
        acc, _ = test_exp("final_opp_held_out", gen_opp_test_pairs(), "word")
        final_curriculum["opp_held_out"] = acc
        acc, _ = test_exp("final_chat_held_out", gen_chat_test_pairs(), "word")
        final_curriculum["chat_held_out"] = acc
        acc, _ = test_exp("final_composition", tests_qa_opposite(), "word")
        final_curriculum["composition"] = acc
        results["final_curriculum"] = final_curriculum

    _summary(results)


def _summary(results):
    log(f"\n{'='*70}")
    log("FINAL SUMMARY — v3 Scaffolding Experiments")
    log(f"{'='*70}")

    if "qa_generalization" in results:
        r = results["qa_generalization"]
        log(f"\nGENERALIZATION:")
        log(f"  Q&A:       train={r['train_acc']:.0%}  held-out={r['test_acc']:.0%}  "
            f"(gen rate: {r['generalization_rate']:.0%})")
    if "opp_generalization" in results:
        r = results["opp_generalization"]
        log(f"  Opposites: train={r['train_acc']:.0%}  held-out={r['test_acc']:.0%}  "
            f"(gen rate: {r['generalization_rate']:.0%})")
    if "chat_generalization" in results:
        r = results["chat_generalization"]
        log(f"  Chat:      train={r['train_acc']:.0%}  held-out={r['test_acc']:.0%}  "
            f"(gen rate: {r['generalization_rate']:.0%})")
    if "math_transfer" in results:
        r = results["math_transfer"]
        log(f"  Math:      add(trained)={r['add_trained']:.0%}  "
            f"sub(zero-shot)={r['sub_zero_shot']:.0%}  mul(zero-shot)={r['mul_zero_shot']:.0%}")

    if "composition_qa_opp" in results:
        log(f"\nCOMPOSITION:")
        r = results["composition_qa_opp"]
        log(f"  Sequential (QA→Opp):  composed={r['composed_acc']:.0%}  "
            f"qa_kept={r['qa_retained']:.0%}  opp_kept={r['opp_retained']:.0%}")
    if "composition_joint" in results:
        r = results["composition_joint"]
        log(f"  Joint (QA+Opp):       composed={r['composed_acc']:.0%}  "
            f"qa_kept={r['qa_retained']:.0%}  opp_kept={r['opp_retained']:.0%}")
    if "composition_direct" in results:
        log(f"  Direct training:      {results['composition_direct']['accuracy']:.0%}")

    if "retention_chat_qa" in results:
        log(f"\nRETENTION:")
        r = results["retention_chat_qa"]
        log(f"  Chat→QA: chat {r['chat_before']:.0%}→{r['chat_after_qa']:.0%} "
            f"(retained {r['chat_retained']:.0%})")
    if "retention_chain" in results:
        r = results["retention_chain"]
        log(f"  QA→Opp→Math chain:")
        log(f"    QA:   {r['qa_after_qa']:.0%} → {r['qa_after_opp']:.0%} → {r['qa_after_math']:.0%}")
        log(f"    Opp:  {r['opp_after_opp']:.0%} → {r['opp_after_math']:.0%}")
        log(f"    Math: {r['math_after_math']:.0%}")
    if "retention_replay" in results:
        r = results["retention_replay"]
        log(f"  Replay benefit: QA with replay={r['qa_with_replay']:.0%} "
            f"vs without={r['qa_without_replay']:.0%} "
            f"(+{r['qa_with_replay'] - r['qa_without_replay']:.0%})")

    for n in [10, 20, 30, 50]:
        key = f"scale_{n}"
        if key in results:
            log(f"\nSCALING: {n} facts → {results[key]['accuracy']:.0%}")

    if "mega_multitask" in results:
        log(f"\nMEGA MULTI-TASK:")
        for k, v in results["mega_multitask"].items():
            log(f"  {k:20s}: {v:.0%}")

    if "final_curriculum" in results:
        log(f"\nFINAL CURRICULUM (incremental with replay):")
        for k, v in results["final_curriculum"].items():
            log(f"  {k:20s}: {v:.0%}")

    log(f"\n{'='*70}")
    log("v3 EXPERIMENTS COMPLETE")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
