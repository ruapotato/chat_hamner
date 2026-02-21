"""
Chat Pretraining: Continue pretrained base on dialogue-heavy data mix
=====================================================================
V4: Updated data mix with SmolTalk, FineWeb-Edu, DCLM, personal voice,
and synthetic tasks. Continues from anneal or base pretrain checkpoint.

Usage:
    python train_chat_pretrain.py                    # start from anneal/base checkpoint
    python train_chat_pretrain.py --resume           # resume from checkpoint
    python train_chat_pretrain.py --checkpoint X     # start from specific base
"""

import os
import sys
import json
import time
import math
import random
import signal
import datetime
import torch
import torch.nn.functional as F
from pathlib import Path

from model import HamnerModel, HamnerConfig
from synthetic_tasks import SyntheticTaskGenerator


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_CHECKPOINT = "checkpoints/pretrain_v4_anneal/latest.pt"
BASE_CHECKPOINT_FALLBACKS = [
    "checkpoints/pretrain_v4/latest.pt",
    "checkpoints/pretrain_v2/latest.pt",
]
CHECKPOINT_DIR = "checkpoints/chat_pretrain"
LOG_FILE = "logs/chat_pretrain_v4.log"
METRICS_FILE = "logs/chat_pretrain_v4_metrics.csv"
SAMPLES_FILE = "logs/chat_pretrain_v4_samples.jsonl"

# V4 Data mix — SmolTalk-heavy with diverse knowledge sources
DATA_RATIOS = {
    "smoltalk": 0.40,     # high-quality dialogue (SmolTalk as raw text, no masking)
    "fineweb": 0.25,      # maintain general knowledge
    "dclm": 0.15,         # general web text diversity
    "personal": 0.10,     # learn personal voice
    "synthetic": 0.10,    # reasoning tasks
}

# Training hyperparameters
BATCH_SIZE = 24
SEQ_LEN = 1024
LR = 5e-5           # moderate — don't destroy pretrain knowledge
LR_MIN = 5e-6       # cosine floor
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
MAX_STEPS = 40000    # ~1B tokens

# Logging / checkpointing
CHECKPOINT_EVERY = 5000
SAMPLE_EVERY = 1000
LOG_EVERY = 50
VAL_EVERY = 500

SAMPLE_PROMPTS = [
    # Chat format
    "<|system|>\nYou are Al Hamner.\n<|user|>\nHello!\n<|assistant|>\n",
    "<|system|>\nYou are Al Hamner.\n<|user|>\nWhat do you think about programming?\n<|assistant|>\n",
    "<|user|>\nTell me about yourself\n<|assistant|>\n",
    "<|user|>\nWhat do you think about AI?\n<|assistant|>\n",
    # Plain text
    "The most important thing about technology is",
    "Once upon a time",
    "In computer science, a hash table is",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def log_metrics(step, loss, perplexity, lr, tokens_per_sec, tokens_total, elapsed_hours):
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    write_header = not os.path.exists(METRICS_FILE)
    with open(METRICS_FILE, "a") as f:
        if write_header:
            f.write("timestamp,step,loss,perplexity,learning_rate,tokens_per_sec,tokens_total,elapsed_hours\n")
        ts = datetime.datetime.now().isoformat()
        f.write(f"{ts},{step},{loss:.6f},{perplexity:.2f},{lr:.6e},{tokens_per_sec:.0f},{tokens_total},{elapsed_hours:.4f}\n")


# ---------------------------------------------------------------------------
# Data mixer (reuses pattern from train_v3.py)
# ---------------------------------------------------------------------------

class DataMixer:
    """Mixes data from multiple sources with fixed ratios."""

    def __init__(self, tokenizer, seq_len=1024,
                 personal_data_path="data/personal/training_samples.jsonl",
                 smoltalk_path="data/sft_smoltalk.jsonl"):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.synthetic_gen = SyntheticTaskGenerator(seed=42)
        self.synthetic_gen.set_difficulty(3)  # moderate difficulty

        self.token_buffers = {name: [] for name in DATA_RATIOS}
        self.ratios = DATA_RATIOS

        self._load_local_data(personal_data_path, "personal")
        self._load_local_data(smoltalk_path, "smoltalk")
        self._init_streams()

    def _load_local_data(self, path, name):
        samples = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    sample = json.loads(line)
                    samples.append(sample["text"])
            log(f"Loaded {len(samples)} {name} samples from {path}")
        else:
            log(f"No {name} data found at {path}")
        setattr(self, f"{name}_samples", samples)

    def _init_streams(self):
        from datasets import load_dataset
        log("Initializing FineWeb-Edu stream...")
        self.fineweb_stream = iter(load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT",
            split="train", streaming=True,
        ))
        log("Initializing DCLM stream...")
        self.dclm_stream = iter(load_dataset(
            "mlfoundations/dclm-baseline-1.0",
            split="train", streaming=True,
        ))
        log("Data streams ready")

    def _restart_stream(self, name):
        from datasets import load_dataset
        log(f"{name} exhausted, restarting...")
        if name == "fineweb":
            self.fineweb_stream = iter(load_dataset(
                "HuggingFaceFW/fineweb-edu", name="sample-10BT",
                split="train", streaming=True,
            ))
        elif name == "dclm":
            self.dclm_stream = iter(load_dataset(
                "mlfoundations/dclm-baseline-1.0",
                split="train", streaming=True,
            ))

    def _get_text(self, source):
        if source == "synthetic":
            return self.synthetic_gen.get_random_task()
        elif source == "fineweb":
            while True:
                try:
                    sample = next(self.fineweb_stream)
                    text = sample.get("text", "")
                    if len(text.strip()) >= 50:
                        return text
                except StopIteration:
                    self._restart_stream("fineweb")
        elif source == "dclm":
            while True:
                try:
                    sample = next(self.dclm_stream)
                    text = sample.get("text", "")
                    if len(text.strip()) >= 50:
                        return text
                except StopIteration:
                    self._restart_stream("dclm")
        elif source == "personal":
            if self.personal_samples:
                return random.choice(self.personal_samples)
            return self._get_text("fineweb")
        elif source == "smoltalk":
            if self.smoltalk_samples:
                return random.choice(self.smoltalk_samples)
            return self._get_text("fineweb")
        return ""

    def _choose_source(self):
        r = random.random()
        cumulative = 0.0
        for source, ratio in self.ratios.items():
            cumulative += ratio
            if r < cumulative:
                return source
        return list(self.ratios.keys())[-1]

    def _fill_buffer(self, source):
        text = self._get_text(source)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        tokens.append(self.tokenizer.eos_token_id or 0)
        self.token_buffers[source].extend(tokens)

    def get_batch(self, batch_size):
        input_ids = []
        labels = []

        for _ in range(batch_size):
            source = self._choose_source()
            buf = self.token_buffers[source]

            while len(buf) < self.seq_len:
                self._fill_buffer(source)
                buf = self.token_buffers[source]

            chunk = buf[:self.seq_len]
            self.token_buffers[source] = buf[self.seq_len:]

            input_ids.append(torch.tensor(chunk, dtype=torch.long))
            labels.append(torch.tensor(chunk, dtype=torch.long))

        return torch.stack(input_ids), torch.stack(labels)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scaler, config, step, loss, tokens_total):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    raw_state = model.state_dict()
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}

    ckpt_data = {
        "model_state_dict": clean_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config.__dict__,
        "step": step,
        "avg_loss": loss,
        "tokens_total": tokens_total,
        "timestamp": datetime.datetime.now().isoformat(),
        "training_type": "chat_pretrain",
    }

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{step:07d}.pt")
    torch.save(ckpt_data, ckpt_path)

    latest_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    torch.save(ckpt_data, latest_path)
    log(f"  Checkpoint saved: {ckpt_path}")

    # Keep last 5 + every 10k
    all_ckpts = sorted(Path(CHECKPOINT_DIR).glob("step_*.pt"))
    to_keep = set(all_ckpts[-5:])
    for c in all_ckpts:
        step_num = int(c.stem.split("_")[1])
        if step_num % 10000 == 0:
            to_keep.add(c)
    for c in all_ckpts:
        if c not in to_keep:
            c.unlink()


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(model, tokenizer, prompts, device="cuda", max_tokens=100):
    model.eval()
    results = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        try:
            output = model.generate(
                input_ids, max_new_tokens=max_tokens,
                temperature=0.7, top_k=40, top_p=0.9,
                repetition_penalty=1.15,
                eos_token_id=tokenizer.eos_token_id or 0,
            )
            text = tokenizer.decode(output[0].tolist(), skip_special_tokens=False)
            # Trim at turn boundaries for chat prompts
            for stop in ['<|user|>', '<|system|>', '<|endoftext|>']:
                idx = text.find(stop, len(prompt))
                if idx != -1:
                    text = text[:idx]
            results.append(text.strip()[:300])
        except Exception as e:
            results.append(f"{prompt} [error: {e}]")
    model.train()
    return results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(base_checkpoint=None, resume=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("CHAT PRETRAINING V4 — Continue pretrained base on dialogue-heavy mix")
    log("=" * 70)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    # Load model
    start_step = 0
    tokens_total = 0

    if resume:
        ckpt_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
        if os.path.exists(ckpt_path):
            log(f"Resuming from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            config = HamnerConfig(**ckpt["config"])
            model = HamnerModel(config).to(device)
            cleaned = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
            model.load_state_dict(cleaned, strict=True)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY
            )
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scaler = torch.amp.GradScaler("cuda")
            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])

            start_step = ckpt.get("step", 0)
            tokens_total = ckpt.get("tokens_total", 0)
            total_p, _ = model.count_parameters()
            log(f"Resumed: {total_p:,} params | step {start_step} | loss {ckpt.get('avg_loss', '?'):.4f}")
        else:
            log("No checkpoint found, starting from base")
            resume = False

    if not resume:
        ckpt_path = base_checkpoint or BASE_CHECKPOINT
        if not os.path.exists(ckpt_path):
            for fallback in BASE_CHECKPOINT_FALLBACKS:
                if os.path.exists(fallback):
                    ckpt_path = fallback
                    log(f"Using fallback checkpoint: {ckpt_path}")
                    break
            else:
                log("ERROR: No base checkpoint found. Run train_pretrain.py first.")
                sys.exit(1)
        log(f"Loading base model from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = HamnerConfig(**ckpt["config"])
        model = HamnerModel(config).to(device)
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(cleaned, strict=True)

        total_p, _ = model.count_parameters()
        base_step = ckpt.get("step", 0)
        base_loss = ckpt.get("avg_loss", float("inf"))
        log(f"Base model: {total_p:,} params | pretrain step {base_step} | loss {base_loss:.4f}")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY
        )
        scaler = torch.amp.GradScaler("cuda")

    # Compile
    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Data mixer
    mixer = DataMixer(tokenizer, seq_len=SEQ_LEN,
                      smoltalk_path="data/sft_smoltalk.jsonl")

    log(f"\nData ratios: {DATA_RATIOS}")
    log(f"Training: {MAX_STEPS} steps | Batch {BATCH_SIZE} | Seq {SEQ_LEN}")
    log(f"LR: {LR} → {LR_MIN} (cosine) | Warmup: {WARMUP_STEPS} steps")
    log(f"Checkpoint every {CHECKPOINT_EVERY} | Samples every {SAMPLE_EVERY}")
    log(f"Expected tokens: ~{MAX_STEPS * BATCH_SIZE * SEQ_LEN / 1e9:.1f}B")
    log("-" * 70)

    # Training
    model.train()
    losses = []
    start_time = time.time()

    shutdown_requested = False
    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        log("Shutdown signal received, will save and exit...")
        shutdown_requested = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    for step in range(start_step, MAX_STEPS):
        if shutdown_requested:
            break

        # LR schedule: warmup then cosine decay
        if step < WARMUP_STEPS:
            current_lr = LR * (step + 1) / WARMUP_STEPS
        else:
            progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
            current_lr = LR_MIN + (LR - LR_MIN) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Get batch
        input_ids, labels = mixer.get_batch(BATCH_SIZE)
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward + backward
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]

        if torch.isnan(loss):
            log(f"NaN loss at step {step}, skipping")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        losses.append(loss_val)
        tokens_total += BATCH_SIZE * SEQ_LEN

        # Log
        if (step + 1) % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            ppl = min(math.exp(avg_loss), 99999)
            tps = tokens_total / elapsed if elapsed > 0 else 0
            hours = elapsed / 3600
            tokens_b = tokens_total / 1e9

            log(f"step {step+1:>6d} | loss {avg_loss:.4f} | ppl {ppl:.1f} | "
                f"lr {current_lr:.2e} | {tps:.0f} tok/s | {tokens_b:.2f}B | {hours:.1f}h")
            log_metrics(step + 1, avg_loss, ppl, current_lr, tps, tokens_total, hours)

        # Generate samples
        if (step + 1) % SAMPLE_EVERY == 0:
            log("--- SAMPLE GENERATIONS ---")
            samples = generate_samples(model, tokenizer, SAMPLE_PROMPTS, device)
            for i, text in enumerate(samples):
                log(f"  [{i+1}] {text[:200]}")
            log("-" * 40)

        # Checkpoint
        if (step + 1) % CHECKPOINT_EVERY == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            save_checkpoint(model, optimizer, scaler, config, step + 1, avg_loss, tokens_total)

    # Final save
    elapsed = time.time() - start_time
    if losses:
        final_loss = sum(losses[-100:]) / len(losses[-100:])
        save_checkpoint(model, optimizer, scaler, config, step + 1, final_loss, tokens_total)

    log("=" * 70)
    status = "STOPPED" if shutdown_requested else "COMPLETE"
    log(f"CHAT PRETRAINING {status}")
    log(f"Steps: {step + 1} | Tokens: {tokens_total/1e9:.2f}B | Time: {elapsed/3600:.1f}h")
    if losses:
        log(f"Final loss: {sum(losses[-100:]) / len(losses[-100:]):.4f}")
    log("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chat Pretraining")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to base checkpoint")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest chat_pretrain checkpoint")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override max steps")
    args = parser.parse_args()

    if args.steps is not None:
        MAX_STEPS = args.steps

    train(base_checkpoint=args.checkpoint, resume=args.resume)
