"""
Hamner Pre-Training Script (Production)
========================================
- Streams from full FineWeb-Edu 10B token dataset (no caching cap)
- Auto-resumes from last checkpoint on crash/restart
- Generates sample text periodically so you can watch progress
- Saves checkpoints every N steps
- Cleans up old checkpoints automatically (keeps milestones)

Usage:
    python train.py --resume                    # resume from latest checkpoint
    python train.py --checkpoint path/to/ckpt   # resume from specific checkpoint
    python train.py --fresh                     # start fresh (will warn if checkpoints exist)
"""

import os
import sys
import json
import time
import math
import signal
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from model import HamnerModel, HamnerConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/pretrain_v2"
LOG_FILE = "logs/pretrain_v2.log"
METRICS_FILE = "logs/pretrain_v2_metrics.csv"
SAMPLES_FILE = "logs/pretrain_v2_samples.jsonl"

# Model config — 164M params, same as validated in concept experiments
# v2: CORRECT label handling (model.forward shifts internally, no pre-shifting)
# v1 was trained with double-shift bug for 122k steps (predicted 2-ahead!)
MODEL_CONFIG = dict(
    hidden_size=768,
    num_layers=20,
    num_attention_heads=12,
    num_kv_heads=4,
    num_experts=1,
    num_active_experts=1,
    expert_intermediate_size=2048,
    use_differential_attention=False,
    gradient_checkpointing=True,
    max_seq_len=1024,
    # vocab_size set from tokenizer
)

# Training hyperparameters
BATCH_SIZE = 24            # 24GB GPU with grad checkpointing
SEQ_LEN = 1024
LR = 2e-4
WARMUP_STEPS = 2000
MAX_STEPS = 400_000       # ~10B tokens (24 * 1024 * 400k = 9.8B)
CHECKPOINT_EVERY = 1000
SAMPLE_EVERY = 500        # generate sample text every N steps
LOG_EVERY = 50
KEEP_CHECKPOINTS = 10     # keep last N checkpoints + milestones every 10k

# Sample prompts for tracking progress
SAMPLE_PROMPTS = [
    "The meaning of life is",
    "Once upon a time there was a",
    "Hello! How are you doing today?",
    "The most important thing about technology is",
    "I think the best way to learn is",
    "When I was young, I used to",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg, log_file=LOG_FILE):
    """Log to both stdout and file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def log_metrics(step, loss, perplexity, lr, tokens_per_sec, tokens_total, elapsed_hours):
    """Append a row to the CSV metrics file for graphing."""
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    write_header = not os.path.exists(METRICS_FILE)
    with open(METRICS_FILE, "a") as f:
        if write_header:
            f.write("timestamp,step,loss,perplexity,learning_rate,tokens_per_sec,tokens_total,tokens_billions,elapsed_hours,phase\n")
        ts = datetime.datetime.now().isoformat()
        tokens_b = tokens_total / 1e9
        f.write(f"{ts},{step},{loss:.6f},{perplexity:.2f},{lr:.6e},{tokens_per_sec:.0f},{tokens_total},{tokens_b:.4f},{elapsed_hours:.4f},training_v2\n")


def log_samples(step, tokens_total, samples_dict):
    """Append sample generations to JSONL for tracking progress over time."""
    os.makedirs(os.path.dirname(SAMPLES_FILE), exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "step": step,
        "tokens_total": tokens_total,
        "tokens_billions": round(tokens_total / 1e9, 4),
        "samples": samples_dict,
    }
    with open(SAMPLES_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Data: True streaming from FineWeb-Edu (no cache cap)
# ---------------------------------------------------------------------------

class InfiniteStreamDataset:
    """
    Streams from FineWeb-Edu (10B tokens) mixed with personal data.
    Every batch has a chance of including personal YouTube transcript data,
    so the model learns David's voice alongside general language.
    """

    def __init__(self, tokenizer, seq_len=1024,
                 dataset_name="HuggingFaceFW/fineweb-edu",
                 dataset_config="sample-10BT",
                 personal_data_path="data/personal/training_samples.jsonl",
                 personal_mix_ratio=0.05):  # 5% personal data
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.token_buffer = []
        self.personal_mix_ratio = personal_mix_ratio
        self._load_personal_data(personal_data_path)
        self._init_stream()

    def _load_personal_data(self, path):
        """Load personal training samples (YouTube transcripts etc)."""
        import json
        self.personal_samples = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    sample = json.loads(line)
                    self.personal_samples.append(sample["text"])
            log(f"Loaded {len(self.personal_samples)} personal training samples")
        else:
            log(f"No personal data found at {path}")

    def _init_stream(self):
        from datasets import load_dataset
        log(f"Initializing data stream from {self.dataset_name}/{self.dataset_config}...")
        self.stream = iter(load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split="train",
            streaming=True,
        ))
        self.epoch = getattr(self, 'epoch', 0) + 1
        log(f"Data stream ready (epoch {self.epoch})")

    def _get_text(self):
        """Get next text, mixing personal data in at the configured ratio."""
        import random
        if self.personal_samples and random.random() < self.personal_mix_ratio:
            return random.choice(self.personal_samples)
        else:
            while True:
                try:
                    sample = next(self.stream)
                    text = sample.get("text", "")
                    if len(text.strip()) < 50:
                        continue
                    return text
                except StopIteration:
                    log("Dataset exhausted, restarting stream...")
                    self._init_stream()

    def get_batch(self, batch_size):
        """Get a batch of (input_ids, labels) tensors."""
        input_ids = []
        labels = []

        for _ in range(batch_size):
            # Fill buffer until we have enough
            while len(self.token_buffer) < self.seq_len:
                text = self._get_text()
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                tokens.append(self.tokenizer.eos_token_id or 0)
                self.token_buffer.extend(tokens)

            # Extract chunk
            chunk = self.token_buffer[:self.seq_len]
            self.token_buffer = self.token_buffer[self.seq_len:]

            # NO pre-shifting — model.forward() handles shift internally
            input_ids.append(torch.tensor(chunk, dtype=torch.long))
            labels.append(torch.tensor(chunk, dtype=torch.long))

        return torch.stack(input_ids), torch.stack(labels)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None

    # Check for latest.pt first
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return str(latest)

    # Otherwise find highest step checkpoint
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    if ckpts:
        return str(ckpts[-1])

    return None


def save_checkpoint(model, optimizer, scaler, config, step, loss, checkpoint_dir, is_milestone=False, tokens_total=0):
    """Save checkpoint and clean up old ones."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Strip _orig_mod. prefix from torch.compile'd models for clean checkpoints
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
    }

    # Save as step-numbered file
    ckpt_path = os.path.join(checkpoint_dir, f"step_{step:07d}.pt")
    torch.save(ckpt_data, ckpt_path)

    # Also save as latest.pt
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(ckpt_data, latest_path)

    log(f"  Checkpoint saved: {ckpt_path}")

    # Cleanup: keep milestones (every 10k) + last N regular checkpoints
    all_ckpts = sorted(Path(checkpoint_dir).glob("step_*.pt"))
    to_keep = set()

    # Keep milestones
    for c in all_ckpts:
        step_num = int(c.stem.split("_")[1])
        if step_num % 10000 == 0:
            to_keep.add(c)

    # Keep last N
    for c in all_ckpts[-KEEP_CHECKPOINTS:]:
        to_keep.add(c)

    # Delete the rest
    for c in all_ckpts:
        if c not in to_keep:
            c.unlink()


def load_checkpoint(path, device="cuda"):
    """Load checkpoint and return model, optimizer, scaler, step."""
    log(f"Loading checkpoint from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    config = HamnerConfig(**ckpt["config"])
    model = HamnerModel(config).to(device)

    # Strip _orig_mod. prefix from torch.compile'd checkpoints
    state_dict = ckpt["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1
    )
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    scaler = torch.amp.GradScaler("cuda")
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    step = ckpt.get("step", 0)
    loss = ckpt.get("avg_loss", float("inf"))
    tokens_total = ckpt.get("tokens_total", None)

    total_p, _ = model.count_parameters()
    log(f"Resumed: {total_p:,} params | step {step} | loss {loss:.4f}")

    return model, optimizer, scaler, config, step, tokens_total


# ---------------------------------------------------------------------------
# Text generation for progress tracking
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(model, tokenizer, prompts, device="cuda", max_tokens=60):
    """Generate sample text from prompts to show training progress."""
    model.eval()
    results = []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        try:
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.15,
                eos_token_id=tokenizer.eos_token_id or 0,
            )
            generated = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
            results.append(generated)
        except Exception as e:
            results.append(f"{prompt} [generation error: {e}]")

    model.train()
    return results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(resume_from=None, fresh=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("HAMNER PRE-TRAINING")
    log("=" * 70)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    # Check for existing checkpoint
    if resume_from is None and not fresh:
        resume_from = find_latest_checkpoint(CHECKPOINT_DIR)
        if resume_from:
            log(f"Found existing checkpoint: {resume_from}")

    # Initialize or resume
    saved_tokens = None
    if resume_from:
        model, optimizer, scaler, config, start_step, saved_tokens = load_checkpoint(resume_from, device)
        config.vocab_size = tokenizer.vocab_size  # ensure consistency
    else:
        log("Starting fresh training...")
        config = HamnerConfig(**MODEL_CONFIG, vocab_size=tokenizer.vocab_size)
        model = HamnerModel(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1
        )
        scaler = torch.amp.GradScaler("cuda")
        start_step = 0

        total_p, _ = model.count_parameters()
        log(f"Model: {total_p:,} params | {total_p*2/1e9:.2f}GB fp16")

    # Compile for speed (PyTorch 2.x)
    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile (first step will be slow)...")
        model = torch.compile(model)

    model.train()

    # Data stream
    data = InfiniteStreamDataset(
        tokenizer, seq_len=SEQ_LEN,
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
    )

    # Training state
    losses = []
    if saved_tokens is not None:
        tokens_total = saved_tokens
    else:
        tokens_total = start_step * BATCH_SIZE * SEQ_LEN
    start_time = time.time()

    # Graceful shutdown
    shutdown_requested = False
    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        log("Shutdown signal received, saving checkpoint...")
        shutdown_requested = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    log(f"Training from step {start_step} to {MAX_STEPS}")
    log(f"Batch size: {BATCH_SIZE} | Seq len: {SEQ_LEN} | LR: {LR}")
    log(f"Tokens so far: {tokens_total:,} | Target: ~{MAX_STEPS * BATCH_SIZE * SEQ_LEN / 1e9:.1f}B tokens")
    log("-" * 70)

    for step in range(start_step, MAX_STEPS):
        if shutdown_requested:
            break

        # Get batch
        input_ids, labels = data.get_batch(BATCH_SIZE)
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # LR schedule: linear warmup then cosine decay
        if step < WARMUP_STEPS:
            current_lr = LR * (step + 1) / WARMUP_STEPS
        else:
            progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
            current_lr = LR * 0.1 + LR * 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Forward + backward
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        losses.append(loss_val)
        tokens_total += BATCH_SIZE * SEQ_LEN

        # Log
        if (step + 1) % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            tps = (step - start_step + 1) * BATCH_SIZE * SEQ_LEN / elapsed
            perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow
            hours = elapsed / 3600
            tokens_b = tokens_total / 1e9
            pct = tokens_total / (MAX_STEPS * BATCH_SIZE * SEQ_LEN) * 100

            log(f"step {step+1:>7d} | loss {avg_loss:.4f} | ppl {perplexity:.1f} | "
                f"lr {current_lr:.2e} | {tps:.0f} tok/s | "
                f"{tokens_b:.2f}B tokens ({pct:.1f}%) | {hours:.1f}h")
            log_metrics(step + 1, avg_loss, perplexity, current_lr, tps, tokens_total, hours)

        # Generate samples
        if (step + 1) % SAMPLE_EVERY == 0:
            log("--- SAMPLE GENERATIONS ---")
            samples = generate_samples(model, tokenizer, SAMPLE_PROMPTS[:3], device)
            samples_dict = {}
            for i, sample in enumerate(samples):
                sample = sample[:300]
                log(f"  [{i+1}] {sample}")
                samples_dict[SAMPLE_PROMPTS[i]] = sample
            log("-" * 40)
            log_samples(step + 1, tokens_total, samples_dict)

        # Checkpoint
        if (step + 1) % CHECKPOINT_EVERY == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            save_checkpoint(
                model, optimizer, scaler, config, step + 1, avg_loss,
                CHECKPOINT_DIR, is_milestone=((step + 1) % 10000 == 0),
                tokens_total=tokens_total,
            )

        if shutdown_requested:
            break

    # Final save
    avg_loss = sum(losses[-100:]) / len(losses[-100:]) if losses else float("inf")
    save_checkpoint(model, optimizer, scaler, config, step + 1, avg_loss, CHECKPOINT_DIR, tokens_total=tokens_total)

    elapsed = time.time() - start_time
    log("=" * 70)
    log(f"TRAINING {'STOPPED' if shutdown_requested else 'COMPLETE'}")
    log(f"Final step: {step + 1} | Loss: {avg_loss:.4f} | Time: {elapsed/3600:.1f}h")
    log(f"Total tokens: {tokens_total:,} ({tokens_total/1e9:.2f}B)")
    log("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hamner Pre-Training")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from specific checkpoint")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore checkpoints)")
    args = parser.parse_args()

    if args.checkpoint:
        train(resume_from=args.checkpoint)
    elif args.fresh:
        train(fresh=True)
    else:
        train()  # auto-detects and resumes
