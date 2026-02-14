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

CHECKPOINT_DIR = "checkpoints/training"
LOG_FILE = "logs/training.log"

# Model config (winner from tournament: v02_dense_medium)
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
BATCH_SIZE = 16
SEQ_LEN = 1024
LR = 3e-4
WARMUP_STEPS = 2000
MAX_STEPS = 300_000       # ~5B tokens
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
            while len(self.token_buffer) < self.seq_len + 1:
                text = self._get_text()
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                tokens.append(self.tokenizer.eos_token_id or 0)
                self.token_buffer.extend(tokens)

            # Extract chunk
            chunk = self.token_buffer[:self.seq_len + 1]
            self.token_buffer = self.token_buffer[self.seq_len:]

            input_ids.append(torch.tensor(chunk[:-1], dtype=torch.long))
            labels.append(torch.tensor(chunk[1:], dtype=torch.long))

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


def save_checkpoint(model, optimizer, scaler, config, step, loss, checkpoint_dir, is_milestone=False):
    """Save checkpoint and clean up old ones."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt_data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config.__dict__,
        "step": step,
        "avg_loss": loss,
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
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

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

    total_p, _ = model.count_parameters()
    log(f"Resumed: {total_p:,} params | step {step} | loss {loss:.4f}")

    return model, optimizer, scaler, config, step


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
        # Also check the old tournament checkpoint dir
        if resume_from is None:
            resume_from = find_latest_checkpoint("checkpoints/final")
        if resume_from:
            log(f"Found existing checkpoint: {resume_from}")

    # Initialize or resume
    if resume_from:
        model, optimizer, scaler, config, start_step = load_checkpoint(resume_from, device)
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

    model.train()

    # Data stream
    data = InfiniteStreamDataset(
        tokenizer, seq_len=SEQ_LEN,
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
    )

    # Training state
    losses = []
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

        # Generate samples
        if (step + 1) % SAMPLE_EVERY == 0:
            log("--- SAMPLE GENERATIONS ---")
            samples = generate_samples(model, tokenizer, SAMPLE_PROMPTS[:3], device)
            for i, sample in enumerate(samples):
                # Truncate long outputs
                sample = sample[:300]
                log(f"  [{i+1}] {sample}")
            log("-" * 40)

        # Checkpoint
        if (step + 1) % CHECKPOINT_EVERY == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            save_checkpoint(
                model, optimizer, scaler, config, step + 1, avg_loss,
                CHECKPOINT_DIR, is_milestone=((step + 1) % 10000 == 0),
            )

        if shutdown_requested:
            break

    # Final save
    avg_loss = sum(losses[-100:]) / len(losses[-100:]) if losses else float("inf")
    save_checkpoint(model, optimizer, scaler, config, step + 1, avg_loss, CHECKPOINT_DIR)

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
