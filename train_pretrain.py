"""
Hamner V4 Pretraining Script — Multi-Source Curriculum
======================================================
SmolLM2-inspired massive pretraining with data diversity.

Two stages:
  base:   FineWeb-Edu 60% + DCLM 40%  → 10B tokens
  anneal: FineWeb 40% + DCLM 20% + Code 25% + Math 15%  → 3B tokens (linear LR decay)

Supports both fresh training and resuming from checkpoint.

Usage:
    python train_pretrain.py                           # fresh base training
    python train_pretrain.py --resume                  # resume from latest checkpoint
    python train_pretrain.py --stage anneal             # code+math annealing
    python train_pretrain.py --stage anneal --resume    # resume annealing
    python train_pretrain.py --checkpoint path/to/ckpt  # resume from specific checkpoint
"""

import os
import sys
import json
import time
import math
import signal
import random
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from model import HamnerModel, HamnerConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR_BASE = "checkpoints/pretrain_v4"
CHECKPOINT_DIR_ANNEAL = "checkpoints/pretrain_v4_anneal"
LOG_DIR = "logs"

# Local code corpus (Debian source packages, DFSG-free)
DEBIAN_CODE_PATH = "data/debian_code.jsonl"

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
)

# --- Base stage hyperparameters ---
BASE_BATCH_SIZE = 24
BASE_SEQ_LEN = 1024
BASE_LR = 2e-4
BASE_WARMUP_STEPS = 2000
BASE_MAX_STEPS = 400_000   # ~10B tokens
BASE_DECAY_FRACTION = 0.10  # last 10% of training decays LR

BASE_DATA_RATIOS = {
    "fineweb": 0.60,
    "dclm": 0.40,
}

# --- Anneal stage hyperparameters ---
ANNEAL_BATCH_SIZE = 24
ANNEAL_SEQ_LEN = 1024
ANNEAL_MAX_STEPS = 122_000  # ~3B tokens
# LR starts from checkpoint's current LR and decays linearly to 0

ANNEAL_DATA_RATIOS = {
    "fineweb": 0.40,
    "dclm": 0.20,
    "code": 0.25,
    "math": 0.15,
}

# --- Common ---
CHECKPOINT_EVERY = 1000
SAMPLE_EVERY = 500
LOG_EVERY = 50
VAL_EVERY = 500
KEEP_CHECKPOINTS = 10

SAMPLE_PROMPTS = [
    "The meaning of life is",
    "Once upon a time there was a",
    "Hello! How are you doing today?",
    "The most important thing about technology is",
    "def fibonacci(n):",
    "The derivative of x^2 is",
]

# AdamW config (SmolLM2 recipe)
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def make_log_paths(stage):
    return {
        "log": f"{LOG_DIR}/pretrain_v4_{stage}.log",
        "metrics": f"{LOG_DIR}/pretrain_v4_{stage}_metrics.csv",
        "samples": f"{LOG_DIR}/pretrain_v4_{stage}_samples.jsonl",
    }


LOG_PATHS = {}


def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    log_file = LOG_PATHS.get("log")
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")


def log_metrics(step, loss, perplexity, lr, tokens_per_sec, tokens_total,
                elapsed_hours, val_loss=None):
    metrics_file = LOG_PATHS.get("metrics")
    if not metrics_file:
        return
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    write_header = not os.path.exists(metrics_file)
    with open(metrics_file, "a") as f:
        if write_header:
            f.write("timestamp,step,loss,perplexity,learning_rate,tokens_per_sec,"
                    "tokens_total,tokens_billions,elapsed_hours,val_loss\n")
        ts = datetime.datetime.now().isoformat()
        tokens_b = tokens_total / 1e9
        val_str = f"{val_loss:.6f}" if val_loss is not None else ""
        f.write(f"{ts},{step},{loss:.6f},{perplexity:.2f},{lr:.6e},"
                f"{tokens_per_sec:.0f},{tokens_total},{tokens_b:.4f},"
                f"{elapsed_hours:.4f},{val_str}\n")


def log_samples(step, tokens_total, samples_dict):
    samples_file = LOG_PATHS.get("samples")
    if not samples_file:
        return
    os.makedirs(os.path.dirname(samples_file), exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "step": step,
        "tokens_total": tokens_total,
        "tokens_billions": round(tokens_total / 1e9, 4),
        "samples": samples_dict,
    }
    with open(samples_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Data: Multi-source streaming
# ---------------------------------------------------------------------------

class MultiSourceStreamer:
    """Streams from multiple HuggingFace datasets with configurable ratios."""

    def __init__(self, tokenizer, seq_len=1024, ratios=None,
                 personal_data_path="data/personal/training_samples.jsonl",
                 personal_mix_ratio=0.05):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.ratios = ratios or BASE_DATA_RATIOS
        self.personal_mix_ratio = personal_mix_ratio

        self.token_buffers = {name: [] for name in self.ratios}
        self.streams = {}
        self._load_personal_data(personal_data_path)
        self._init_streams()

    def _load_personal_data(self, path):
        self.personal_samples = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    sample = json.loads(line)
                    self.personal_samples.append(sample["text"])
            log(f"Loaded {len(self.personal_samples)} personal training samples")

    def _local_jsonl_stream(self, path):
        """Yield samples from a local JSONL file, looping indefinitely."""
        while True:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            log(f"Local file {path} exhausted, restarting...")

    def _init_streams(self):
        from datasets import load_dataset
        source_configs = {
            "fineweb": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train"),
            "dclm": ("mlfoundations/dclm-baseline-1.0", None, "train"),
            "math": ("HuggingFaceTB/finemath", "finemath-4plus", "train"),
        }

        for name in self.ratios:
            if name == "code":
                # Load from local Debian code corpus (DFSG-free)
                if not os.path.exists(DEBIAN_CODE_PATH):
                    log(f"ERROR: Code corpus not found at {DEBIAN_CODE_PATH}")
                    log("Run: python prepare_debian_code.py")
                    sys.exit(1)
                self.streams["code"] = self._local_jsonl_stream(DEBIAN_CODE_PATH)
                log(f"Initialized code stream from local file: {DEBIAN_CODE_PATH}")
                continue
            if name not in source_configs:
                continue
            dataset_name, config_name, split = source_configs[name]
            log(f"Initializing {name} stream: {dataset_name}"
                f"{f'/{config_name}' if config_name else ''}...")
            kwargs = {"split": split, "streaming": True}
            if config_name:
                kwargs["name"] = config_name
            self.streams[name] = iter(load_dataset(dataset_name, **kwargs))
        log("All data streams ready")

    def _restart_stream(self, name):
        if name == "code":
            log("code stream restarted from local file")
            self.streams["code"] = self._local_jsonl_stream(DEBIAN_CODE_PATH)
            return
        from datasets import load_dataset
        source_configs = {
            "fineweb": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train"),
            "dclm": ("mlfoundations/dclm-baseline-1.0", None, "train"),
            "math": ("HuggingFaceTB/finemath", "finemath-4plus", "train"),
        }
        log(f"{name} stream exhausted, restarting...")
        dataset_name, config_name, split = source_configs[name]
        kwargs = {"split": split, "streaming": True}
        if config_name:
            kwargs["name"] = config_name
        self.streams[name] = iter(load_dataset(dataset_name, **kwargs))

    def _get_text(self, source):
        # Small chance of personal data injection regardless of source
        if self.personal_samples and random.random() < self.personal_mix_ratio:
            return random.choice(self.personal_samples)

        max_retries = 10
        retry_delay = 5  # seconds, doubles each retry
        retries = 0

        while True:
            try:
                sample = next(self.streams[source])
                text = sample.get("text", sample.get("content", ""))
                if len(text.strip()) < 50:
                    continue
                retries = 0  # reset on success
                return text
            except StopIteration:
                self._restart_stream(source)
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    log(f"FATAL: {source} stream failed {max_retries} times, "
                        f"last error: {e}")
                    raise
                wait = retry_delay * (2 ** (retries - 1))
                log(f"WARNING: {source} stream error (attempt {retries}/"
                    f"{max_retries}): {e}")
                log(f"  Restarting stream in {wait}s...")
                import time
                time.sleep(wait)
                self._restart_stream(source)

    def _choose_source(self):
        r = random.random()
        cumulative = 0.0
        for source, ratio in self.ratios.items():
            cumulative += ratio
            if r < cumulative:
                return source
        return list(self.ratios.keys())[-1]

    def get_batch(self, batch_size):
        input_ids = []
        labels = []

        for _ in range(batch_size):
            source = self._choose_source()
            buf = self.token_buffers[source]

            while len(buf) < self.seq_len:
                text = self._get_text(source)
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                tokens.append(self.tokenizer.eos_token_id or 0)
                buf.extend(tokens)
                self.token_buffers[source] = buf

            chunk = buf[:self.seq_len]
            self.token_buffers[source] = buf[self.seq_len:]

            input_ids.append(torch.tensor(chunk, dtype=torch.long))
            labels.append(torch.tensor(chunk, dtype=torch.long))

        return torch.stack(input_ids), torch.stack(labels)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationSet:
    """Small held-out set from FineWeb for tracking generalization."""

    def __init__(self, tokenizer, seq_len=1024, n_batches=10, batch_size=24):
        from datasets import load_dataset
        self.batches = []
        log("Building validation set from FineWeb-Edu...")

        stream = iter(load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT",
            split="train", streaming=True,
        ))
        # Skip ahead to get different data than training
        for _ in range(50000):
            try:
                next(stream)
            except StopIteration:
                break

        for _ in range(n_batches):
            input_ids = []
            for _ in range(batch_size):
                buf = []
                while len(buf) < seq_len:
                    sample = next(stream)
                    text = sample.get("text", "")
                    if len(text.strip()) < 50:
                        continue
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    tokens.append(tokenizer.eos_token_id or 0)
                    buf.extend(tokens)
                chunk = buf[:seq_len]
                input_ids.append(torch.tensor(chunk, dtype=torch.long))
            self.batches.append(torch.stack(input_ids))

        log(f"Validation set: {n_batches} batches of {batch_size} sequences")

    @torch.no_grad()
    def evaluate(self, model, device="cuda"):
        model.eval()
        total_loss = 0.0
        for batch in self.batches:
            batch = batch.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(batch, labels=batch)
                total_loss += outputs["loss"].item()
        model.train()
        return total_loss / len(self.batches)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def find_latest_checkpoint(checkpoint_dir):
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return str(latest)
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    if ckpts:
        return str(ckpts[-1])
    return None


def save_checkpoint(model, optimizer, scaler, config, step, loss,
                    checkpoint_dir, tokens_total=0, stage="base",
                    extra=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
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
        "training_type": f"pretrain_v4_{stage}",
    }
    if extra:
        ckpt_data.update(extra)

    ckpt_path = os.path.join(checkpoint_dir, f"step_{step:07d}.pt")
    torch.save(ckpt_data, ckpt_path)
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(ckpt_data, latest_path)
    log(f"  Checkpoint saved: {ckpt_path}")

    # Cleanup: keep milestones (every 10k) + last N
    all_ckpts = sorted(Path(checkpoint_dir).glob("step_*.pt"))
    to_keep = set()
    for c in all_ckpts:
        step_num = int(c.stem.split("_")[1])
        if step_num % 10000 == 0:
            to_keep.add(c)
    for c in all_ckpts[-KEEP_CHECKPOINTS:]:
        to_keep.add(c)
    for c in all_ckpts:
        if c not in to_keep:
            c.unlink()


def load_checkpoint(path, device="cuda"):
    log(f"Loading checkpoint from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    config = HamnerConfig(**ckpt["config"])
    model = HamnerModel(config).to(device)

    state_dict = ckpt["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=BASE_LR, betas=BETAS, weight_decay=WEIGHT_DECAY
    )
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    scaler = torch.amp.GradScaler("cuda")
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    step = ckpt.get("step", 0)
    loss = ckpt.get("avg_loss", float("inf"))
    tokens_total = ckpt.get("tokens_total", 0)

    total_p, _ = model.count_parameters()
    log(f"Resumed: {total_p:,} params | step {step} | loss {loss:.4f} | "
        f"tokens {tokens_total/1e9:.2f}B")

    return model, optimizer, scaler, config, step, tokens_total


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(model, tokenizer, prompts, device="cuda", max_tokens=60):
    model.eval()
    results = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        try:
            output = model.generate(
                input_ids, max_new_tokens=max_tokens,
                temperature=0.8, top_k=40, top_p=0.9,
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
# LR Schedules
# ---------------------------------------------------------------------------

def wsd_lr(step, max_steps, peak_lr, warmup_steps, decay_fraction):
    """Warmup-Stable-Decay schedule (SmolLM2 style).
    - Linear warmup for warmup_steps
    - Stable at peak_lr for most of training
    - Cosine decay in the last decay_fraction of training
    - Min LR = peak_lr * 0.1
    """
    min_lr = peak_lr * 0.1
    decay_start = int(max_steps * (1 - decay_fraction))

    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    elif step < decay_start:
        return peak_lr
    else:
        progress = (step - decay_start) / max(1, max_steps - decay_start)
        return min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def linear_decay_lr(step, max_steps, start_lr):
    """Linear decay from start_lr to 0 over max_steps (for annealing)."""
    return start_lr * max(0, 1.0 - step / max_steps)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(stage="base", resume_from=None, fresh=False):
    global LOG_PATHS
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Stage-specific config
    if stage == "base":
        checkpoint_dir = CHECKPOINT_DIR_BASE
        batch_size = BASE_BATCH_SIZE
        seq_len = BASE_SEQ_LEN
        max_steps = BASE_MAX_STEPS
        data_ratios = BASE_DATA_RATIOS
    elif stage == "anneal":
        checkpoint_dir = CHECKPOINT_DIR_ANNEAL
        batch_size = ANNEAL_BATCH_SIZE
        seq_len = ANNEAL_SEQ_LEN
        max_steps = ANNEAL_MAX_STEPS
        data_ratios = ANNEAL_DATA_RATIOS
    else:
        raise ValueError(f"Unknown stage: {stage}. Use 'base' or 'anneal'.")

    LOG_PATHS = make_log_paths(stage)

    log("=" * 70)
    log(f"HAMNER V4 PRETRAINING — Stage: {stage.upper()}")
    log("=" * 70)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    # Resolve checkpoint
    if resume_from is None and not fresh:
        resume_from = find_latest_checkpoint(checkpoint_dir)
        if resume_from:
            log(f"Found existing checkpoint: {resume_from}")

    # For anneal stage, if no anneal checkpoint, look for base checkpoint
    if stage == "anneal" and resume_from is None:
        base_ckpt = find_latest_checkpoint(CHECKPOINT_DIR_BASE)
        if base_ckpt:
            log(f"Anneal: starting from base checkpoint: {base_ckpt}")
            resume_from = base_ckpt
        else:
            log("ERROR: Anneal stage requires a base pretrain checkpoint.")
            log("Run base stage first: python train_pretrain.py --stage base")
            sys.exit(1)

    # Initialize or resume
    if resume_from:
        model, optimizer, scaler, config, start_step, tokens_total = \
            load_checkpoint(resume_from, device)
        config.vocab_size = tokenizer.vocab_size
        # If loading base checkpoint for anneal, reset step counter
        if stage == "anneal" and "anneal" not in (resume_from or ""):
            log(f"Starting anneal from base model (resetting step to 0)")
            start_step = 0
    else:
        if stage == "anneal":
            log("ERROR: Anneal stage requires a base checkpoint.")
            sys.exit(1)
        log("Starting fresh training...")
        config = HamnerConfig(**MODEL_CONFIG, vocab_size=tokenizer.vocab_size)
        model = HamnerModel(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=BASE_LR, betas=BETAS, weight_decay=WEIGHT_DECAY
        )
        scaler = torch.amp.GradScaler("cuda")
        start_step = 0
        tokens_total = 0

        total_p, _ = model.count_parameters()
        log(f"Model: {total_p:,} params | {total_p*2/1e9:.2f}GB fp16")

    # Compile for speed
    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile (first step will be slow)...")
        model = torch.compile(model)

    model.train()

    # Data stream
    data = MultiSourceStreamer(
        tokenizer, seq_len=seq_len, ratios=data_ratios,
    )

    # Validation set
    val_set = ValidationSet(tokenizer, seq_len=seq_len, n_batches=5, batch_size=batch_size)

    # Anneal LR setup
    anneal_start_lr = BASE_LR
    if stage == "anneal":
        # Get current LR from optimizer
        anneal_start_lr = optimizer.param_groups[0]["lr"]
        if anneal_start_lr < 1e-6:
            anneal_start_lr = BASE_LR * 0.1  # reasonable fallback
        log(f"Anneal starting LR: {anneal_start_lr:.2e}")

    # Training state
    losses = []
    start_time = time.time()
    best_val_loss = float("inf")

    # Graceful shutdown
    shutdown_requested = False
    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        log("Shutdown signal received, saving checkpoint...")
        shutdown_requested = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    log(f"\nStage: {stage}")
    log(f"Data ratios: {data_ratios}")
    log(f"Training from step {start_step} to {max_steps}")
    log(f"Batch size: {batch_size} | Seq len: {seq_len}")
    if stage == "base":
        log(f"LR: {BASE_LR} (WSD: warmup {BASE_WARMUP_STEPS}, "
            f"decay last {BASE_DECAY_FRACTION*100:.0f}%)")
    else:
        log(f"LR: {anneal_start_lr:.2e} → 0 (linear decay)")
    log(f"Tokens so far: {tokens_total:,}")
    log(f"Target: ~{max_steps * batch_size * seq_len / 1e9:.1f}B new tokens")
    log("-" * 70)

    for step in range(start_step, max_steps):
        if shutdown_requested:
            break

        # LR schedule
        if stage == "base":
            current_lr = wsd_lr(step, max_steps, BASE_LR, BASE_WARMUP_STEPS,
                                BASE_DECAY_FRACTION)
        else:
            current_lr = linear_decay_lr(step - start_step,
                                         max_steps - start_step,
                                         anneal_start_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Get batch
        input_ids, labels = data.get_batch(batch_size)
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
        tokens_total += batch_size * seq_len

        # Log
        if (step + 1) % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            tps = (step - start_step + 1) * batch_size * seq_len / elapsed
            perplexity = math.exp(min(avg_loss, 20))
            hours = elapsed / 3600
            tokens_b = tokens_total / 1e9
            pct = (step + 1) / max_steps * 100

            log(f"step {step+1:>7d} | loss {avg_loss:.4f} | ppl {perplexity:.1f} | "
                f"lr {current_lr:.2e} | {tps:.0f} tok/s | "
                f"{tokens_b:.2f}B tokens ({pct:.1f}%) | {hours:.1f}h")
            log_metrics(step + 1, avg_loss, perplexity, current_lr, tps,
                        tokens_total, hours)

        # Validation
        if (step + 1) % VAL_EVERY == 0:
            val_loss = val_set.evaluate(model, device)
            val_ppl = math.exp(min(val_loss, 20))
            improved = " *NEW BEST*" if val_loss < best_val_loss else ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            log(f"  VAL loss {val_loss:.4f} | ppl {val_ppl:.1f}{improved}")

            # Log val_loss to metrics
            elapsed = time.time() - start_time
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            tps = (step - start_step + 1) * batch_size * seq_len / elapsed
            log_metrics(step + 1, avg_loss, math.exp(min(avg_loss, 20)),
                        current_lr, tps, tokens_total, elapsed / 3600,
                        val_loss=val_loss)

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
                checkpoint_dir, tokens_total=tokens_total, stage=stage,
            )

        if shutdown_requested:
            break

    # Final save
    avg_loss = sum(losses[-100:]) / len(losses[-100:]) if losses else float("inf")
    save_checkpoint(
        model, optimizer, scaler, config, step + 1, avg_loss,
        checkpoint_dir, tokens_total=tokens_total, stage=stage,
    )

    elapsed = time.time() - start_time
    log("=" * 70)
    log(f"PRETRAINING {stage.upper()} {'STOPPED' if shutdown_requested else 'COMPLETE'}")
    log(f"Final step: {step + 1} | Loss: {avg_loss:.4f} | Time: {elapsed/3600:.1f}h")
    log(f"Total tokens: {tokens_total:,} ({tokens_total/1e9:.2f}B)")
    log(f"Best val loss: {best_val_loss:.4f}")
    log("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hamner V4 Pretraining")
    parser.add_argument("--stage", type=str, default="base",
                        choices=["base", "anneal"],
                        help="Training stage: base or anneal")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from specific checkpoint")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh (ignore existing checkpoints)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override max steps")
    args = parser.parse_args()

    if args.steps is not None:
        if args.stage == "base":
            BASE_MAX_STEPS = args.steps
        else:
            ANNEAL_MAX_STEPS = args.steps

    if args.checkpoint:
        train(stage=args.stage, resume_from=args.checkpoint)
    elif args.fresh:
        train(stage=args.stage, fresh=True)
    elif args.resume:
        train(stage=args.stage)  # auto-detects checkpoint
    else:
        train(stage=args.stage, fresh=True)  # default: fresh start
