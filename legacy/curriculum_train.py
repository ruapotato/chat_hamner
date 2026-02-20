"""
Hamner Curriculum Training
===========================
Loads from an existing pretrained checkpoint and runs multi-phase
curriculum training to reorganize embeddings via structured tasks,
build narrative coherence via TinyStories, then learn personal voice.

Phases:
  1. Structure  - 60% synthetic tasks, 20% TinyStories, 15% FineWeb, 5% personal
  2. Stories    - 20% synthetic, 50% TinyStories, 20% FineWeb, 10% personal
  3. Voice      - 20% synthetic, 30% TinyStories, 20% FineWeb, 30% personal
  4. Polish     - 25% each source, balanced mix

Applies "emotional transformer" idea: middle layers train at 0.2x base LR.

Usage:
    python curriculum_train.py                             # auto-find latest checkpoint
    python curriculum_train.py --checkpoint path/to/ckpt   # specific checkpoint
    python curriculum_train.py --resume                    # resume curriculum training
    python curriculum_train.py --phase 2                   # skip to phase N
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
from dataclasses import dataclass
from typing import Optional

from model import HamnerModel, HamnerConfig
from variants import emotional_param_groups
from synthetic_tasks import SyntheticTaskGenerator


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/curriculum"
LOG_FILE = "logs/curriculum.log"
METRICS_FILE = "logs/curriculum_metrics.csv"
SAMPLES_FILE = "logs/curriculum_samples.jsonl"

BASE_LR = 1e-4
WARMUP_STEPS = 500
BATCH_SIZE = 32
SEQ_LEN = 1024
CHECKPOINT_EVERY = 2000
SAMPLE_EVERY = 500
LOG_EVERY = 50
KEEP_CHECKPOINTS = 10

EMOTIONAL_LAYERS = 6
EMOTIONAL_LR_SCALE = 0.2

# Plateau detection
PLATEAU_WINDOW = 500       # steps to average over for "recent" loss
PLATEAU_COMPARE = 500      # steps back to compare against
PLATEAU_THRESHOLD = 0.005  # < 0.5% improvement = plateau
PLATEAU_MIN_STEPS = 1000   # don't check before this many steps in a phase

SAMPLE_PROMPTS = [
    # Structured tasks
    "Calculate: 15 + 23 =",
    "Count the items: apple, banana, cherry.",
    "Sort: 5 2 8 1 ->",
    # Narrative
    "Once upon a time there was a little",
    "The cat sat on the",
    # Voice / tech
    "Hello and welcome back everybody",
    "The thing about Linux is",
    "Today we are going to look at",
]


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

@dataclass
class PhaseConfig:
    name: str
    steps: int
    ratios: dict        # source -> fraction, must sum to 1.0
    difficulty_start: int
    difficulty_end: int
    lr_multiplier: float

CURRICULUM_PHASES = [
    PhaseConfig(
        name="structure",
        steps=2_000,
        ratios={"synthetic": 0.60, "tinystories": 0.20, "fineweb": 0.15, "personal": 0.05},
        difficulty_start=1, difficulty_end=3,
        lr_multiplier=1.0,
    ),
    PhaseConfig(
        name="stories",
        steps=20_000,
        ratios={"synthetic": 0.0, "tinystories": 1.0, "fineweb": 0.0, "personal": 0.0},
        difficulty_start=3, difficulty_end=4,
        lr_multiplier=0.8,
    ),
    PhaseConfig(
        name="voice",
        steps=10_000,
        ratios={"synthetic": 0.10, "tinystories": 0.30, "fineweb": 0.0, "personal": 0.60},
        difficulty_start=4, difficulty_end=5,
        lr_multiplier=0.6,
    ),
    PhaseConfig(
        name="polish",
        steps=5_000,
        ratios={"synthetic": 0.15, "tinystories": 0.35, "fineweb": 0.0, "personal": 0.50},
        difficulty_start=5, difficulty_end=5,
        lr_multiplier=0.4,
    ),
]


# ---------------------------------------------------------------------------
# Logging (mirrors train.py)
# ---------------------------------------------------------------------------

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def log_metrics(step, loss, perplexity, lr, tokens_per_sec, tokens_total, elapsed_hours, phase_name):
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    write_header = not os.path.exists(METRICS_FILE)
    with open(METRICS_FILE, "a") as f:
        if write_header:
            f.write("timestamp,step,loss,perplexity,learning_rate,tokens_per_sec,tokens_total,tokens_billions,elapsed_hours,phase\n")
        ts = datetime.datetime.now().isoformat()
        tokens_b = tokens_total / 1e9
        f.write(f"{ts},{step},{loss:.6f},{perplexity:.2f},{lr:.6e},{tokens_per_sec:.0f},{tokens_total},{tokens_b:.4f},{elapsed_hours:.4f},{phase_name}\n")


def log_samples(step, tokens_total, phase_name, samples_dict):
    os.makedirs(os.path.dirname(SAMPLES_FILE), exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "step": step,
        "tokens_total": tokens_total,
        "phase": phase_name,
        "samples": samples_dict,
    }
    with open(SAMPLES_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Data mixer
# ---------------------------------------------------------------------------

class CurriculumDataMixer:
    """Mixes data from 4 sources with configurable ratios."""

    def __init__(self, tokenizer, seq_len=1024, personal_data_path="data/personal/training_samples.jsonl"):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.synthetic_gen = SyntheticTaskGenerator(seed=42)
        self.token_buffers = {
            "synthetic": [],
            "tinystories": [],
            "fineweb": [],
            "personal": [],
        }
        self.ratios = {"synthetic": 0.6, "tinystories": 0.2, "fineweb": 0.15, "personal": 0.05}
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
        else:
            log(f"No personal data found at {path}")

    def _init_streams(self):
        from datasets import load_dataset
        log("Initializing TinyStories stream...")
        self.tinystories_stream = iter(load_dataset(
            "roneneldan/TinyStories", split="train", streaming=True,
        ))
        log("Initializing FineWeb-Edu stream...")
        self.fineweb_stream = iter(load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT",
            split="train", streaming=True,
        ))
        log("Data streams ready")

    def _restart_tinystories(self):
        from datasets import load_dataset
        log("TinyStories exhausted, restarting...")
        self.tinystories_stream = iter(load_dataset(
            "roneneldan/TinyStories", split="train", streaming=True,
        ))

    def _restart_fineweb(self):
        from datasets import load_dataset
        log("FineWeb exhausted, restarting...")
        self.fineweb_stream = iter(load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT",
            split="train", streaming=True,
        ))

    def set_ratios(self, ratios: dict):
        self.ratios = ratios

    def set_difficulty(self, level: int):
        self.synthetic_gen.set_difficulty(level)

    def _get_text(self, source: str) -> str:
        if source == "synthetic":
            return self.synthetic_gen.get_random_task()
        elif source == "tinystories":
            while True:
                try:
                    sample = next(self.tinystories_stream)
                    text = sample.get("text", "")
                    if len(text.strip()) >= 30:
                        return text
                except StopIteration:
                    self._restart_tinystories()
        elif source == "fineweb":
            while True:
                try:
                    sample = next(self.fineweb_stream)
                    text = sample.get("text", "")
                    if len(text.strip()) >= 50:
                        return text
                except StopIteration:
                    self._restart_fineweb()
        elif source == "personal":
            if self.personal_samples:
                return random.choice(self.personal_samples)
            return self._get_text("tinystories")  # fallback
        return ""

    def _choose_source(self) -> str:
        r = random.random()
        cumulative = 0.0
        for source, ratio in self.ratios.items():
            cumulative += ratio
            if r < cumulative:
                return source
        return list(self.ratios.keys())[-1]

    def _fill_buffer(self, source: str):
        text = self._get_text(source)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        tokens.append(self.tokenizer.eos_token_id or 0)
        self.token_buffers[source].extend(tokens)

    def get_batch(self, batch_size: int):
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

            # NO pre-shifting — model.forward() handles shift internally
            input_ids.append(torch.tensor(chunk, dtype=torch.long))
            labels.append(torch.tensor(chunk, dtype=torch.long))

        return torch.stack(input_ids), torch.stack(labels)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scaler, config, step, loss, tokens_total,
                    phase_idx, step_in_phase, phase_name):
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
        "curriculum_phase": phase_idx,
        "curriculum_step_in_phase": step_in_phase,
        "curriculum_phase_name": phase_name,
    }

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{step:07d}.pt")
    torch.save(ckpt_data, ckpt_path)

    latest_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    torch.save(ckpt_data, latest_path)

    log(f"  Checkpoint saved: {ckpt_path} (phase={phase_name})")

    # Cleanup: keep milestones (every 10k) + last N
    all_ckpts = sorted(Path(CHECKPOINT_DIR).glob("step_*.pt"))
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
            results.append(f"{prompt} [error: {e}]")
    model.train()
    return results


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def check_plateau(phase_losses, step_in_phase):
    """Check if loss has plateaued within the current phase.

    Compares average loss over the most recent PLATEAU_WINDOW steps
    against the average from PLATEAU_COMPARE steps before that.
    Returns (is_plateau, improvement_pct).
    """
    if step_in_phase < PLATEAU_MIN_STEPS:
        return False, 0.0

    needed = PLATEAU_WINDOW + PLATEAU_COMPARE
    if len(phase_losses) < needed:
        return False, 0.0

    recent = phase_losses[-PLATEAU_WINDOW:]
    older = phase_losses[-(PLATEAU_WINDOW + PLATEAU_COMPARE):-PLATEAU_WINDOW]

    avg_recent = sum(recent) / len(recent)
    avg_older = sum(older) / len(older)

    if avg_older == 0:
        return False, 0.0

    # Positive = loss went down (good), negative = loss went up
    improvement = (avg_older - avg_recent) / avg_older
    return improvement < PLATEAU_THRESHOLD, improvement


def compute_lr(step_in_phase, phase_steps, phase_lr_mult):
    """Cosine decay within each phase, with warmup at the very start."""
    peak_lr = BASE_LR * phase_lr_mult
    min_lr = peak_lr * 0.1

    # Global warmup only applies for the first WARMUP_STEPS of overall training
    # (handled externally). Within each phase, just cosine decay.
    progress = step_in_phase / max(1, phase_steps - 1)
    return min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def curriculum_train(checkpoint_path=None, resume=False, start_phase=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("HAMNER CURRICULUM TRAINING")
    log("=" * 70)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    # Find checkpoint
    if checkpoint_path is None:
        # Check curriculum checkpoints first, then original training
        curriculum_latest = Path(CHECKPOINT_DIR) / "latest.pt"
        if resume and curriculum_latest.exists():
            checkpoint_path = str(curriculum_latest)
        else:
            training_latest = Path("checkpoints/training/latest.pt")
            if training_latest.exists():
                checkpoint_path = str(training_latest)

    if checkpoint_path is None:
        log("ERROR: No checkpoint found. Curriculum training requires an existing checkpoint.")
        sys.exit(1)

    log(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Build config, patching in emotional settings
    config_dict = ckpt["config"]
    config = HamnerConfig(**config_dict)
    config.emotional_layers = EMOTIONAL_LAYERS
    config.emotional_lr_scale = EMOTIONAL_LR_SCALE
    config.gradient_checkpointing = True
    config.vocab_size = tokenizer.vocab_size

    # Build model and load weights
    model = HamnerModel(config).to(device)
    state_dict = ckpt["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)

    total_p, _ = model.count_parameters()
    log(f"Model: {total_p:,} params | {config.hidden_size}h x {config.num_layers}L")
    log(f"Emotional layers: middle {EMOTIONAL_LAYERS} at {EMOTIONAL_LR_SCALE}x LR")
    pretrain_tokens = ckpt.get("tokens_total", 0)
    pretrain_step = ckpt.get("step", 0)
    log(f"Pretrained: step {pretrain_step}, {pretrain_tokens/1e9:.2f}B tokens, loss {ckpt.get('avg_loss', '?')}")

    # Determine resume state
    resume_phase = start_phase
    resume_step_in_phase = 0
    if resume and "curriculum_phase" in ckpt:
        resume_phase = ckpt["curriculum_phase"]
        resume_step_in_phase = ckpt.get("curriculum_step_in_phase", 0)
        log(f"Resuming from phase {resume_phase} ({ckpt.get('curriculum_phase_name', '?')}), step {resume_step_in_phase}")

    # Create optimizer with emotional param groups (FRESH - no old momentum)
    param_groups = emotional_param_groups(model, BASE_LR)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.1)

    # If resuming curriculum, restore optimizer state
    if resume and "curriculum_phase" in ckpt and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            log("Restored curriculum optimizer state")
        except Exception as e:
            log(f"Could not restore optimizer state: {e}, using fresh optimizer")

    scaler = torch.amp.GradScaler("cuda")
    if resume and "curriculum_phase" in ckpt and "scaler_state_dict" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception:
            pass

    # Compile
    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile...")
        model = torch.compile(model)

    model.train()

    # Data mixer
    mixer = CurriculumDataMixer(tokenizer, seq_len=SEQ_LEN)

    # Graceful shutdown
    shutdown_requested = False
    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        log("Shutdown signal received, saving checkpoint...")
        shutdown_requested = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Training state
    global_step = 0
    tokens_total = pretrain_tokens
    losses = []
    start_time = time.time()

    # Compute total steps for display
    total_steps = sum(p.steps for p in CURRICULUM_PHASES)
    log(f"Total curriculum: {total_steps:,} steps across {len(CURRICULUM_PHASES)} phases")
    log(f"Estimated tokens: ~{total_steps * BATCH_SIZE * SEQ_LEN / 1e9:.2f}B")
    log("-" * 70)

    for phase_idx, phase in enumerate(CURRICULUM_PHASES):
        if phase_idx < resume_phase:
            global_step += phase.steps
            continue

        log(f"\n{'='*70}")
        log(f"PHASE {phase_idx + 1}/{len(CURRICULUM_PHASES)}: {phase.name.upper()}")
        log(f"Steps: {phase.steps:,} | LR mult: {phase.lr_multiplier}")
        log(f"Ratios: {phase.ratios}")
        log(f"Difficulty: {phase.difficulty_start} -> {phase.difficulty_end}")
        log(f"{'='*70}")

        mixer.set_ratios(phase.ratios)

        phase_start = resume_step_in_phase if phase_idx == resume_phase else 0
        phase_losses = []  # track losses within this phase for plateau detection
        plateau_hit = False

        for step_in_phase in range(phase_start, phase.steps):
            if shutdown_requested:
                break

            # Interpolate difficulty within phase
            progress = step_in_phase / max(1, phase.steps - 1)
            difficulty = round(phase.difficulty_start + progress * (phase.difficulty_end - phase.difficulty_start))
            mixer.set_difficulty(difficulty)

            # LR schedule: warmup then cosine per-phase
            if global_step < WARMUP_STEPS:
                current_lr = BASE_LR * phase.lr_multiplier * (global_step + 1) / WARMUP_STEPS
            else:
                current_lr = compute_lr(step_in_phase, phase.steps, phase.lr_multiplier)

            # Set LR for both param groups (fast and slow/emotional)
            for pg in optimizer.param_groups:
                if pg.get("lr", BASE_LR) < BASE_LR * 0.5:
                    # This is the emotional (slow) group
                    pg["lr"] = current_lr * EMOTIONAL_LR_SCALE
                else:
                    pg["lr"] = current_lr

            # Get batch
            input_ids, labels_ = mixer.get_batch(BATCH_SIZE)
            input_ids = input_ids.to(device)
            labels_ = labels_.to(device)

            # Forward + backward
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids, labels=labels_)
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            losses.append(loss_val)
            phase_losses.append(loss_val)
            tokens_total += BATCH_SIZE * SEQ_LEN
            global_step += 1

            # Plateau detection (check every LOG_EVERY steps to avoid overhead)
            if step_in_phase > 0 and step_in_phase % LOG_EVERY == 0:
                is_plateau, improvement = check_plateau(phase_losses, step_in_phase)
                if is_plateau:
                    avg_recent = sum(phase_losses[-PLATEAU_WINDOW:]) / PLATEAU_WINDOW
                    log(f"  ** PLATEAU DETECTED at step {step_in_phase} in phase '{phase.name}' **")
                    log(f"     Improvement: {improvement*100:.2f}% over last {PLATEAU_COMPARE} steps (threshold: {PLATEAU_THRESHOLD*100:.1f}%)")
                    log(f"     Recent avg loss: {avg_recent:.4f}")
                    log(f"     Advancing to next phase early (saved {phase.steps - step_in_phase} steps)")
                    plateau_hit = True
                    break

            # Log
            if global_step % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                avg_loss = sum(losses[-100:]) / len(losses[-100:])
                tps = (global_step - (sum(p.steps for i, p in enumerate(CURRICULUM_PHASES) if i < resume_phase) + phase_start)) * BATCH_SIZE * SEQ_LEN / max(elapsed, 1)
                perplexity = math.exp(min(avg_loss, 20))
                hours = elapsed / 3600
                pct = global_step / total_steps * 100

                log(f"step {global_step:>6d} | phase {phase.name:<9s} | loss {avg_loss:.4f} | ppl {perplexity:.1f} | "
                    f"lr {current_lr:.2e} | {tps:.0f} tok/s | {pct:.1f}% | {hours:.1f}h")
                log_metrics(global_step, avg_loss, perplexity, current_lr, tps, tokens_total, hours, phase.name)

            # Generate samples
            if global_step % SAMPLE_EVERY == 0:
                log("--- SAMPLE GENERATIONS ---")
                samples = generate_samples(model, tokenizer, SAMPLE_PROMPTS[:4], device)
                samples_dict = {}
                for i, sample in enumerate(samples):
                    sample = sample[:300]
                    log(f"  [{i+1}] {sample}")
                    samples_dict[SAMPLE_PROMPTS[i]] = sample
                log("-" * 40)
                log_samples(global_step, tokens_total, phase.name, samples_dict)

            # Checkpoint
            if global_step % CHECKPOINT_EVERY == 0:
                avg_loss = sum(losses[-100:]) / len(losses[-100:])
                save_checkpoint(
                    model, optimizer, scaler, config,
                    global_step, avg_loss, tokens_total,
                    phase_idx, step_in_phase + 1, phase.name,
                )

        if shutdown_requested:
            break

        if plateau_hit:
            log(f"\nPhase {phase.name} ended early at step {step_in_phase} (plateau) | global step {global_step}")
        else:
            log(f"\nPhase {phase.name} complete! (step {global_step})")

        # Save phase-end checkpoint
        avg_loss = sum(losses[-100:]) / len(losses[-100:]) if losses else float("inf")
        save_checkpoint(
            model, optimizer, scaler, config,
            global_step, avg_loss, tokens_total,
            phase_idx + 1, 0, f"{phase.name}_done",
        )

    # Final summary
    elapsed = time.time() - start_time
    avg_loss = sum(losses[-100:]) / len(losses[-100:]) if losses else float("inf")
    log("=" * 70)
    log(f"CURRICULUM TRAINING {'STOPPED' if shutdown_requested else 'COMPLETE'}")
    log(f"Final step: {global_step} | Loss: {avg_loss:.4f} | Time: {elapsed/3600:.1f}h")
    log(f"Total tokens (pretrain + curriculum): {tokens_total:,} ({tokens_total/1e9:.2f}B)")
    log("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hamner Curriculum Training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained checkpoint")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest curriculum checkpoint")
    parser.add_argument("--phase", type=int, default=0,
                        help="Start from this phase (0-indexed)")
    args = parser.parse_args()

    curriculum_train(
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        start_phase=args.phase,
    )
