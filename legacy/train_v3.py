"""
Hamner Train V3.1: 50M Model with Stricter Plateau Detection + 8 Stages
========================================================================
Fresh-from-scratch 50M parameter model trained through 8 stages,
each advancing only when loss plateaus (triple-confirmed). Every stage
adds a new data focus while keeping all prior sources active.

Plateau detection is stricter than V3: wider windows (1000/2000 steps),
tighter threshold (0.1%), minimum 5000 steps, and requires 3 consecutive
strike confirmations before advancing.

Stages:
  1. Structure      - 70% TinyStories, 30% synthetic (English grammar)
  2. Reasoning      - 55% synthetic, 35% TinyStories, 10% FineWeb (logic/math/patterns)
  3. Knowledge      - 60% FineWeb, 20% TinyStories, 20% synthetic (world knowledge)
  4. Context        - 35% FineWeb, 35% synthetic, 20% TinyStories, 10% SFT (fact tracking/recall)
  5. Deep Knowledge - 40% FineWeb, 25% synthetic, 15% TinyStories, 15% SFT, 5% personal (consolidate)
  6. Dialogue       - 40% SFT, 20% FineWeb, 15% TinyStories, 15% synthetic, 10% personal (conversational)
  7. Synthesis      - 25% FineWeb, 25% SFT, 20% synthetic, 15% TinyStories, 15% personal (integrate)
  8. Voice          - 35% personal, 25% SFT, 15% FineWeb, 10% TinyStories, 15% synthetic (personality)

Expected: ~200k-400k steps, 5-10B tokens, ~30-60 hours at 74k tok/s.

Usage:
    python train_v3.py                    # start fresh
    python train_v3.py --resume           # resume from latest checkpoint
    python train_v3.py --stage 2          # skip to stage N (0-indexed)
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
from dataclasses import dataclass, field
from typing import Dict, List

from model import HamnerModel, HamnerConfig
from synthetic_tasks import SyntheticTaskGenerator


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/v3"
LOG_FILE = "logs/v3.log"
METRICS_FILE = "logs/v3_metrics.csv"
SAMPLES_FILE = "logs/v3_samples.jsonl"

MODEL_CONFIG = dict(
    hidden_size=512,
    num_layers=8,
    num_attention_heads=8,
    num_kv_heads=2,          # GQA 4:1
    num_experts=1,
    num_active_experts=1,
    expert_intermediate_size=1365,  # ~8/3 × 512
    use_differential_attention=False,
    gradient_checkpointing=True,
    max_seq_len=1024,
)

BATCH_SIZE = 24
SEQ_LEN = 1024
CHECKPOINT_EVERY = 2000
SAMPLE_EVERY = 500
LOG_EVERY = 50
KEEP_CHECKPOINTS = 10

# Plateau detection — stricter, triple-confirmed
PLATEAU_WINDOW = 1000       # steps to average over for "recent" loss
PLATEAU_COMPARE = 2000      # steps back to compare against
PLATEAU_THRESHOLD = 0.001   # < 0.1% improvement = plateau (was 0.5%)
PLATEAU_MIN_STEPS = 5000    # don't check before this many steps per stage
PLATEAU_STRIKES_NEEDED = 3  # must trigger 3 consecutive checks to advance
MAX_STEPS_PER_STAGE = 200_000  # safety fallback

WARMUP_STEPS_PER_STAGE = 300  # brief warm restart at each stage

SAMPLE_PROMPTS = [
    "Calculate: 15 + 23 =",                        # arithmetic skill
    "Once upon a time",                             # narrative coherence
    "The most important thing is",                  # general knowledge
    "Hello! How are you",                           # conversational
    "The thing about computers is",                 # personal/tech voice
    "If it rains then the ground is wet. It rains. Therefore",  # logic skill
    "Which is bigger: 45 or 32?",                   # comparison skill
]


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

@dataclass
class StageConfig:
    name: str
    ratios: Dict[str, float]   # source -> fraction, must sum to 1.0
    base_lr: float
    synthetic_difficulty: int = 1  # 1-5, ramps across stages

STAGES = [
    StageConfig(
        name="structure",
        ratios={"tinystories": 0.70, "synthetic": 0.30},
        base_lr=3e-4,
        synthetic_difficulty=1,
    ),
    StageConfig(
        name="reasoning",
        ratios={"synthetic": 0.55, "tinystories": 0.35, "fineweb": 0.10},
        base_lr=2.5e-4,
        synthetic_difficulty=3,
    ),
    StageConfig(
        name="knowledge",
        ratios={"fineweb": 0.60, "tinystories": 0.20, "synthetic": 0.20},
        base_lr=2e-4,
        synthetic_difficulty=3,
    ),
    StageConfig(
        name="context",
        ratios={"fineweb": 0.35, "synthetic": 0.35, "tinystories": 0.20, "sft": 0.10},
        base_lr=1.5e-4,
        synthetic_difficulty=4,
    ),
    StageConfig(
        name="deep_knowledge",
        ratios={"fineweb": 0.40, "synthetic": 0.25, "tinystories": 0.15, "sft": 0.15, "personal": 0.05},
        base_lr=1.2e-4,
        synthetic_difficulty=4,
    ),
    StageConfig(
        name="dialogue",
        ratios={"sft": 0.40, "fineweb": 0.20, "tinystories": 0.15, "synthetic": 0.15, "personal": 0.10},
        base_lr=8e-5,
        synthetic_difficulty=4,
    ),
    StageConfig(
        name="synthesis",
        ratios={"fineweb": 0.25, "sft": 0.25, "synthetic": 0.20, "tinystories": 0.15, "personal": 0.15},
        base_lr=5e-5,
        synthetic_difficulty=5,
    ),
    StageConfig(
        name="voice",
        ratios={"personal": 0.35, "sft": 0.25, "fineweb": 0.15, "tinystories": 0.10, "synthetic": 0.15},
        base_lr=3e-5,
        synthetic_difficulty=5,
    ),
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


def log_metrics(step, loss, perplexity, lr, tokens_per_sec, tokens_total, elapsed_hours, stage_name):
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    write_header = not os.path.exists(METRICS_FILE)
    with open(METRICS_FILE, "a") as f:
        if write_header:
            f.write("timestamp,step,loss,perplexity,learning_rate,tokens_per_sec,tokens_total,tokens_billions,elapsed_hours,phase\n")
        ts = datetime.datetime.now().isoformat()
        tokens_b = tokens_total / 1e9
        f.write(f"{ts},{step},{loss:.6f},{perplexity:.2f},{lr:.6e},{tokens_per_sec:.0f},{tokens_total},{tokens_b:.4f},{elapsed_hours:.4f},{stage_name}\n")


def log_samples(step, tokens_total, stage_name, samples_dict):
    os.makedirs(os.path.dirname(SAMPLES_FILE), exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "step": step,
        "tokens_total": tokens_total,
        "phase": stage_name,
        "samples": samples_dict,
    }
    with open(SAMPLES_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Data mixer
# ---------------------------------------------------------------------------

class StagedDataMixer:
    """Mixes data from 5 sources with configurable ratios per stage."""

    def __init__(self, tokenizer, seq_len=1024,
                 personal_data_path="data/personal/training_samples.jsonl",
                 sft_data_path="data/personal/sft_conversations.jsonl"):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.synthetic_gen = SyntheticTaskGenerator(seed=42)

        self.token_buffers = {
            "synthetic": [],
            "tinystories": [],
            "fineweb": [],
            "personal": [],
            "sft": [],
        }
        self.ratios = {"synthetic": 1.0}  # default, overridden per stage

        self._load_local_data(personal_data_path, "personal")
        self._load_local_data(sft_data_path, "sft")
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
        elif source == "sft":
            if self.sft_samples:
                return random.choice(self.sft_samples)
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
# Plateau detection
# ---------------------------------------------------------------------------

def check_plateau(stage_losses, step_in_stage):
    """Check if loss has plateaued within the current stage.

    Compares average loss over the most recent PLATEAU_WINDOW steps
    against the average from PLATEAU_COMPARE steps before that.
    Returns (is_below_threshold, improvement_pct).

    Note: caller is responsible for tracking strikes (consecutive
    below-threshold checks) and only advancing after PLATEAU_STRIKES_NEEDED.
    """
    if step_in_stage < PLATEAU_MIN_STEPS:
        return False, 0.0

    needed = PLATEAU_WINDOW + PLATEAU_COMPARE
    if len(stage_losses) < needed:
        return False, 0.0

    recent = stage_losses[-PLATEAU_WINDOW:]
    older = stage_losses[-(PLATEAU_WINDOW + PLATEAU_COMPARE):-PLATEAU_WINDOW]

    avg_recent = sum(recent) / len(recent)
    avg_older = sum(older) / len(older)

    if avg_older == 0:
        return False, 0.0

    # Positive = loss went down (good), negative = loss went up
    improvement = (avg_older - avg_recent) / avg_older
    return improvement < PLATEAU_THRESHOLD, improvement


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scaler, config, global_step, loss,
                    tokens_total, stage_idx, step_in_stage, stage_name):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    raw_state = model.state_dict()
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}

    ckpt_data = {
        "model_state_dict": clean_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config.__dict__,
        "step": global_step,
        "avg_loss": loss,
        "tokens_total": tokens_total,
        "timestamp": datetime.datetime.now().isoformat(),
        "stage_idx": stage_idx,
        "stage_name": stage_name,
        "step_in_stage": step_in_stage,
    }

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{global_step:07d}.pt")
    torch.save(ckpt_data, ckpt_path)

    latest_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    torch.save(ckpt_data, latest_path)

    log(f"  Checkpoint saved: {ckpt_path} (stage={stage_name})")

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
# Main training loop
# ---------------------------------------------------------------------------

def train_v3(resume=False, start_stage=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("HAMNER TRAIN V3 — 50M Model, Plateau-Driven Staged Learning")
    log("=" * 70)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    # Initialize or resume
    resume_stage = start_stage
    resume_step_in_stage = 0
    resume_global_step = 0
    resume_tokens = 0

    if resume:
        latest_path = Path(CHECKPOINT_DIR) / "latest.pt"
        if latest_path.exists():
            log(f"Resuming from {latest_path}")
            ckpt = torch.load(str(latest_path), map_location="cpu", weights_only=False)
            config = HamnerConfig(**ckpt["config"])
            model = HamnerModel(config).to(device)
            cleaned = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
            model.load_state_dict(cleaned, strict=True)

            resume_stage = ckpt.get("stage_idx", 0)
            resume_step_in_stage = ckpt.get("step_in_stage", 0)
            resume_global_step = ckpt.get("step", 0)
            resume_tokens = ckpt.get("tokens_total", 0)

            total_p, _ = model.count_parameters()
            log(f"Resumed: {total_p:,} params | global step {resume_global_step} | "
                f"stage {resume_stage} ({ckpt.get('stage_name', '?')}) step {resume_step_in_stage} | "
                f"loss {ckpt.get('avg_loss', '?')}")

            # Restore optimizer and scaler
            stage_lr = STAGES[resume_stage].base_lr
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=stage_lr, betas=(0.9, 0.95), weight_decay=0.1
            )
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    log("Restored optimizer state")
                except Exception as e:
                    log(f"Could not restore optimizer: {e}, using fresh")

            scaler = torch.amp.GradScaler("cuda")
            if "scaler_state_dict" in ckpt:
                try:
                    scaler.load_state_dict(ckpt["scaler_state_dict"])
                except Exception:
                    pass
        else:
            log("No checkpoint found, starting fresh")
            resume = False

    if not resume:
        log("Starting fresh 50M model from scratch")
        config = HamnerConfig(**MODEL_CONFIG, vocab_size=tokenizer.vocab_size)
        model = HamnerModel(config).to(device)

        total_p, _ = model.count_parameters()
        log(f"Model: {total_p:,} params | {config.hidden_size}h x {config.num_layers}L")
        log(f"Config: GQA {config.num_attention_heads}h/{config.num_kv_heads}kv | "
            f"MLP {config.expert_intermediate_size} | seq {config.max_seq_len}")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=STAGES[0].base_lr, betas=(0.9, 0.95), weight_decay=0.1
        )
        scaler = torch.amp.GradScaler("cuda")

    # Compile for speed
    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile...")
        model = torch.compile(model)

    model.train()

    # Data mixer
    mixer = StagedDataMixer(tokenizer, seq_len=SEQ_LEN)

    # Graceful shutdown
    shutdown_requested = False
    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        log("Shutdown signal received, saving checkpoint...")
        shutdown_requested = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Training state
    global_step = resume_global_step
    tokens_total = resume_tokens
    all_losses = []
    start_time = time.time()

    log(f"Stages: {len(STAGES)} | Batch: {BATCH_SIZE} | Seq: {SEQ_LEN}")
    log(f"Plateau detection: window={PLATEAU_WINDOW}, compare={PLATEAU_COMPARE}, "
        f"threshold={PLATEAU_THRESHOLD*100:.1f}%, min_steps={PLATEAU_MIN_STEPS}, "
        f"strikes={PLATEAU_STRIKES_NEEDED}")
    for i, s in enumerate(STAGES):
        log(f"  Stage {i}: {s.name} | LR {s.base_lr:.1e} | diff={s.synthetic_difficulty} | {s.ratios}")
    log("-" * 70)

    for stage_idx, stage in enumerate(STAGES):
        if stage_idx < resume_stage:
            continue

        log(f"\n{'='*70}")
        log(f"STAGE {stage_idx + 1}/{len(STAGES)}: {stage.name.upper()}")
        log(f"Base LR: {stage.base_lr:.1e} | Synthetic diff: {stage.synthetic_difficulty} | Ratios: {stage.ratios}")
        log(f"{'='*70}")

        mixer.set_ratios(stage.ratios)
        mixer.set_difficulty(stage.synthetic_difficulty)

        # Reset optimizer momentum for new data distribution (fresh optimizer)
        if stage_idx > resume_stage or not resume:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=stage.base_lr, betas=(0.9, 0.95), weight_decay=0.1
            )
            scaler = torch.amp.GradScaler("cuda")
            log("  Optimizer reset for new stage")

        stage_start = resume_step_in_stage if stage_idx == resume_stage else 0
        stage_losses = []
        plateau_hit = False
        plateau_strikes = 0

        for step_in_stage in range(stage_start, MAX_STEPS_PER_STAGE):
            if shutdown_requested:
                break

            # LR schedule: warmup then constant
            warmup_step = step_in_stage - stage_start  # steps since we started this run
            if warmup_step < WARMUP_STEPS_PER_STAGE:
                current_lr = stage.base_lr * (warmup_step + 1) / WARMUP_STEPS_PER_STAGE
            else:
                current_lr = stage.base_lr

            for pg in optimizer.param_groups:
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
            all_losses.append(loss_val)
            stage_losses.append(loss_val)
            tokens_total += BATCH_SIZE * SEQ_LEN
            global_step += 1

            # Plateau detection — triple-strike (check every LOG_EVERY steps)
            if step_in_stage > 0 and step_in_stage % LOG_EVERY == 0:
                is_below, improvement = check_plateau(stage_losses, step_in_stage)
                if is_below:
                    plateau_strikes += 1
                    avg_recent = sum(stage_losses[-PLATEAU_WINDOW:]) / PLATEAU_WINDOW
                    log(f"  PLATEAU STRIKE {plateau_strikes}/{PLATEAU_STRIKES_NEEDED} "
                        f"at step {step_in_stage} in stage '{stage.name}' | "
                        f"improvement {improvement*100:.3f}% | recent loss {avg_recent:.4f}")
                    if plateau_strikes >= PLATEAU_STRIKES_NEEDED:
                        log(f"  ** PLATEAU CONFIRMED ({PLATEAU_STRIKES_NEEDED} strikes) **")
                        log(f"     Threshold: {PLATEAU_THRESHOLD*100:.1f}% | "
                            f"Window: {PLATEAU_WINDOW}/{PLATEAU_COMPARE}")
                        log(f"     Advancing to next stage")
                        plateau_hit = True
                        break
                elif plateau_strikes > 0:
                    log(f"  Plateau strike reset (was {plateau_strikes}/{PLATEAU_STRIKES_NEEDED}) | "
                        f"improvement {improvement*100:.3f}%")
                    plateau_strikes = 0

            # Log
            if global_step % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                avg_loss = sum(all_losses[-100:]) / len(all_losses[-100:])
                tps = (global_step - resume_global_step) * BATCH_SIZE * SEQ_LEN / max(elapsed, 1)
                perplexity = math.exp(min(avg_loss, 20))
                hours = elapsed / 3600

                log(f"step {global_step:>6d} | stage {stage.name:<10s} | s_step {step_in_stage:>6d} | "
                    f"loss {avg_loss:.4f} | ppl {perplexity:.1f} | "
                    f"lr {current_lr:.2e} | {tps:.0f} tok/s | {hours:.1f}h")
                log_metrics(global_step, avg_loss, perplexity, current_lr, tps,
                           tokens_total, hours, stage.name)

            # Generate samples
            if global_step % SAMPLE_EVERY == 0:
                log("--- SAMPLE GENERATIONS ---")
                samples = generate_samples(model, tokenizer, SAMPLE_PROMPTS, device)
                samples_dict = {}
                for i, sample in enumerate(samples):
                    sample = sample[:300]
                    log(f"  [{i+1}] {sample}")
                    samples_dict[SAMPLE_PROMPTS[i]] = sample
                log("-" * 40)
                log_samples(global_step, tokens_total, stage.name, samples_dict)

            # Checkpoint
            if global_step % CHECKPOINT_EVERY == 0:
                avg_loss = sum(all_losses[-100:]) / len(all_losses[-100:])
                save_checkpoint(
                    model, optimizer, scaler, config,
                    global_step, avg_loss, tokens_total,
                    stage_idx, step_in_stage + 1, stage.name,
                )

        if shutdown_requested:
            # Save on shutdown
            avg_loss = sum(all_losses[-100:]) / len(all_losses[-100:]) if all_losses else float("inf")
            save_checkpoint(
                model, optimizer, scaler, config,
                global_step, avg_loss, tokens_total,
                stage_idx, step_in_stage + 1, stage.name,
            )
            break

        # Stage complete (plateau or max steps)
        if plateau_hit:
            log(f"\nStage '{stage.name}' ended at step {step_in_stage} (plateau) | "
                f"global step {global_step}")
        else:
            log(f"\nStage '{stage.name}' ended at step {step_in_stage} (max steps) | "
                f"global step {global_step}")

        # Save stage-end checkpoint
        avg_loss = sum(all_losses[-100:]) / len(all_losses[-100:]) if all_losses else float("inf")
        save_checkpoint(
            model, optimizer, scaler, config,
            global_step, avg_loss, tokens_total,
            stage_idx + 1, 0, f"{stage.name}_done",
        )

        # Reset resume offset for next stage
        resume_step_in_stage = 0

    # Final summary
    elapsed = time.time() - start_time
    avg_loss = sum(all_losses[-100:]) / len(all_losses[-100:]) if all_losses else float("inf")
    log("=" * 70)
    log(f"TRAIN V3 {'STOPPED' if shutdown_requested else 'COMPLETE'}")
    log(f"Final step: {global_step} | Loss: {avg_loss:.4f} | Time: {elapsed/3600:.1f}h")
    log(f"Total tokens: {tokens_total:,} ({tokens_total/1e9:.2f}B)")
    log("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hamner Train V3 — 50M Staged Learning")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--stage", type=int, default=0,
                        help="Start from this stage (0-indexed)")
    args = parser.parse_args()

    train_v3(resume=args.resume, start_stage=args.stage)
