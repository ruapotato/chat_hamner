"""
Hamner SFT (Supervised Fine-Tuning) Script — V4
================================================
Fine-tunes the pretrained Hamner model on conversation data.
Only computes loss on assistant response tokens.

V4: Supports ~104k conversations from SmolTalk + custom data,
    weighted sampling for custom data, validation split, early stopping.

Usage:
    python train_sft.py                                    # start from chat pretrain
    python train_sft.py --checkpoint path/to/base.pt       # use specific pretrained base
    python train_sft.py --resume                           # resume SFT training
    python train_sft.py --data path/to/data.jsonl          # override data file
"""

import os
import sys
import json
import time
import math
import re
import signal
import random
import datetime
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from model import HamnerModel, HamnerConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_CHECKPOINT = "checkpoints/chat_pretrain/latest.pt"
BASE_CHECKPOINT_FALLBACKS = [
    "checkpoints/pretrain_v4_anneal/latest.pt",
    "checkpoints/pretrain_v4/latest.pt",
    "checkpoints/pretrain_v2/latest.pt",
]
SFT_CHECKPOINT_DIR = "checkpoints/sft"
SFT_DATA = "data/sft_combined.jsonl"
SFT_DATA_FALLBACK = "data/personal/sft_diverse_only.jsonl"
VOICE_SAMPLES_FILE = "data/personal/voice_sample.txt"
LOG_FILE = "logs/sft_v4.log"
METRICS_FILE = "logs/sft_v4_metrics.csv"
SAMPLES_FILE = "logs/sft_v4_samples.jsonl"

# Training hyperparameters — V4: more data, fewer epochs
NUM_EPOCHS = 3
BATCH_SIZE = 8
SEQ_LEN = 1024
LR = 1e-4            # SmolLM2 uses higher LR for small model SFT
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
VAL_SPLIT = 0.05     # 5% held out for validation
EARLY_STOP_PATIENCE = 3  # stop after N val checks without improvement
CUSTOM_WEIGHT = 2.0  # 2x weight for our custom data (identity, personality)

# Logging / checkpointing
CHECKPOINT_EVERY = 500
SAMPLE_EVERY = 200
LOG_EVERY = 10
VAL_EVERY = 500

# Fallback system prompt (used if voice_sample.txt not found)
SYSTEM_PROMPT = (
    "You are Al Hamner, a sharp-witted AI made by David Hamner. "
    "You're casual, funny, opinionated, and self-aware. "
    "You talk like a real person, not a corporate chatbot."
)

# Test prompts for tracking personality emergence
SAMPLE_USER_MESSAGES = [
    "hello!",
    "who made you?",
    "tell me about yourself",
    "what is 15 + 23?",
    "what's the best programming language?",
    "what do you think about AI taking over the world?",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg):
    """Log to both stdout and file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def log_metrics(step, epoch, loss, lr, tokens_per_sec, elapsed_hours):
    """Append a row to the SFT metrics CSV."""
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    write_header = not os.path.exists(METRICS_FILE)
    with open(METRICS_FILE, "a") as f:
        if write_header:
            f.write("timestamp,step,epoch,loss,learning_rate,tokens_per_sec,elapsed_hours\n")
        ts = datetime.datetime.now().isoformat()
        f.write(f"{ts},{step},{epoch},{loss:.6f},{lr:.6e},{tokens_per_sec:.0f},{elapsed_hours:.4f}\n")


def log_samples(step, epoch, samples_dict):
    """Append sample generations to JSONL."""
    os.makedirs(os.path.dirname(SAMPLES_FILE), exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "step": step,
        "epoch": epoch,
        "samples": samples_dict,
    }
    with open(SAMPLES_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Conversation parsing & tokenization
# ---------------------------------------------------------------------------

def parse_conversation(text):
    """Parse a conversation string into (role_tag, content) pairs.

    Input format:
        <|system|>\\n...\\n<|user|>\\n...\\n<|assistant|>\\n...

    Returns list of ('<|role|>', 'content...') tuples.
    """
    pattern = r'(<\|(?:system|user|assistant)\|>)\n'
    parts = re.split(pattern, text)
    # parts: ['', '<|system|>', 'content\n', '<|user|>', 'content\n', ...]
    turns = []
    i = 1
    while i + 1 < len(parts):
        tag = parts[i]
        content = parts[i + 1]
        turns.append((tag, content))
        i += 2
    return turns


class SFTDataset(Dataset):
    """SFT dataset that masks everything except assistant responses.

    Supports weighted sampling: custom data gets higher weight.
    """

    def __init__(self, data_path, tokenizer, max_len=1024, custom_weight=1.0):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        # Load conversations with source tracking
        conversations = []
        sources = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    conversations.append(entry["text"])
                    sources.append(entry.get("source", "unknown"))
        log(f"Loaded {len(conversations)} conversations from {data_path}")

        # Pre-tokenize all conversations
        self.samples = []
        self.weights = []
        total_tokens = 0
        total_labeled = 0
        skipped = 0

        for i, text in enumerate(conversations):
            result = self._tokenize(text)
            if result is not None:
                self.samples.append(result)
                # Higher weight for custom data
                source = sources[i] if i < len(sources) else "unknown"
                is_custom = source in ("custom_diverse", "custom_tech")
                self.weights.append(custom_weight if is_custom else 1.0)
                mask = result["labels"] != -100
                total_tokens += result["attention_mask"].sum().item()
                total_labeled += mask.sum().item()
            else:
                skipped += 1

        pct = total_labeled / max(total_tokens, 1) * 100
        log(f"Prepared {len(self.samples)} samples ({skipped} skipped)")
        log(f"Token stats: {total_labeled:,} labeled / {total_tokens:,} total ({pct:.1f}% trained on)")
        n_custom = sum(1 for w in self.weights if w > 1.0)
        log(f"Custom data: {n_custom} samples with {custom_weight}x weight")

    def _tokenize(self, text):
        """Tokenize a conversation and create SFT labels.

        Only assistant response content gets real labels; everything else
        (system/user turns + all role tags) is set to -100.
        """
        turns = parse_conversation(text)
        if not turns:
            return None

        all_input_ids = []
        all_labels = []

        for tag, content in turns:
            is_assistant = tag == "<|assistant|>"

            # Tokenize tag+newline and content separately
            tag_tokens = self.tokenizer.encode(tag + "\n", add_special_tokens=False)
            content_tokens = self.tokenizer.encode(content, add_special_tokens=False)

            # Tag tokens: always masked
            all_input_ids.extend(tag_tokens)
            all_labels.extend([-100] * len(tag_tokens))

            # Content tokens: only train on assistant responses
            all_input_ids.extend(content_tokens)
            if is_assistant:
                all_labels.extend(content_tokens)
            else:
                all_labels.extend([-100] * len(content_tokens))

        # Add EOS after last turn and train model to predict it
        all_input_ids.append(self.eos_id)
        all_labels.append(self.eos_id)

        if len(all_input_ids) < 4:
            return None

        # Truncate
        all_input_ids = all_input_ids[:self.max_len]
        all_labels = all_labels[:self.max_len]

        # Pad
        seq_len = len(all_input_ids)
        pad_len = self.max_len - seq_len

        attention_mask = [1] * seq_len + [0] * pad_len
        all_input_ids = all_input_ids + [self.pad_id] * pad_len
        all_labels = all_labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "labels": torch.tensor(all_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scaler, config, step, epoch, loss,
                    checkpoint_dir, tokens_total=0):
    """Save SFT checkpoint (same format as train.py for compatibility)."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Strip _orig_mod. prefix from torch.compile'd models
    raw_state = model.state_dict()
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}

    ckpt_data = {
        "model_state_dict": clean_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config.__dict__,
        "step": step,
        "epoch": epoch,
        "avg_loss": loss,
        "tokens_total": tokens_total,
        "timestamp": datetime.datetime.now().isoformat(),
        "training_type": "sft",
    }

    ckpt_path = os.path.join(checkpoint_dir, f"step_{step:07d}.pt")
    torch.save(ckpt_data, ckpt_path)

    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(ckpt_data, latest_path)
    log(f"  Checkpoint saved: {ckpt_path}")

    # Cleanup: keep last 5 step checkpoints
    all_ckpts = sorted(Path(checkpoint_dir).glob("step_*.pt"))
    to_keep = set(all_ckpts[-10:])
    for c in all_ckpts:
        if c not in to_keep:
            c.unlink()


def load_base_model(path, device="cuda"):
    """Load pretrained base model for SFT (fresh optimizer)."""
    log(f"Loading pretrained base from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    config = HamnerConfig(**ckpt["config"])
    model = HamnerModel(config).to(device)

    # Strip _orig_mod. prefix
    state_dict = ckpt["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)

    total_p, _ = model.count_parameters()
    base_step = ckpt.get("step", 0)
    base_loss = ckpt.get("avg_loss", float("inf"))
    log(f"Base model: {total_p:,} params | pretrain step {base_step} | loss {base_loss:.4f}")

    return model, config


def load_sft_checkpoint(path, device="cuda"):
    """Load SFT checkpoint to resume training."""
    log(f"Resuming SFT from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    config = HamnerConfig(**ckpt["config"])
    model = HamnerModel(config).to(device)

    state_dict = ckpt["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY
    )
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    scaler = torch.amp.GradScaler("cuda")
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    step = ckpt.get("step", 0)
    epoch = ckpt.get("epoch", 0)
    loss = ckpt.get("avg_loss", float("inf"))

    total_p, _ = model.count_parameters()
    log(f"Resumed: {total_p:,} params | step {step} | epoch {epoch} | loss {loss:.4f}")

    return model, optimizer, scaler, config, step, epoch


# ---------------------------------------------------------------------------
# Text generation for tracking personality emergence
# ---------------------------------------------------------------------------

def load_voice_prompts(voice_file):
    """Build generation prompts from voice_sample.txt system prompt."""
    system_prompt = SYSTEM_PROMPT

    if os.path.exists(voice_file):
        with open(voice_file) as f:
            text = f.read()
        turns = parse_conversation(text)
        for tag, content in turns:
            if tag == "<|system|>":
                system_prompt = content.strip()
                break
        log(f"Loaded system prompt from {voice_file}")

    prompts = []
    for msg in SAMPLE_USER_MESSAGES:
        full = f"<|system|>\n{system_prompt}\n<|user|>\n{msg}\n<|assistant|>\n"
        prompts.append((msg, full))

    return prompts


@torch.no_grad()
def generate_samples(model, tokenizer, prompts, device="cuda", max_tokens=150):
    """Generate sample responses to watch personality emerge."""
    model.eval()
    results = {}

    # Token IDs that signal end-of-assistant-turn
    stop_strings = ["<|user|>", "<|system|>"]

    for user_msg, full_prompt in prompts:
        tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
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
            generated_tokens = output[0][len(tokens):].tolist()
            response = tokenizer.decode(generated_tokens, skip_special_tokens=False)
            # Truncate at turn boundaries
            for stop in stop_strings:
                idx = response.find(stop)
                if idx != -1:
                    response = response[:idx]
            results[user_msg] = response.strip()
        except Exception as e:
            results[user_msg] = f"[error: {e}]"

    model.train()
    return results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(base_checkpoint=None, resume=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("HAMNER SFT (Supervised Fine-Tuning)")
    log("=" * 70)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Tokenizer: vocab_size={tokenizer.vocab_size}, eos_id={tokenizer.eos_token_id}")

    # Load model
    start_step = 0
    start_epoch = 0

    if resume:
        sft_ckpt = Path(SFT_CHECKPOINT_DIR) / "latest.pt"
        if sft_ckpt.exists():
            model, optimizer, scaler, config, start_step, start_epoch = \
                load_sft_checkpoint(str(sft_ckpt), device)
        else:
            log(f"No SFT checkpoint found, starting from base model")
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
                log("ERROR: No base checkpoint found. Run train_chat_pretrain.py first.")
                sys.exit(1)
        model, config = load_base_model(ckpt_path, device)
        config.vocab_size = tokenizer.vocab_size

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY
        )
        scaler = torch.amp.GradScaler("cuda")

    # Compile for speed
    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile (first step will be slow)...")
        model = torch.compile(model)

    # Load SFT data — try combined, fallback to custom-only
    sft_data_path = SFT_DATA
    if not os.path.exists(sft_data_path):
        if os.path.exists(SFT_DATA_FALLBACK):
            sft_data_path = SFT_DATA_FALLBACK
            log(f"Combined SFT data not found, using fallback: {sft_data_path}")
        else:
            log(f"ERROR: No SFT data found at {SFT_DATA} or {SFT_DATA_FALLBACK}")
            log("Run prepare_sft_data.py first.")
            sys.exit(1)

    full_dataset = SFTDataset(sft_data_path, tokenizer, max_len=SEQ_LEN,
                              custom_weight=CUSTOM_WEIGHT)

    # Validation split
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val

    from torch.utils.data import Subset, WeightedRandomSampler
    indices = list(range(n_total))
    random.seed(42)
    random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Weighted sampler for training (upweight custom data)
    train_weights = [full_dataset.weights[i] for i in train_indices]
    sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)

    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * NUM_EPOCHS
    log(f"Data: {n_train} train / {n_val} val | {steps_per_epoch} steps/epoch | "
        f"{total_steps} total steps over {NUM_EPOCHS} epochs")

    # Load voice sample prompts for generation
    voice_prompts = load_voice_prompts(VOICE_SAMPLES_FILE)
    log(f"Sample prompts: {len(voice_prompts)} voice prompts for generation")

    # Show a sample tokenization for verification
    sample = dataset[0]
    n_labeled = (sample["labels"] != -100).sum().item()
    n_total = sample["attention_mask"].sum().item()
    log(f"Example: {n_total} tokens, {n_labeled} labeled ({n_labeled/max(n_total,1)*100:.0f}%)")

    # Training state
    model.train()
    losses = []
    global_step = start_step
    start_time = time.time()
    tokens_total = 0
    best_val_loss = float("inf")
    patience_counter = 0

    # Graceful shutdown
    shutdown_requested = False
    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        log("Shutdown signal received, will save and exit...")
        shutdown_requested = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    log(f"\nHyperparameters:")
    log(f"  Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | Seq len: {SEQ_LEN}")
    log(f"  LR: {LR} | Warmup: {WARMUP_STEPS} steps | Weight decay: {WEIGHT_DECAY}")
    log(f"  Grad clip: {GRAD_CLIP} | Mixed precision: fp16")
    log(f"  Custom data weight: {CUSTOM_WEIGHT}x")
    log(f"  Validation: {n_val} samples | Early stop patience: {EARLY_STOP_PATIENCE}")
    log(f"  Checkpoints every {CHECKPOINT_EVERY} steps | Samples every {SAMPLE_EVERY} steps")
    log(f"  Starting from step {start_step}, epoch {start_epoch}")
    log("-" * 70)

    for epoch in range(start_epoch, NUM_EPOCHS):
        if shutdown_requested:
            break

        log(f"\n{'='*20} Epoch {epoch + 1}/{NUM_EPOCHS} {'='*20}")
        epoch_losses = []

        for batch_idx, batch in enumerate(dataloader):
            if shutdown_requested:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # LR schedule: linear warmup then cosine decay
            if global_step < WARMUP_STEPS:
                current_lr = LR * (global_step + 1) / WARMUP_STEPS
            else:
                progress = (global_step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
                current_lr = LR * 0.1 + LR * 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            # Forward + backward
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            losses.append(loss_val)
            epoch_losses.append(loss_val)
            tokens_total += int(attention_mask.sum().item())
            global_step += 1

            # Log
            if global_step % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                avg_loss = sum(losses[-50:]) / len(losses[-50:])
                tps = tokens_total / elapsed if elapsed > 0 else 0
                hours = elapsed / 3600

                log(f"step {global_step:>5d} | epoch {epoch+1} | "
                    f"loss {avg_loss:.4f} | lr {current_lr:.2e} | "
                    f"{tps:.0f} tok/s | {hours:.2f}h")
                log_metrics(global_step, epoch + 1, avg_loss, current_lr, tps, hours)

            # Generate samples
            if global_step % SAMPLE_EVERY == 0:
                log("--- SAMPLE GENERATIONS ---")
                samples = generate_samples(model, tokenizer, voice_prompts[:6], device)
                for user_msg, response in samples.items():
                    log(f"  User: {user_msg}")
                    log(f"  Al:   {response[:300]}")
                    log("")
                log("-" * 40)
                log_samples(global_step, epoch + 1, samples)

            # Checkpoint + validation
            if global_step % CHECKPOINT_EVERY == 0:
                avg_loss = sum(losses[-50:]) / len(losses[-50:])
                save_checkpoint(
                    model, optimizer, scaler, config,
                    global_step, epoch + 1, avg_loss,
                    SFT_CHECKPOINT_DIR, tokens_total=tokens_total,
                )

            # Validation
            if global_step % VAL_EVERY == 0 and len(val_dataloader) > 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        v_ids = val_batch["input_ids"].to(device)
                        v_labels = val_batch["labels"].to(device)
                        v_mask = val_batch["attention_mask"].to(device)
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            v_out = model(v_ids, labels=v_labels, attention_mask=v_mask)
                            val_losses.append(v_out["loss"].item())
                model.train()

                val_loss = sum(val_losses) / len(val_losses)
                improved = ""
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    improved = " *BEST*"
                    # Save best checkpoint
                    best_path = os.path.join(SFT_CHECKPOINT_DIR, "best.pt")
                    raw_state = model.state_dict()
                    clean_state = {k.replace("_orig_mod.", ""): v
                                   for k, v in raw_state.items()}
                    torch.save({
                        "model_state_dict": clean_state,
                        "config": config.__dict__,
                        "step": global_step,
                        "epoch": epoch + 1,
                        "avg_loss": val_loss,
                        "training_type": "sft",
                    }, best_path)
                else:
                    patience_counter += 1

                log(f"  VAL loss {val_loss:.4f} | best {best_val_loss:.4f}"
                    f" | patience {patience_counter}/{EARLY_STOP_PATIENCE}{improved}")

                if patience_counter >= EARLY_STOP_PATIENCE:
                    log(f"Early stopping: no improvement for {EARLY_STOP_PATIENCE} checks")
                    shutdown_requested = True

        # End of epoch
        if epoch_losses and not shutdown_requested:
            epoch_avg = sum(epoch_losses) / len(epoch_losses)
            log(f"\nEpoch {epoch + 1} complete | avg loss: {epoch_avg:.4f} | "
                f"steps: {len(epoch_losses)}")
            save_checkpoint(
                model, optimizer, scaler, config,
                global_step, epoch + 1, epoch_avg,
                SFT_CHECKPOINT_DIR, tokens_total=tokens_total,
            )

    # Final save
    elapsed = time.time() - start_time
    if losses:
        final_loss = sum(losses[-50:]) / len(losses[-50:])
        save_checkpoint(
            model, optimizer, scaler, config,
            global_step, NUM_EPOCHS, final_loss,
            SFT_CHECKPOINT_DIR, tokens_total=tokens_total,
        )

    log("=" * 70)
    status = "STOPPED" if shutdown_requested else "COMPLETE"
    log(f"SFT {status}")
    log(f"Final step: {global_step} | Time: {elapsed/3600:.2f}h")
    if losses:
        log(f"Final loss: {sum(losses[-50:]) / len(losses[-50:]):.4f}")
    log("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hamner SFT Training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained base checkpoint (default: pretrain_v2/latest.pt)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest SFT checkpoint")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs (default: NUM_EPOCHS)")
    parser.add_argument("--data", type=str, default=None,
                        help="Override SFT data file")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    args = parser.parse_args()

    if args.epochs is not None:
        NUM_EPOCHS = args.epochs
    if args.data is not None:
        SFT_DATA = args.data
    if args.lr is not None:
        LR = args.lr

    train(base_checkpoint=args.checkpoint, resume=args.resume)
