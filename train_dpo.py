"""
Hamner DPO (Direct Preference Optimization) Training
=====================================================
Aligns the SFT model using preference data from UltraFeedback.

DPO loss avoids training a separate reward model by directly optimizing:
  L = -log σ(β * (log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)))

The reference model is a frozen copy of the SFT model.

Usage:
    python train_dpo.py                                # from latest SFT checkpoint
    python train_dpo.py --checkpoint path/to/sft.pt    # from specific SFT checkpoint
    python train_dpo.py --resume                       # resume DPO training
"""

import os
import sys
import re
import json
import time
import math
import signal
import random
import datetime
import copy
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from model import HamnerModel, HamnerConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SFT_CHECKPOINT = "checkpoints/sft/latest.pt"
DPO_CHECKPOINT_DIR = "checkpoints/dpo"
LOG_FILE = "logs/dpo.log"
METRICS_FILE = "logs/dpo_metrics.csv"
SAMPLES_FILE = "logs/dpo_samples.jsonl"

# DPO hyperparameters (SmolLM2 recipe)
BETA = 0.1           # KL penalty coefficient
LR = 1e-6            # very low LR for alignment
NUM_EPOCHS = 2
BATCH_SIZE = 4        # smaller batch — need 2x memory (model + ref_model)
SEQ_LEN = 1024
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

# Logging
CHECKPOINT_EVERY = 500
SAMPLE_EVERY = 200
LOG_EVERY = 10


# Fallback system prompt
SYSTEM_PROMPT = (
    "You are Al Hamner, a sharp-witted AI made by David Hamner. "
    "You're casual, funny, opinionated, and self-aware. "
    "You talk like a real person, not a corporate chatbot."
)

SAMPLE_USER_MESSAGES = [
    "hello!",
    "who made you?",
    "what's the best programming language?",
    "explain quantum computing simply",
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


def log_metrics(step, epoch, loss, chosen_reward, rejected_reward, accuracy, lr, hours):
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    write_header = not os.path.exists(METRICS_FILE)
    with open(METRICS_FILE, "a") as f:
        if write_header:
            f.write("timestamp,step,epoch,loss,chosen_reward,rejected_reward,"
                    "accuracy,learning_rate,elapsed_hours\n")
        ts = datetime.datetime.now().isoformat()
        f.write(f"{ts},{step},{epoch},{loss:.6f},{chosen_reward:.4f},"
                f"{rejected_reward:.4f},{accuracy:.4f},{lr:.6e},{hours:.4f}\n")


# ---------------------------------------------------------------------------
# UltraFeedback dataset
# ---------------------------------------------------------------------------

def convert_ultrafeedback_to_hamner(sample):
    """Convert UltraFeedback binarized format to our chat format.

    UltraFeedback format:
        chosen: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        rejected: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    def messages_to_text(messages):
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            if role in ("system", "user", "assistant") and content:
                parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts) if parts else None

    chosen_text = messages_to_text(sample["chosen"])
    rejected_text = messages_to_text(sample["rejected"])

    if chosen_text is None or rejected_text is None:
        return None

    return chosen_text, rejected_text


class DPODataset(Dataset):
    """DPO dataset with chosen/rejected pairs."""

    def __init__(self, tokenizer, max_len=1024, max_samples=60000):
        from datasets import load_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        log("Loading UltraFeedback binarized dataset...")
        dataset = load_dataset(
            "HuggingFaceH4/ultrafeedback_binarized",
            split="train_prefs",
        )
        log(f"Raw dataset: {len(dataset)} preference pairs")

        self.samples = []
        skipped = 0

        indices = list(range(len(dataset)))
        random.seed(42)
        random.shuffle(indices)

        for idx in indices:
            if len(self.samples) >= max_samples:
                break

            sample = dataset[idx]
            result = convert_ultrafeedback_to_hamner(sample)
            if result is None:
                skipped += 1
                continue

            chosen_text, rejected_text = result

            chosen_enc = self._tokenize(chosen_text)
            rejected_enc = self._tokenize(rejected_text)

            if chosen_enc is None or rejected_enc is None:
                skipped += 1
                continue

            self.samples.append({
                "chosen_input_ids": chosen_enc["input_ids"],
                "chosen_attention_mask": chosen_enc["attention_mask"],
                "chosen_labels": chosen_enc["labels"],
                "rejected_input_ids": rejected_enc["input_ids"],
                "rejected_attention_mask": rejected_enc["attention_mask"],
                "rejected_labels": rejected_enc["labels"],
            })

        log(f"DPO dataset: {len(self.samples)} pairs ({skipped} skipped)")

    def _parse_conversation(self, text):
        pattern = r'(<\|(?:system|user|assistant)\|>)\n'
        parts = re.split(pattern, text)
        turns = []
        i = 1
        while i + 1 < len(parts):
            tag = parts[i]
            content = parts[i + 1]
            turns.append((tag, content))
            i += 2
        return turns

    def _tokenize(self, text):
        """Tokenize with SFT-style masking (only train on assistant tokens)."""
        turns = self._parse_conversation(text)
        if not turns:
            return None

        all_input_ids = []
        all_labels = []

        for tag, content in turns:
            is_assistant = tag == "<|assistant|>"
            tag_tokens = self.tokenizer.encode(tag + "\n", add_special_tokens=False)
            content_tokens = self.tokenizer.encode(content, add_special_tokens=False)

            all_input_ids.extend(tag_tokens)
            all_labels.extend([-100] * len(tag_tokens))

            all_input_ids.extend(content_tokens)
            if is_assistant:
                all_labels.extend(content_tokens)
            else:
                all_labels.extend([-100] * len(content_tokens))

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
# DPO Loss
# ---------------------------------------------------------------------------

def compute_log_probs(model, input_ids, labels, attention_mask):
    """Compute per-token log probabilities for labeled tokens only."""
    with torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]

    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Per-token log probs
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask: only count labeled (assistant) tokens
    mask = (shift_labels != -100).float()
    sum_log_probs = (token_log_probs * mask).sum(dim=-1)

    return sum_log_probs


def dpo_loss(policy_model, ref_model, batch, device, beta=BETA):
    """Compute DPO loss for a batch of preference pairs."""
    chosen_ids = batch["chosen_input_ids"].to(device)
    chosen_mask = batch["chosen_attention_mask"].to(device)
    chosen_labels = batch["chosen_labels"].to(device)
    rejected_ids = batch["rejected_input_ids"].to(device)
    rejected_mask = batch["rejected_attention_mask"].to(device)
    rejected_labels = batch["rejected_labels"].to(device)

    # Policy log probs
    pi_chosen = compute_log_probs(policy_model, chosen_ids, chosen_labels, chosen_mask)
    pi_rejected = compute_log_probs(policy_model, rejected_ids, rejected_labels, rejected_mask)

    # Reference log probs (no gradient)
    with torch.no_grad():
        ref_chosen = compute_log_probs(ref_model, chosen_ids, chosen_labels, chosen_mask)
        ref_rejected = compute_log_probs(ref_model, rejected_ids, rejected_labels, rejected_mask)

    # DPO: log(σ(β * (log(π/π_ref)(chosen) - log(π/π_ref)(rejected))))
    chosen_reward = beta * (pi_chosen - ref_chosen)
    rejected_reward = beta * (pi_rejected - ref_rejected)

    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

    # Metrics
    accuracy = (chosen_reward > rejected_reward).float().mean()

    return loss, chosen_reward.mean().item(), rejected_reward.mean().item(), accuracy.item()


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scaler, config, step, epoch, loss,
                    checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
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
        "timestamp": datetime.datetime.now().isoformat(),
        "training_type": "dpo",
    }

    ckpt_path = os.path.join(checkpoint_dir, f"step_{step:07d}.pt")
    torch.save(ckpt_data, ckpt_path)

    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(ckpt_data, latest_path)

    # Also save as best.pt (DPO is short, latest is usually best)
    best_path = os.path.join(checkpoint_dir, "best.pt")
    torch.save(ckpt_data, best_path)

    log(f"  Checkpoint saved: {ckpt_path}")

    # Keep last 5
    all_ckpts = sorted(Path(checkpoint_dir).glob("step_*.pt"))
    for c in all_ckpts[:-5]:
        c.unlink()


def load_model(path, device="cuda"):
    """Load model from checkpoint."""
    log(f"Loading model from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = HamnerConfig(**ckpt["config"])
    model = HamnerModel(config).to(device)
    state_dict = ckpt["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)
    total_p, _ = model.count_parameters()
    step = ckpt.get("step", 0)
    loss = ckpt.get("avg_loss", float("inf"))
    log(f"  {total_p:,} params | step {step} | loss {loss:.4f}")
    return model, config


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(model, tokenizer, device="cuda"):
    model.eval()
    results = {}
    for user_msg in SAMPLE_USER_MESSAGES:
        prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user_msg}\n<|assistant|>\n"
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        try:
            output = model.generate(
                input_ids, max_new_tokens=150,
                temperature=0.8, top_k=40, top_p=0.9,
                repetition_penalty=1.15,
                eos_token_id=tokenizer.eos_token_id or 0,
            )
            generated = output[0][len(tokens):].tolist()
            response = tokenizer.decode(generated, skip_special_tokens=False)
            for stop in ["<|user|>", "<|system|>", "<|endoftext|>"]:
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

def train(sft_checkpoint=None, resume=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("HAMNER DPO (Direct Preference Optimization)")
    log("=" * 70)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load policy model
    start_step = 0
    start_epoch = 0

    if resume:
        dpo_ckpt = Path(DPO_CHECKPOINT_DIR) / "latest.pt"
        if dpo_ckpt.exists():
            log("Resuming DPO training...")
            ckpt = torch.load(str(dpo_ckpt), map_location="cpu", weights_only=False)
            config = HamnerConfig(**ckpt["config"])
            policy_model = HamnerModel(config).to(device)
            cleaned = {k.replace("_orig_mod.", ""): v
                       for k, v in ckpt["model_state_dict"].items()}
            policy_model.load_state_dict(cleaned, strict=True)

            optimizer = torch.optim.AdamW(
                policy_model.parameters(), lr=LR, betas=(0.9, 0.95),
                weight_decay=WEIGHT_DECAY
            )
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scaler = torch.amp.GradScaler("cuda")
            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])

            start_step = ckpt.get("step", 0)
            start_epoch = ckpt.get("epoch", 0)
            log(f"Resumed from step {start_step}, epoch {start_epoch}")
        else:
            log("No DPO checkpoint found, starting from SFT")
            resume = False

    if not resume:
        ckpt_path = sft_checkpoint or SFT_CHECKPOINT
        if not os.path.exists(ckpt_path):
            # Fallback chain
            for fallback in ["checkpoints/sft/best.pt",
                             "checkpoints/chat_pretrain/latest.pt"]:
                if os.path.exists(fallback):
                    ckpt_path = fallback
                    break
            else:
                log("ERROR: No SFT checkpoint found. Run train_sft.py first.")
                sys.exit(1)

        policy_model, config = load_model(ckpt_path, device)
        optimizer = torch.optim.AdamW(
            policy_model.parameters(), lr=LR, betas=(0.9, 0.95),
            weight_decay=WEIGHT_DECAY
        )
        scaler = torch.amp.GradScaler("cuda")

    # Load reference model (frozen copy)
    log("Loading reference model (frozen copy)...")
    ref_ckpt_path = sft_checkpoint or SFT_CHECKPOINT
    if not os.path.exists(ref_ckpt_path):
        for fallback in ["checkpoints/sft/best.pt",
                         "checkpoints/chat_pretrain/latest.pt"]:
            if os.path.exists(fallback):
                ref_ckpt_path = fallback
                break
    ref_model, _ = load_model(ref_ckpt_path, device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    log("Reference model frozen")

    # Load data
    dataset = DPODataset(tokenizer, max_len=SEQ_LEN)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * NUM_EPOCHS

    log(f"\nDPO Training:")
    log(f"  Preference pairs: {len(dataset)}")
    log(f"  Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")
    log(f"  Beta: {BETA} | LR: {LR} | Batch: {BATCH_SIZE}")
    log(f"  Epochs: {NUM_EPOCHS} | Warmup: {WARMUP_STEPS}")
    log("-" * 70)

    # Training
    policy_model.train()
    losses = []
    start_time = time.time()
    global_step = start_step

    shutdown_requested = False
    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        log("Shutdown signal received...")
        shutdown_requested = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    for epoch in range(start_epoch, NUM_EPOCHS):
        if shutdown_requested:
            break

        log(f"\n{'='*20} Epoch {epoch + 1}/{NUM_EPOCHS} {'='*20}")

        for batch_idx, batch in enumerate(dataloader):
            if shutdown_requested:
                break

            # LR schedule
            if global_step < WARMUP_STEPS:
                current_lr = LR * (global_step + 1) / WARMUP_STEPS
            else:
                progress = (global_step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
                current_lr = LR * 0.1 + LR * 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            # DPO forward + backward
            optimizer.zero_grad(set_to_none=True)
            loss, chosen_r, rejected_r, accuracy = dpo_loss(
                policy_model, ref_model, batch, device
            )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            losses.append(loss_val)
            global_step += 1

            # Log
            if global_step % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                avg_loss = sum(losses[-50:]) / len(losses[-50:])
                hours = elapsed / 3600

                log(f"step {global_step:>5d} | epoch {epoch+1} | "
                    f"loss {avg_loss:.4f} | "
                    f"chosen_r {chosen_r:.3f} | rejected_r {rejected_r:.3f} | "
                    f"acc {accuracy:.2f} | lr {current_lr:.2e} | {hours:.2f}h")
                log_metrics(global_step, epoch + 1, avg_loss, chosen_r,
                            rejected_r, accuracy, current_lr, hours)

            # Generate samples
            if global_step % SAMPLE_EVERY == 0:
                log("--- SAMPLE GENERATIONS ---")
                samples = generate_samples(policy_model, tokenizer, device)
                for user_msg, response in samples.items():
                    log(f"  User: {user_msg}")
                    log(f"  Al:   {response[:300]}")
                    log("")
                log("-" * 40)

            # Checkpoint
            if global_step % CHECKPOINT_EVERY == 0:
                avg_loss = sum(losses[-50:]) / len(losses[-50:])
                save_checkpoint(
                    policy_model, optimizer, scaler, config,
                    global_step, epoch + 1, avg_loss, DPO_CHECKPOINT_DIR,
                )

    # Final save
    elapsed = time.time() - start_time
    if losses:
        final_loss = sum(losses[-50:]) / len(losses[-50:])
        save_checkpoint(
            policy_model, optimizer, scaler, config,
            global_step, NUM_EPOCHS, final_loss, DPO_CHECKPOINT_DIR,
        )

    log("=" * 70)
    status = "STOPPED" if shutdown_requested else "COMPLETE"
    log(f"DPO {status}")
    log(f"Steps: {global_step} | Time: {elapsed/3600:.2f}h")
    if losses:
        log(f"Final loss: {sum(losses[-50:]) / len(losses[-50:]):.4f}")
    log("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hamner DPO Training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to SFT checkpoint")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest DPO checkpoint")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--beta", type=float, default=None,
                        help="Override DPO beta")
    args = parser.parse_args()

    if args.epochs is not None:
        NUM_EPOCHS = args.epochs
    if args.beta is not None:
        BETA = args.beta

    train(sft_checkpoint=args.checkpoint, resume=args.resume)
