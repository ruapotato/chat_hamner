"""
Tournament Training for Hamner Architecture Variants
=====================================================
Trains 10 architecture variants, eliminates the worst performers,
and continues training the survivors. The winner gets full training.

Tournament structure:
  Round 1: All 10 variants, 500 steps each (~10 min each)
  Round 2: Top 5 variants, 1000 more steps (~15 min each)
  Round 3: Top 3 variants, 2000 more steps (~20 min each)
  Final:   Winner trains for remaining time with full dataset
"""

import os
import sys
import json
import time
import math
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from rich import print as rprint

from model import HamnerModel, HamnerConfig
from variants import get_variant_configs, describe_variant

console = Console()

# ---------------------------------------------------------------------------
# Simple streaming dataset that pulls from HuggingFace on-the-fly
# ---------------------------------------------------------------------------

class StreamingTextDataset(Dataset):
    """Tokenizes text from HuggingFace datasets on-the-fly."""

    def __init__(self, tokenizer, seq_len=512, num_samples=50000, dataset_name="HuggingFaceFW/fineweb-edu",
                 dataset_config="sample-10BT", split="train"):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_samples = num_samples

        console.print(f"[bold blue]Loading dataset {dataset_name}...[/bold blue]")
        from datasets import load_dataset
        self.dataset = load_dataset(
            dataset_name, name=dataset_config, split=split, streaming=True
        )
        self._cache = []
        self._token_buffer = []
        self._iter = iter(self.dataset)
        self._exhausted = False

    def _fill_buffer(self):
        """Fill the token buffer with more data."""
        while len(self._token_buffer) < self.seq_len + 1 and not self._exhausted:
            try:
                sample = next(self._iter)
                text = sample.get("text", "")
                if len(text.strip()) < 50:
                    continue
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                tokens.append(self.tokenizer.eos_token_id or 0)
                self._token_buffer.extend(tokens)
            except StopIteration:
                self._exhausted = True
                break

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Use cached if available
        if idx < len(self._cache):
            return self._cache[idx]

        # Fill buffer and extract a chunk
        self._fill_buffer()

        if len(self._token_buffer) >= self.seq_len + 1:
            chunk = self._token_buffer[:self.seq_len + 1]
            self._token_buffer = self._token_buffer[self.seq_len:]
        else:
            # Pad if we ran out
            chunk = self._token_buffer + [0] * (self.seq_len + 1 - len(self._token_buffer))
            self._token_buffer = []

        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)

        result = {"input_ids": input_ids, "labels": labels}
        self._cache.append(result)
        return result


class PreTokenizedDataset(Dataset):
    """Dataset from pre-tokenized binary shards (for full training)."""

    def __init__(self, data_dir, seq_len=2048):
        self.seq_len = seq_len
        shard_files = sorted(Path(data_dir).glob("shard_*.bin"))
        if not shard_files:
            raise FileNotFoundError(f"No shard files found in {data_dir}")
        # Memory-map all shards
        self.tokens = np.concatenate([
            np.fromfile(str(f), dtype=np.uint16) for f in shard_files
        ])
        self.num_samples = len(self.tokens) // (seq_len + 1)
        console.print(f"[bold green]Loaded {len(self.tokens):,} tokens from {len(shard_files)} shards[/bold green]")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        chunk = self.tokens[start:start + self.seq_len + 1].astype(np.int64)
        return {
            "input_ids": torch.from_numpy(chunk[:-1]),
            "labels": torch.from_numpy(chunk[1:]),
        }


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_variant(
    name: str,
    config: HamnerConfig,
    dataset: Dataset,
    num_steps: int,
    batch_size: int = 4,
    lr: float = 3e-4,
    warmup_steps: int = 100,
    checkpoint_dir: str = "checkpoints",
    existing_model: torch.nn.Module = None,
    existing_optimizer=None,
    existing_step: int = 0,
    device: str = "cuda",
):
    """Train a single variant for a given number of steps. Returns (model, optimizer, final_loss, step)."""

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create or reuse model
    if existing_model is not None:
        model = existing_model
        optimizer = existing_optimizer
        start_step = existing_step
    else:
        model = HamnerModel(config).to(device)
        total_params, _ = model.count_parameters()
        console.print(f"  [cyan]{name}[/cyan]: {total_params:,} params, {total_params*2/1e6:.1f}MB (fp16)")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
        start_step = 0

    model.train()
    scaler = torch.amp.GradScaler("cuda")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                           pin_memory=True, drop_last=True)
    data_iter = iter(dataloader)

    losses = []
    start_time = time.time()

    for step in range(start_step, start_step + num_steps):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # LR schedule: linear warmup then cosine decay
        total_steps = start_step + num_steps
        if step < warmup_steps:
            current_lr = lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            current_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Forward pass with AMP
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        losses.append(loss_val)

        # Log every 50 steps
        if (step + 1) % 50 == 0 or step == start_step:
            elapsed = time.time() - start_time
            tokens_per_sec = (step - start_step + 1) * batch_size * config.max_seq_len / elapsed
            avg_loss = sum(losses[-50:]) / len(losses[-50:])
            console.print(
                f"  [{name}] step {step+1}/{start_step+num_steps} | "
                f"loss={avg_loss:.4f} | lr={current_lr:.2e} | "
                f"{tokens_per_sec:.0f} tok/s"
            )

    # Save checkpoint
    ckpt_path = os.path.join(checkpoint_dir, f"{name}_step{start_step+num_steps}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.__dict__,
        "step": start_step + num_steps,
        "loss": losses[-1] if losses else float("inf"),
        "avg_loss_last_100": sum(losses[-100:]) / len(losses[-100:]) if losses else float("inf"),
    }, ckpt_path)
    console.print(f"  [green]Saved checkpoint: {ckpt_path}[/green]")

    avg_final_loss = sum(losses[-100:]) / len(losses[-100:]) if losses else float("inf")

    # Clean up GPU memory
    torch.cuda.empty_cache()

    return model, optimizer, avg_final_loss, start_step + num_steps


def evaluate_variant(model, dataset, batch_size=4, num_eval_steps=50, device="cuda"):
    """Evaluate on held-out data."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                           pin_memory=True, drop_last=True)
    data_iter = iter(dataloader)

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for _ in range(num_eval_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids, labels=labels)

            total_loss += outputs["loss"].item()
            count += 1

    model.train()
    return total_loss / count if count > 0 else float("inf")


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------

def run_tournament(
    checkpoint_dir: str = "checkpoints",
    seq_len: int = 512,       # shorter for tournament (faster)
    batch_size: int = 4,
    lr: float = 3e-4,
    round1_steps: int = 500,
    round2_steps: int = 1000,
    round3_steps: int = 2000,
    final_steps: int = 0,     # 0 = unlimited (train until interrupted)
    device: str = "cuda",
):
    """Run the full tournament."""

    console.print("\n[bold magenta]" + "=" * 70 + "[/bold magenta]")
    console.print("[bold magenta]  HAMNER ARCHITECTURE TOURNAMENT[/bold magenta]")
    console.print("[bold magenta]" + "=" * 70 + "[/bold magenta]\n")

    # Load tokenizer
    console.print("[bold]Loading tokenizer...[/bold]")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Update all configs with correct vocab size
    variants = get_variant_configs()
    for name, config in variants.items():
        config.vocab_size = tokenizer.vocab_size
        config.max_seq_len = seq_len

    # Create datasets
    console.print("[bold]Preparing training data (streaming from FineWeb-Edu)...[/bold]")
    train_dataset = StreamingTextDataset(
        tokenizer, seq_len=seq_len, num_samples=200000,
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
    )

    # Print all variants
    console.print("\n[bold yellow]CONTESTANTS:[/bold yellow]")
    table = Table(title="Architecture Variants")
    table.add_column("Name", style="cyan")
    table.add_column("Params (est)", style="green")
    table.add_column("Attention", style="magenta")
    table.add_column("MLP", style="blue")
    table.add_column("Layers", style="yellow")
    table.add_column("Hidden", style="white")

    for name, config in variants.items():
        est = config.total_params_estimate()
        attn = "DiffAttn" if config.use_differential_attention else "Standard"
        mlp = f"MoE-{config.num_experts}" if config.is_moe else "Dense"
        table.add_row(name, f"{est:,}", attn, mlp, str(config.num_layers), str(config.hidden_size))
    console.print(table)

    results_log = []

    # ==================== ROUND 1 ====================
    console.print(f"\n[bold red]{'='*70}[/bold red]")
    console.print(f"[bold red]  ROUND 1: All 10 variants x {round1_steps} steps[/bold red]")
    console.print(f"[bold red]{'='*70}[/bold red]\n")

    round1_results = {}
    models = {}
    optimizers = {}
    steps = {}

    for name, config in variants.items():
        console.print(f"\n[bold cyan]Training {name}...[/bold cyan]")
        try:
            model, opt, avg_loss, final_step = train_variant(
                name, config, train_dataset,
                num_steps=round1_steps, batch_size=batch_size, lr=lr,
                checkpoint_dir=os.path.join(checkpoint_dir, "round1"),
                device=device,
            )
            round1_results[name] = avg_loss
            models[name] = model
            optimizers[name] = opt
            steps[name] = final_step

            # Move model to CPU to save GPU memory
            model.cpu()
            torch.cuda.empty_cache()

        except Exception as e:
            console.print(f"  [red]FAILED: {e}[/red]")
            round1_results[name] = float("inf")

    # Show Round 1 results
    console.print(f"\n[bold yellow]ROUND 1 RESULTS:[/bold yellow]")
    sorted_r1 = sorted(round1_results.items(), key=lambda x: x[1])
    r1_table = Table(title="Round 1 Rankings")
    r1_table.add_column("Rank", style="bold")
    r1_table.add_column("Variant", style="cyan")
    r1_table.add_column("Avg Loss", style="green")
    r1_table.add_column("Status", style="magenta")

    top5_names = [name for name, _ in sorted_r1[:5]]
    for i, (name, loss) in enumerate(sorted_r1):
        status = "ADVANCES" if name in top5_names else "ELIMINATED"
        style = "green" if status == "ADVANCES" else "red"
        r1_table.add_row(str(i+1), name, f"{loss:.4f}", f"[{style}]{status}[/{style}]")
    console.print(r1_table)

    results_log.append({"round": 1, "results": dict(sorted_r1), "advanced": top5_names})

    # Clean up eliminated models
    for name in list(models.keys()):
        if name not in top5_names:
            del models[name]
            del optimizers[name]
            del steps[name]
    torch.cuda.empty_cache()

    # ==================== ROUND 2 ====================
    console.print(f"\n[bold red]{'='*70}[/bold red]")
    console.print(f"[bold red]  ROUND 2: Top 5 x {round2_steps} more steps[/bold red]")
    console.print(f"[bold red]{'='*70}[/bold red]\n")

    round2_results = {}
    for name in top5_names:
        config = variants[name]
        config.max_seq_len = seq_len
        console.print(f"\n[bold cyan]Continuing {name}...[/bold cyan]")
        try:
            models[name] = models[name].to(device)
            model, opt, avg_loss, final_step = train_variant(
                name, config, train_dataset,
                num_steps=round2_steps, batch_size=batch_size, lr=lr * 0.5,
                checkpoint_dir=os.path.join(checkpoint_dir, "round2"),
                existing_model=models[name],
                existing_optimizer=optimizers[name],
                existing_step=steps[name],
                device=device,
            )
            round2_results[name] = avg_loss
            models[name] = model.cpu()
            optimizers[name] = opt
            steps[name] = final_step
            torch.cuda.empty_cache()
        except Exception as e:
            console.print(f"  [red]FAILED: {e}[/red]")
            round2_results[name] = float("inf")

    sorted_r2 = sorted(round2_results.items(), key=lambda x: x[1])
    top3_names = [name for name, _ in sorted_r2[:3]]

    console.print(f"\n[bold yellow]ROUND 2 RESULTS:[/bold yellow]")
    r2_table = Table(title="Round 2 Rankings")
    r2_table.add_column("Rank", style="bold")
    r2_table.add_column("Variant", style="cyan")
    r2_table.add_column("Avg Loss", style="green")
    r2_table.add_column("Status", style="magenta")
    for i, (name, loss) in enumerate(sorted_r2):
        status = "ADVANCES" if name in top3_names else "ELIMINATED"
        style = "green" if status == "ADVANCES" else "red"
        r2_table.add_row(str(i+1), name, f"{loss:.4f}", f"[{style}]{status}[/{style}]")
    console.print(r2_table)

    results_log.append({"round": 2, "results": dict(sorted_r2), "advanced": top3_names})

    for name in list(models.keys()):
        if name not in top3_names:
            del models[name]
            del optimizers[name]
            del steps[name]
    torch.cuda.empty_cache()

    # ==================== ROUND 3 ====================
    console.print(f"\n[bold red]{'='*70}[/bold red]")
    console.print(f"[bold red]  ROUND 3: Top 3 x {round3_steps} more steps[/bold red]")
    console.print(f"[bold red]{'='*70}[/bold red]\n")

    round3_results = {}
    for name in top3_names:
        config = variants[name]
        config.max_seq_len = seq_len
        console.print(f"\n[bold cyan]Continuing {name}...[/bold cyan]")
        try:
            models[name] = models[name].to(device)
            model, opt, avg_loss, final_step = train_variant(
                name, config, train_dataset,
                num_steps=round3_steps, batch_size=batch_size, lr=lr * 0.25,
                checkpoint_dir=os.path.join(checkpoint_dir, "round3"),
                existing_model=models[name],
                existing_optimizer=optimizers[name],
                existing_step=steps[name],
                device=device,
            )
            round3_results[name] = avg_loss
            models[name] = model.cpu()
            optimizers[name] = opt
            steps[name] = final_step
            torch.cuda.empty_cache()
        except Exception as e:
            console.print(f"  [red]FAILED: {e}[/red]")
            round3_results[name] = float("inf")

    sorted_r3 = sorted(round3_results.items(), key=lambda x: x[1])
    winner_name = sorted_r3[0][0]

    console.print(f"\n[bold yellow]ROUND 3 RESULTS:[/bold yellow]")
    r3_table = Table(title="Round 3 Rankings")
    r3_table.add_column("Rank", style="bold")
    r3_table.add_column("Variant", style="cyan")
    r3_table.add_column("Avg Loss", style="green")
    r3_table.add_column("Status", style="magenta")
    for i, (name, loss) in enumerate(sorted_r3):
        status = "WINNER!" if name == winner_name else "Runner-up"
        style = "bold green" if status == "WINNER!" else "yellow"
        r3_table.add_row(str(i+1), name, f"{loss:.4f}", f"[{style}]{status}[/{style}]")
    console.print(r3_table)

    results_log.append({"round": 3, "results": dict(sorted_r3), "winner": winner_name})

    # Save tournament results
    results_path = os.path.join(checkpoint_dir, "tournament_results.json")
    with open(results_path, "w") as f:
        json.dump(results_log, f, indent=2)

    # ==================== FINAL: Extended training for winner ====================
    console.print(f"\n[bold magenta]{'='*70}[/bold magenta]")
    console.print(f"[bold magenta]  CHAMPION: {winner_name}[/bold magenta]")
    console.print(f"[bold magenta]  Starting extended training...[/bold magenta]")
    console.print(f"[bold magenta]{'='*70}[/bold magenta]\n")

    winner_config = variants[winner_name]
    # Scale up for final training
    winner_config.max_seq_len = 1024  # longer sequences now

    # Rebuild model with longer seq len (need to update RoPE)
    winner_model = HamnerModel(winner_config).to(device)
    # Load weights from tournament (seq_len agnostic except RoPE buffer which is re-created)
    # We need to be careful - the old model had shorter RoPE. The new one has longer.
    # Since RoPE is a buffer (not parameter), the state_dict loading handles it.
    old_state = models[winner_name].state_dict()
    # Filter out RoPE buffers as they'll be different size
    new_state = {k: v for k, v in old_state.items() if "rope_" not in k}
    winner_model.load_state_dict(new_state, strict=False)

    total_p, _ = winner_model.count_parameters()
    console.print(f"[bold green]Winner: {winner_name} ({total_p:,} params)[/bold green]")
    console.print(f"[bold green]{describe_variant(winner_name, winner_config)}[/bold green]")

    # Create new dataset with longer sequences
    final_dataset = StreamingTextDataset(
        tokenizer, seq_len=1024, num_samples=500000,
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
    )

    # Train continuously with checkpoints
    final_optimizer = torch.optim.AdamW(
        winner_model.parameters(), lr=lr * 0.1,
        betas=(0.9, 0.95), weight_decay=0.1
    )
    scaler = torch.amp.GradScaler("cuda")
    winner_model.train()

    final_dir = os.path.join(checkpoint_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=True,
                           num_workers=2, pin_memory=True, drop_last=True)

    step = 0
    best_loss = float("inf")
    losses = []
    start_time = time.time()

    console.print("[bold]Training winner continuously (Ctrl+C to stop, checkpoints saved every 500 steps)...[/bold]\n")

    try:
        while True:
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                # Cosine LR
                current_lr = lr * 0.1 * 0.5 * (1.0 + math.cos(math.pi * min(step / 50000, 1.0)))
                for pg in final_optimizer.param_groups:
                    pg["lr"] = current_lr

                final_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = winner_model(input_ids, labels=labels)
                    loss = outputs["loss"]

                scaler.scale(loss).backward()
                scaler.unscale_(final_optimizer)
                torch.nn.utils.clip_grad_norm_(winner_model.parameters(), 1.0)
                scaler.step(final_optimizer)
                scaler.update()

                loss_val = loss.item()
                losses.append(loss_val)
                step += 1

                if step % 100 == 0:
                    elapsed = time.time() - start_time
                    avg = sum(losses[-100:]) / len(losses[-100:])
                    tps = step * batch_size * 1024 / elapsed
                    hours = elapsed / 3600
                    console.print(
                        f"[{winner_name}] step {step} | loss={avg:.4f} | "
                        f"lr={current_lr:.2e} | {tps:.0f} tok/s | {hours:.1f}h elapsed"
                    )

                if step % 500 == 0:
                    # Save checkpoint
                    ckpt = os.path.join(final_dir, f"checkpoint_step{step}.pt")
                    torch.save({
                        "model_state_dict": winner_model.state_dict(),
                        "optimizer_state_dict": final_optimizer.state_dict(),
                        "config": winner_config.__dict__,
                        "step": step,
                        "avg_loss": sum(losses[-100:]) / len(losses[-100:]),
                        "variant_name": winner_name,
                    }, ckpt)
                    console.print(f"  [green]Checkpoint saved: {ckpt}[/green]")

                    # Also save as "latest"
                    latest = os.path.join(final_dir, "latest.pt")
                    torch.save({
                        "model_state_dict": winner_model.state_dict(),
                        "optimizer_state_dict": final_optimizer.state_dict(),
                        "config": winner_config.__dict__,
                        "step": step,
                        "avg_loss": sum(losses[-100:]) / len(losses[-100:]),
                        "variant_name": winner_name,
                    }, latest)

                if final_steps > 0 and step >= final_steps:
                    raise KeyboardInterrupt("Reached target steps")

    except KeyboardInterrupt:
        console.print(f"\n[bold yellow]Training stopped at step {step}[/bold yellow]")
        # Final save
        final_ckpt = os.path.join(final_dir, f"final_step{step}.pt")
        torch.save({
            "model_state_dict": winner_model.state_dict(),
            "optimizer_state_dict": final_optimizer.state_dict(),
            "config": winner_config.__dict__,
            "step": step,
            "avg_loss": sum(losses[-100:]) / len(losses[-100:]) if losses else float("inf"),
            "variant_name": winner_name,
        }, final_ckpt)
        console.print(f"[bold green]Final checkpoint: {final_ckpt}[/bold green]")

    # Summary
    elapsed_total = time.time() - start_time
    console.print(f"\n[bold magenta]{'='*70}[/bold magenta]")
    console.print(f"[bold magenta]  TRAINING COMPLETE[/bold magenta]")
    console.print(f"[bold magenta]  Winner: {winner_name}[/bold magenta]")
    console.print(f"[bold magenta]  Total steps: {step}[/bold magenta]")
    console.print(f"[bold magenta]  Final loss: {sum(losses[-100:])/len(losses[-100:]) if losses else 'N/A'}[/bold magenta]")
    console.print(f"[bold magenta]  Time: {elapsed_total/3600:.1f} hours[/bold magenta]")
    console.print(f"[bold magenta]{'='*70}[/bold magenta]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hamner Architecture Tournament")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Where to save checkpoints")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length for tournament rounds")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--round1-steps", type=int, default=500, help="Steps per variant in round 1")
    parser.add_argument("--round2-steps", type=int, default=1000, help="Additional steps in round 2")
    parser.add_argument("--round3-steps", type=int, default=2000, help="Additional steps in round 3")
    parser.add_argument("--final-steps", type=int, default=0, help="Steps for winner (0=unlimited)")
    args = parser.parse_args()

    run_tournament(
        checkpoint_dir=args.checkpoint_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        round1_steps=args.round1_steps,
        round2_steps=args.round2_steps,
        round3_steps=args.round3_steps,
        final_steps=args.final_steps,
    )
