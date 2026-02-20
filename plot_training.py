#!/usr/bin/env python3
"""
Plot training metrics for Hamner.

Supports V3 (50M staged), V2 (164M pretrain), and legacy tournament data.
Auto-detects which training run has data, or use --v3 / --v2 to force.

Usage:
    python plot_training.py                    # auto-detect, show all plots
    python plot_training.py --save             # save to logs/plots/
    python plot_training.py --dashboard        # single dashboard image
    python plot_training.py --live             # auto-refresh every 60s
    python plot_training.py --v3               # force V3 metrics
    python plot_training.py --v2               # force V2 metrics
    python plot_training.py --tournament       # include legacy tournament data
"""

import os
import sys
import csv
import json
import argparse
import math
from pathlib import Path
import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("matplotlib not installed. Install with: pip install matplotlib")
    sys.exit(1)


V3_METRICS_FILE = "logs/v3_metrics.csv"
V3_SAMPLES_FILE = "logs/v3_samples.jsonl"

V2_METRICS_FILE = "logs/pretrain_v2_metrics.csv"
V2_SAMPLES_FILE = "logs/pretrain_v2_samples.jsonl"

PLOT_DIR = "logs/plots"

# Legacy files (v1 pretraining + curriculum + tournament)
LEGACY_METRICS_FILE = "logs/metrics.csv"
LEGACY_SAMPLES_FILE = "logs/samples.jsonl"
TOURNAMENT_FILE = "logs/tournament_metrics.csv"
CURRICULUM_METRICS_FILE = "logs/curriculum_metrics.csv"
CURRICULUM_SAMPLES_FILE = "logs/curriculum_samples.jsonl"

# V3 stage definitions — colors and display order
V3_STAGES = [
    ("structure", "#2196F3"),  # blue
    ("knowledge", "#4CAF50"),  # green
    ("dialogue",  "#FF9800"),  # orange
    ("voice",     "#9C27B0"),  # purple
]
V3_STAGE_COLORS = dict(V3_STAGES)
V3_STAGE_ORDER = [s[0] for s in V3_STAGES]
# Also match "xxx_done" stage transition markers
for _name, _color in list(V3_STAGES):
    V3_STAGE_COLORS[f"{_name}_done"] = _color

# V2 target parameters (for V2 dashboard/projection)
V2_TARGET_STEPS = 400_000
V2_TARGET_BATCH_SIZE = 24
V2_TARGET_SEQ_LEN = 1024
V2_TARGET_TOKENS_B = V2_TARGET_STEPS * V2_TARGET_BATCH_SIZE * V2_TARGET_SEQ_LEN / 1e9  # ~9.8B

# Backwards compat aliases used by existing functions
TARGET_TOKENS_B = V2_TARGET_TOKENS_B
TRAINING_EVENTS = []


def annotate_events(ax, metrics, x_key="step"):
    """Add vertical lines and labels for notable training events."""
    if not metrics or not TRAINING_EVENTS:
        return
    x_vals = [m[x_key] for m in metrics]
    x_min, x_max = min(x_vals), max(x_vals)
    for step, label, _ in TRAINING_EVENTS:
        # Map step to the appropriate x-axis value
        if x_key == "step":
            x = step
        else:
            # Find the closest metric entry to this step
            closest = min(metrics, key=lambda m: abs(m["step"] - step))
            x = closest[x_key]
        if x < x_min or x > x_max:
            continue
        ax.axvline(x=x, color="#FF9800", linestyle="--", alpha=0.7, linewidth=1.2)
        ax.text(x, ax.get_ylim()[1], f" {label}", rotation=45, fontsize=7,
                color="#E65100", ha="left", va="top")


def load_metrics(path):
    """Load metrics CSV into list of dicts."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["step", "tokens_total"]:
                if key in row:
                    row[key] = int(float(row[key]))
            for key in ["loss", "perplexity", "learning_rate", "tokens_per_sec",
                         "tokens_billions", "elapsed_hours"]:
                if key in row and row[key]:
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def load_tournament(path):
    """Load tournament metrics CSV."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["step"] = int(row["step"])
            row["round"] = int(row["round"])
            row["loss"] = float(row["loss"])
            if row.get("tokens_per_sec"):
                row["tokens_per_sec"] = float(row["tokens_per_sec"])
            rows.append(row)
    return rows


def load_samples(path):
    """Load sample generations JSONL."""
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def clean_metrics(metrics, jump_pct=0.4):
    """Deduplicate by step, find the last restart spike, start from there.

    Returns metrics from the last spike onward with spike artifacts removed.
    This gives a clean view of the current training arc.
    """
    if not metrics:
        return []

    # Deduplicate: keep lowest loss per step
    by_step = {}
    for m in metrics:
        step = m["step"]
        if step not in by_step or m["loss"] < by_step[step]["loss"]:
            by_step[step] = m
    deduped = sorted(by_step.values(), key=lambda m: m["step"])

    if len(deduped) < 2:
        return deduped

    # Find the last spike: where loss jumps up > jump_pct from previous
    last_spike_idx = 0
    for i in range(1, len(deduped)):
        if deduped[i]["loss"] > deduped[i - 1]["loss"] * (1 + jump_pct):
            last_spike_idx = i

    # Start from the point before the last spike (the clean anchor)
    start = max(0, last_spike_idx - 1)
    return deduped[start:]


def fit_projection(clean, target_tokens_b=None):
    """Fit power law to the clean tail of the loss curve and project forward.

    Fits log(loss) = slope * log(tokens_B) + intercept on the last 50%
    of the data, then extrapolates to target_tokens_b.

    Returns dict with projection arrays and stats, or None on failure.
    """
    if target_tokens_b is None:
        target_tokens_b = TARGET_TOKENS_B

    if len(clean) < 40:
        return None

    # Fit on last 50% of clean data for stable extrapolation
    n = len(clean)
    fit_data = clean[n // 2:]

    x = np.array([m["tokens_billions"] for m in fit_data])
    y = np.array([m["loss"] for m in fit_data])

    mask = (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    if len(x) < 20:
        return None

    try:
        slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
        x_last = x[-1]
        x_proj = np.linspace(x_last, target_tokens_b, 300)
        y_proj = np.exp(intercept + slope * np.log(x_proj))
        final_loss = np.exp(intercept + slope * np.log(target_tokens_b))
        final_ppl = math.exp(min(final_loss, 20))
        return {
            "tokens": x_proj,
            "loss": y_proj,
            "ppl": np.array([math.exp(min(l, 20)) for l in y_proj]),
            "final_loss": final_loss,
            "final_ppl": final_ppl,
            "slope": slope,
        }
    except Exception:
        return None


def plot_training_loss(metrics, save_dir=None, projection=None):
    """Plot loss over training steps and tokens, with optional projection."""
    if not metrics:
        print("No training metrics to plot.")
        return

    steps = [m["step"] for m in metrics]
    losses = [m["loss"] for m in metrics]
    tokens_b = [m["tokens_billions"] for m in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Smoothing helper
    def smooth(data, window=20):
        return [sum(data[max(0,i-window):i+1]) / len(data[max(0,i-window):i+1])
                for i in range(len(data))]

    # Loss vs steps
    ax1.plot(steps, losses, color="#2196F3", linewidth=0.8, alpha=0.4, label="Raw")
    if len(losses) > 20:
        ax1.plot(steps, smooth(losses), color="#F44336", linewidth=2, label="Smoothed")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss vs Steps")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    annotate_events(ax1, metrics, x_key="step")

    # Loss vs tokens + projection
    ax2.plot(tokens_b, losses, color="#2196F3", linewidth=0.8, alpha=0.4, label="Raw")
    if len(losses) > 20:
        ax2.plot(tokens_b, smooth(losses), color="#F44336", linewidth=2, label="Smoothed")
    if projection:
        ax2.plot(projection["tokens"], projection["loss"], color="#F44336",
                 linewidth=2, linestyle="--", alpha=0.7, label=f"Projected (final {projection['final_loss']:.2f})")
        ax2.axhline(y=projection["final_loss"], color="gray", linestyle=":", alpha=0.3)
    ax2.set_xlabel("Tokens (Billions)")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Loss vs Tokens Processed")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    annotate_events(ax2, metrics, x_key="tokens_billions")

    plt.suptitle("Hamner Training Progress", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "training_loss.png"), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_dir}/training_loss.png")
    else:
        plt.show()
    plt.close()


def plot_perplexity(metrics, save_dir=None, projection=None):
    """Plot perplexity over training with optional projection."""
    if not metrics:
        return

    tokens_b = [m["tokens_billions"] for m in metrics]
    ppls = [m["perplexity"] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(tokens_b, ppls, color="#4CAF50", linewidth=0.8, alpha=0.4, label="Raw")
    if len(ppls) > 20:
        window = 20
        smoothed = [sum(ppls[max(0,i-window):i+1]) / len(ppls[max(0,i-window):i+1])
                     for i in range(len(ppls))]
        ax.plot(tokens_b, smoothed, color="#FF9800", linewidth=2, label="Smoothed")
    if projection:
        ax.plot(projection["tokens"], projection["ppl"], color="#FF9800",
                linewidth=2, linestyle="--", alpha=0.7, label=f"Projected (final {projection['final_ppl']:.0f})")
        ax.axhline(y=projection["final_ppl"], color="gray", linestyle=":", alpha=0.3)

    ax.set_xlabel("Tokens (Billions)")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity Over Training")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    annotate_events(ax, metrics, x_key="tokens_billions")

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "perplexity.png"), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_dir}/perplexity.png")
    else:
        plt.show()
    plt.close()


def plot_learning_rate(metrics, save_dir=None):
    """Plot learning rate schedule."""
    if not metrics:
        return

    steps = [m["step"] for m in metrics]
    lrs = [m["learning_rate"] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, lrs, color="#9C27B0", linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "learning_rate.png"), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_dir}/learning_rate.png")
    else:
        plt.show()
    plt.close()


def plot_throughput(metrics, save_dir=None):
    """Plot tokens/sec throughput over training."""
    if not metrics:
        return

    steps = [m["step"] for m in metrics]
    tps = [m["tokens_per_sec"] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, tps, color="#00BCD4", linewidth=0.8, alpha=0.5)
    if len(tps) > 20:
        window = 20
        smoothed = [sum(tps[max(0,i-window):i+1]) / len(tps[max(0,i-window):i+1])
                     for i in range(len(tps))]
        ax.plot(steps, smoothed, color="#E91E63", linewidth=2, label="Smoothed")
    avg_tps = sum(tps) / len(tps)
    ax.axhline(y=avg_tps, color="gray", linestyle="--", alpha=0.5, label=f"Avg: {avg_tps:.0f} tok/s")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Training Throughput")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    annotate_events(ax, metrics, x_key="step")

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "throughput.png"), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_dir}/throughput.png")
    else:
        plt.show()
    plt.close()


def plot_tournament(tournament_data, save_dir=None):
    """Plot tournament results across all rounds."""
    if not tournament_data:
        print("No tournament data to plot.")
        return

    # Group by variant
    variants = {}
    for row in tournament_data:
        v = row["variant"]
        if v not in variants:
            variants[v] = {"steps": [], "losses": [], "rounds": []}
        variants[v]["steps"].append(row["step"])
        variants[v]["losses"].append(row["loss"])
        variants[v]["rounds"].append(row["round"])

    # Color map
    colors = plt.cm.tab10(range(len(variants)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (name, data) in enumerate(sorted(variants.items())):
        style = "-" if "medium" in name else "--" if "small" in name else ":"
        lw = 2.5 if name == "v02_dense_medium" else 1.2
        alpha = 1.0 if name == "v02_dense_medium" else 0.7
        ax.plot(data["steps"], data["losses"], style, color=colors[i],
                linewidth=lw, alpha=alpha, label=name, marker="o", markersize=2)

    # Draw round boundaries
    ax.axvline(x=500, color="gray", linestyle=":", alpha=0.4)
    ax.axvline(x=1500, color="gray", linestyle=":", alpha=0.4)
    ax.text(250, ax.get_ylim()[1], "Round 1", ha="center", va="top", fontsize=9, color="gray")
    ax.text(1000, ax.get_ylim()[1], "Round 2", ha="center", va="top", fontsize=9, color="gray")
    ax.text(2750, ax.get_ylim()[1], "Round 3", ha="center", va="top", fontsize=9, color="gray")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Tournament Architecture Search — 10 Variants, 3 Rounds")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "tournament.png"), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_dir}/tournament.png")
    else:
        plt.show()
    plt.close()


def plot_tournament_final(tournament_data, save_dir=None):
    """Bar chart of final losses for each variant."""
    if not tournament_data:
        return

    # Get final loss per variant
    final = {}
    for row in tournament_data:
        v = row["variant"]
        final[v] = row["loss"]  # last entry wins

    # Sort by loss
    sorted_variants = sorted(final.items(), key=lambda x: x[1])
    names = [v[0] for v in sorted_variants]
    losses = [v[1] for v in sorted_variants]

    # Color the winner
    colors = ["#4CAF50" if i == 0 else "#2196F3" for i in range(len(names))]
    # Gray out OOM
    for i, (name, loss) in enumerate(sorted_variants):
        if loss > 50:
            colors[i] = "#BDBDBD"

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(names, losses, color=colors)
    ax.set_xlabel("Final Loss (lower is better)")
    ax.set_title("Tournament Final Rankings")
    ax.invert_yaxis()

    # Add labels
    for bar, loss in zip(bars, losses):
        if loss < 50:
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{loss:.3f}", va="center", fontsize=9)
        else:
            ax.text(0.1, bar.get_y() + bar.get_height()/2,
                    "OOM", va="center", fontsize=9, color="red")

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "tournament_rankings.png"), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_dir}/tournament_rankings.png")
    else:
        plt.show()
    plt.close()


def plot_sample_evolution(samples_data, save_dir=None):
    """Show how sample quality evolves over training."""
    if not samples_data:
        print("No sample data to plot.")
        return

    # Create a text-based visualization
    fig, ax = plt.subplots(figsize=(14, max(6, len(samples_data) * 0.8)))
    ax.axis("off")

    y = 0.98
    dy = 0.9 / max(len(samples_data), 1)

    ax.text(0.5, 1.0, "Sample Generation Evolution", ha="center", va="top",
            fontsize=14, fontweight="bold", transform=ax.transAxes)

    for entry in samples_data:
        step = entry["step"]
        tokens_b = entry.get("tokens_billions", 0)
        samples = entry.get("samples", {})

        # Pick one sample to show
        sample_text = ""
        for prompt, output in samples.items():
            sample_text = output[:120]
            break

        color = "#333333"
        ax.text(0.02, y, f"Step {step:,} ({tokens_b:.2f}B tokens):",
                fontsize=9, fontweight="bold", color="#1565C0",
                transform=ax.transAxes, verticalalignment="top")
        ax.text(0.02, y - 0.02, sample_text,
                fontsize=8, color=color, style="italic",
                transform=ax.transAxes, verticalalignment="top",
                wrap=True)
        y -= dy

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "sample_evolution.png"), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_dir}/sample_evolution.png")
    else:
        plt.show()
    plt.close()


def plot_all_dashboard(metrics, tournament_data, samples_data, save_dir=None, projection=None):
    """Single dashboard image with key metrics and projection."""
    if not metrics:
        print("No metrics for dashboard.")
        return

    fig = plt.figure(figsize=(16, 10))

    steps = [m["step"] for m in metrics]
    losses = [m["loss"] for m in metrics]
    ppls = [m["perplexity"] for m in metrics]
    tokens_b = [m["tokens_billions"] for m in metrics]
    tps_list = [m["tokens_per_sec"] for m in metrics]

    # Smoothing helper
    def smooth(data, window=20):
        return [sum(data[max(0,i-window):i+1]) / len(data[max(0,i-window):i+1])
                for i in range(len(data))]

    # 1. Loss curve (top left) + projection
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(tokens_b, losses, color="#2196F3", linewidth=0.5, alpha=0.3)
    if len(losses) > 20:
        ax1.plot(tokens_b, smooth(losses), color="#F44336", linewidth=2)
    if projection:
        ax1.plot(projection["tokens"], projection["loss"], color="#F44336",
                 linewidth=2, linestyle="--", alpha=0.6)
        ax1.axhline(y=projection["final_loss"], color="gray", linestyle=":", alpha=0.3)
        ax1.text(TARGET_TOKENS_B, projection["final_loss"],
                 f"  {projection['final_loss']:.2f}", fontsize=8, va="bottom", color="#888")
    ax1.set_xlabel("Tokens (Billions)")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    annotate_events(ax1, metrics, x_key="tokens_billions")

    # 2. Perplexity (top right) + projection
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(tokens_b, ppls, color="#4CAF50", linewidth=0.5, alpha=0.3)
    if len(ppls) > 20:
        ax2.plot(tokens_b, smooth(ppls), color="#FF9800", linewidth=2)
    if projection:
        ax2.plot(projection["tokens"], projection["ppl"], color="#FF9800",
                 linewidth=2, linestyle="--", alpha=0.6)
        ax2.axhline(y=projection["final_ppl"], color="gray", linestyle=":", alpha=0.3)
        ax2.text(TARGET_TOKENS_B, projection["final_ppl"],
                 f"  {projection['final_ppl']:.0f}", fontsize=8, va="bottom", color="#888")
    ax2.set_xlabel("Tokens (Billions)")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Perplexity")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    annotate_events(ax2, metrics, x_key="tokens_billions")

    # 3. Throughput (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(steps, tps_list, color="#00BCD4", linewidth=0.5, alpha=0.3)
    if len(tps_list) > 20:
        ax3.plot(steps, smooth(tps_list), color="#E91E63", linewidth=2)
    avg_tps = sum(tps_list) / len(tps_list)
    ax3.axhline(y=avg_tps, color="gray", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Tokens/sec")
    ax3.set_title(f"Throughput (avg: {avg_tps:.0f} tok/s)")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    annotate_events(ax3, metrics, x_key="step")

    # 4. Stats summary (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")

    current = metrics[-1]
    pct = current['tokens_billions'] / TARGET_TOKENS_B * 100
    stats_text = (
        f"Model: Hamner Pretrain v2 (164M params)\n"
        f"─────────────────────────────────\n"
        f"Current Step:     {current['step']:,}\n"
        f"Current Loss:     {current['loss']:.4f}\n"
        f"Current PPL:      {current['perplexity']:.1f}\n"
        f"Tokens Processed: {current['tokens_billions']:.2f}B\n"
        f"Avg Throughput:   {avg_tps:.0f} tok/s\n"
        f"─────────────────────────────────\n"
        f"Target: {TARGET_STEPS/1000:.0f}K steps / {TARGET_TOKENS_B:.1f}B tokens\n"
        f"Progress: {pct:.1f}%\n"
    )

    if current['tokens_per_sec'] > 0:
        remaining_tokens = TARGET_TOKENS_B * 1e9 - current['tokens_billions'] * 1e9
        remaining_hours = remaining_tokens / current['tokens_per_sec'] / 3600
        stats_text += f"ETA: ~{remaining_hours:.0f}h ({remaining_hours/24:.1f} days)\n"

    if projection:
        stats_text += (
            f"─────────────────────────────────\n"
            f"Projected final loss: {projection['final_loss']:.2f}\n"
            f"Projected final PPL:  {projection['final_ppl']:.0f}\n"
            f"Scaling slope:        {projection['slope']:.3f}\n"
        )

    ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#BDBDBD"))

    plt.suptitle("Hamner Training Dashboard", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "dashboard.png"), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_dir}/dashboard.png")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# V3 dashboard — stage-colored loss with plateau transitions
# ---------------------------------------------------------------------------

def _v3_stage_name(phase_str):
    """Normalize V3 phase column to canonical stage name."""
    if not phase_str:
        return "structure"
    base = phase_str.replace("_done", "")
    if base in V3_STAGE_COLORS:
        return base
    return phase_str


def _v3_stage_spans(metrics):
    """Find step and tokens_billions ranges for each V3 stage."""
    spans = {}
    for m in metrics:
        stage = _v3_stage_name(m.get("phase", ""))
        tb = m["tokens_billions"]
        step = m["step"]
        if stage not in spans:
            spans[stage] = {"tb_min": tb, "tb_max": tb, "s_min": step, "s_max": step}
        else:
            spans[stage]["tb_min"] = min(spans[stage]["tb_min"], tb)
            spans[stage]["tb_max"] = max(spans[stage]["tb_max"], tb)
            spans[stage]["s_min"] = min(spans[stage]["s_min"], step)
            spans[stage]["s_max"] = max(spans[stage]["s_max"], step)
    return spans


def _v3_shade(ax, spans, x_key="tokens_billions"):
    """Add colored background shading for V3 stages."""
    for stage in V3_STAGE_ORDER:
        if stage not in spans:
            continue
        color = V3_STAGE_COLORS.get(stage, "#999")
        if x_key == "tokens_billions":
            s, e = spans[stage]["tb_min"], spans[stage]["tb_max"]
        else:
            s, e = spans[stage]["s_min"], spans[stage]["s_max"]
        if s == e:
            continue
        ax.axvspan(s, e, alpha=0.10, color=color)
        mid = (s + e) / 2
        ymin, ymax = ax.get_ylim()
        ax.text(mid, ymax - (ymax - ymin) * 0.03, stage,
                ha="center", va="top", fontsize=9,
                color=color, fontweight="bold", alpha=0.9)


def plot_v3_dashboard(metrics, samples_data, save_dir=None):
    """V3-specific dashboard: stage-colored loss, throughput, LR, stats."""
    if not metrics:
        print("No V3 metrics for dashboard.")
        return

    fig = plt.figure(figsize=(16, 10))

    steps = [m["step"] for m in metrics]
    losses = [m["loss"] for m in metrics]
    ppls = [m["perplexity"] for m in metrics]
    tokens_b = [m["tokens_billions"] for m in metrics]
    tps_list = [m["tokens_per_sec"] for m in metrics]
    stages = [_v3_stage_name(m.get("phase", "")) for m in metrics]

    spans = _v3_stage_spans(metrics)

    # ── 1. Loss vs Tokens — stage-colored ──
    ax1 = fig.add_subplot(2, 2, 1)
    for stage in V3_STAGE_ORDER:
        idx = [i for i, s in enumerate(stages) if s == stage]
        if not idx:
            continue
        tb = [tokens_b[i] for i in idx]
        lo = [losses[i] for i in idx]
        color = V3_STAGE_COLORS.get(stage, "#999")
        ax1.plot(tb, lo, color=color, linewidth=0.5, alpha=0.25)
        if len(lo) > 15:
            ax1.plot(tb, smooth(lo, min(30, len(lo) // 4)),
                     color=color, linewidth=2.5, label=stage)
        else:
            ax1.plot(tb, lo, color=color, linewidth=2, label=stage)
    # Mark stage transitions
    for stage in V3_STAGE_ORDER:
        if stage in spans:
            tb_start = spans[stage]["tb_min"]
            if tb_start > tokens_b[0]:
                ax1.axvline(x=tb_start, color=V3_STAGE_COLORS.get(stage, "#999"),
                           linestyle="--", alpha=0.5, linewidth=1)
    ax1.set_xlabel("Tokens (Billions)")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss by Stage")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ── 2. Perplexity vs Tokens ──
    ax2 = fig.add_subplot(2, 2, 2)
    for stage in V3_STAGE_ORDER:
        idx = [i for i, s in enumerate(stages) if s == stage]
        if not idx:
            continue
        tb = [tokens_b[i] for i in idx]
        pp = [ppls[i] for i in idx]
        color = V3_STAGE_COLORS.get(stage, "#999")
        ax2.plot(tb, pp, color=color, linewidth=0.5, alpha=0.25)
        if len(pp) > 15:
            ax2.plot(tb, smooth(pp, min(30, len(pp) // 4)),
                     color=color, linewidth=2.5, label=stage)
        else:
            ax2.plot(tb, pp, color=color, linewidth=2, label=stage)
    ax2.set_xlabel("Tokens (Billions)")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Perplexity by Stage")
    ax2.set_yscale("log")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # ── 3. Loss vs Steps — stage-shaded ──
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(steps, losses, color="#2196F3", linewidth=0.5, alpha=0.3)
    if len(losses) > 20:
        ax3.plot(steps, smooth(losses), color="#F44336", linewidth=2)
    # Shade after plotting so ylim is set
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Loss")
    ax3.set_title("Loss vs Steps (stage-shaded)")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    _v3_shade(ax3, spans, x_key="step")

    # ── 4. Stats + latest samples ──
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")

    cur = metrics[-1]
    cur_stage = _v3_stage_name(cur.get("phase", ""))
    avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0

    # Per-stage summary
    stage_summary = ""
    for stage in V3_STAGE_ORDER:
        if stage in spans:
            s_losses = [losses[i] for i, s in enumerate(stages) if s == stage]
            min_loss = min(s_losses) if s_losses else 0
            n_steps = spans[stage]["s_max"] - spans[stage]["s_min"]
            stage_summary += f"  {stage:<10s} {n_steps:>6,} steps  loss {min_loss:.4f}\n"

    stats_text = (
        f"Hamner V3 — 50M Staged Learning\n"
        f"{'='*40}\n"
        f"Current Stage:  {cur_stage}\n"
        f"Global Step:    {cur['step']:,}\n"
        f"Current Loss:   {cur['loss']:.4f}\n"
        f"Current PPL:    {cur['perplexity']:.1f}\n"
        f"Tokens:         {cur['tokens_billions']:.3f}B\n"
        f"Avg Throughput: {avg_tps:.0f} tok/s\n"
        f"{'='*40}\n"
        f"Stage Progress:\n"
        f"{stage_summary}"
    )

    # Latest sample
    if samples_data:
        latest = samples_data[-1]
        stats_text += f"{'='*40}\n"
        stats_text += f"Latest sample (step {latest.get('step', '?')}):\n"
        for prompt, output in latest.get("samples", {}).items():
            stats_text += f"  \"{output[:150]}...\"\n"
            break

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#BDBDBD"))

    plt.suptitle("Hamner V3 Training Dashboard", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "v3_dashboard.png"), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_dir}/v3_dashboard.png")
    else:
        plt.show()
    plt.close()


def plot_v3_stages(metrics, save_dir=None):
    """Detailed per-stage loss curves for V3 — one subplot per stage."""
    if not metrics:
        return

    stages_present = []
    for stage in V3_STAGE_ORDER:
        if any(_v3_stage_name(m.get("phase", "")) == stage for m in metrics):
            stages_present.append(stage)

    if not stages_present:
        return

    n = len(stages_present)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for col, stage in enumerate(stages_present):
        ax = axes[0][col]
        idx = [i for i, m in enumerate(metrics) if _v3_stage_name(m.get("phase", "")) == stage]
        if not idx:
            continue
        # Use step-in-stage (relative) for x-axis
        step_offset = metrics[idx[0]]["step"]
        local_steps = [metrics[i]["step"] - step_offset for i in idx]
        lo = [metrics[i]["loss"] for i in idx]
        color = V3_STAGE_COLORS.get(stage, "#999")

        ax.plot(local_steps, lo, color=color, linewidth=0.5, alpha=0.3)
        if len(lo) > 15:
            ax.plot(local_steps, smooth(lo, min(20, len(lo) // 4)),
                    color=color, linewidth=2.5)
        ax.set_xlabel("Steps in Stage")
        ax.set_ylabel("Loss")
        ax.set_title(f"{stage.title()} (loss {min(lo):.4f})")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    plt.suptitle("V3 Per-Stage Loss", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "v3_stages.png"), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_dir}/v3_stages.png")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Full journey plot: pretraining + curriculum on unified timeline
# ---------------------------------------------------------------------------

# All training stages in order, with colors
ALL_PHASES = [
    ("pretrain",  "#78909C"),   # blue-grey — original pretraining
    ("structure", "#2196F3"),   # blue
    ("stories",   "#4CAF50"),   # green
    ("voice",     "#FF9800"),   # orange
    ("polish",    "#9C27B0"),   # purple
]
ALL_PHASE_COLORS = dict(ALL_PHASES)
ALL_PHASE_ORDER = [p[0] for p in ALL_PHASES]


def _classify_phase(row_phase):
    """Map raw CSV phase labels to unified phase names."""
    if row_phase in ("structure", "stories", "voice", "polish"):
        return row_phase
    return "pretrain"


def _merge_metrics(pretrain_metrics, curriculum_metrics):
    """Merge pretraining and curriculum metrics into one sorted list."""
    combined = []
    for m in pretrain_metrics:
        m = dict(m)
        m["unified_phase"] = _classify_phase(m.get("phase", ""))
        combined.append(m)
    for m in curriculum_metrics:
        m = dict(m)
        m["unified_phase"] = _classify_phase(m.get("phase", ""))
        combined.append(m)
    combined.sort(key=lambda m: m["tokens_billions"])
    return combined


def _phase_token_spans(metrics):
    """Find tokens_billions ranges for each phase."""
    spans = {}
    for m in metrics:
        phase = m["unified_phase"]
        tb = m["tokens_billions"]
        if phase not in spans:
            spans[phase] = [tb, tb]
        else:
            spans[phase][0] = min(spans[phase][0], tb)
            spans[phase][1] = max(spans[phase][1], tb)
    return spans


def _shade_token_phases(ax, spans):
    """Add colored background shading by tokens_billions."""
    for phase in ALL_PHASE_ORDER:
        if phase in spans:
            s, e = spans[phase]
            ax.axvspan(s, e, alpha=0.10, color=ALL_PHASE_COLORS[phase])
            mid = (s + e) / 2
            ymin, ymax = ax.get_ylim()
            ax.text(mid, ymax - (ymax - ymin) * 0.03, phase,
                    ha="center", va="top", fontsize=9,
                    color=ALL_PHASE_COLORS[phase], fontweight="bold", alpha=0.9)


def smooth(data, window=20):
    """Simple moving average."""
    return [sum(data[max(0, i-window):i+1]) / len(data[max(0, i-window):i+1])
            for i in range(len(data))]


def plot_full_journey(pretrain_metrics, curriculum_metrics, curriculum_samples, save_dir=None):
    """Plot the complete training journey: pretraining through curriculum phases."""
    combined = _merge_metrics(pretrain_metrics, curriculum_metrics)
    if not combined:
        print("No metrics for full journey plot.")
        return

    spans = _phase_token_spans(combined)

    fig = plt.figure(figsize=(18, 14))

    # ── 1. Loss vs Tokens (full journey) ──────────────────────────
    ax1 = fig.add_subplot(3, 2, 1)
    for phase in ALL_PHASE_ORDER:
        idx = [i for i, m in enumerate(combined) if m["unified_phase"] == phase]
        if not idx:
            continue
        tb = [combined[i]["tokens_billions"] for i in idx]
        lo = [combined[i]["loss"] for i in idx]
        ax1.plot(tb, lo, color=ALL_PHASE_COLORS[phase], linewidth=0.5, alpha=0.25)
        if len(lo) > 15:
            ax1.plot(tb, smooth(lo, min(30, len(lo)//4)),
                     color=ALL_PHASE_COLORS[phase], linewidth=2.5, label=phase)
        else:
            ax1.plot(tb, lo, color=ALL_PHASE_COLORS[phase], linewidth=2, label=phase)
    _shade_token_phases(ax1, spans)
    ax1.set_xlabel("Tokens (Billions)")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss — Full Journey")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ── 2. Perplexity vs Tokens ───────────────────────────────────
    ax2 = fig.add_subplot(3, 2, 2)
    for phase in ALL_PHASE_ORDER:
        idx = [i for i, m in enumerate(combined) if m["unified_phase"] == phase]
        if not idx:
            continue
        tb = [combined[i]["tokens_billions"] for i in idx]
        pp = [combined[i]["perplexity"] for i in idx]
        ax2.plot(tb, pp, color=ALL_PHASE_COLORS[phase], linewidth=0.5, alpha=0.25)
        if len(pp) > 15:
            ax2.plot(tb, smooth(pp, min(30, len(pp)//4)),
                     color=ALL_PHASE_COLORS[phase], linewidth=2.5, label=phase)
        else:
            ax2.plot(tb, pp, color=ALL_PHASE_COLORS[phase], linewidth=2, label=phase)
    _shade_token_phases(ax2, spans)
    ax2.set_xlabel("Tokens (Billions)")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Perplexity — Full Journey")
    ax2.set_yscale("log")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # ── 3. Curriculum loss zoomed (step-based) ────────────────────
    ax3 = fig.add_subplot(3, 2, 3)
    if curriculum_metrics:
        cur_phases = [m.get("phase", "") for m in curriculum_metrics]
        for phase in ALL_PHASE_ORDER[1:]:  # skip pretrain
            idx = [i for i, p in enumerate(cur_phases) if p == phase]
            if not idx:
                continue
            s = [curriculum_metrics[i]["step"] for i in idx]
            lo = [curriculum_metrics[i]["loss"] for i in idx]
            ax3.plot(s, lo, color=ALL_PHASE_COLORS[phase], linewidth=0.5, alpha=0.25)
            if len(lo) > 10:
                ax3.plot(s, smooth(lo, min(20, len(lo)//3)),
                         color=ALL_PHASE_COLORS[phase], linewidth=2.5, label=phase)
            else:
                ax3.plot(s, lo, color=ALL_PHASE_COLORS[phase], linewidth=2, label=phase)
        ax3.legend(fontsize=9)
    ax3.set_xlabel("Curriculum Step")
    ax3.set_ylabel("Loss")
    ax3.set_title("Curriculum Phases — Zoomed")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # ── 4. Learning rate (both runs) ──────────────────────────────
    ax4 = fig.add_subplot(3, 2, 4)
    for phase in ALL_PHASE_ORDER:
        idx = [i for i, m in enumerate(combined) if m["unified_phase"] == phase]
        if not idx:
            continue
        tb = [combined[i]["tokens_billions"] for i in idx]
        lr = [combined[i]["learning_rate"] for i in idx]
        ax4.plot(tb, lr, color=ALL_PHASE_COLORS[phase], linewidth=1.5, label=phase)
    _shade_token_phases(ax4, spans)
    ax4.set_xlabel("Tokens (Billions)")
    ax4.set_ylabel("Learning Rate")
    ax4.set_title("Learning Rate Schedule")
    ax4.legend(fontsize=9, loc="upper right")
    ax4.grid(True, alpha=0.3)

    # ── 5. Throughput ─────────────────────────────────────────────
    ax5 = fig.add_subplot(3, 2, 5)
    for phase in ALL_PHASE_ORDER:
        idx = [i for i, m in enumerate(combined) if m["unified_phase"] == phase]
        if not idx:
            continue
        tb = [combined[i]["tokens_billions"] for i in idx]
        tp = [combined[i]["tokens_per_sec"] for i in idx]
        ax5.plot(tb, tp, color=ALL_PHASE_COLORS[phase], linewidth=0.5, alpha=0.25)
        if len(tp) > 15:
            ax5.plot(tb, smooth(tp, min(30, len(tp)//4)),
                     color=ALL_PHASE_COLORS[phase], linewidth=2, label=phase)
        else:
            ax5.plot(tb, tp, color=ALL_PHASE_COLORS[phase], linewidth=1.5, label=phase)
    _shade_token_phases(ax5, spans)
    ax5.set_xlabel("Tokens (Billions)")
    ax5.set_ylabel("Tokens/sec")
    ax5.set_title("Throughput")
    ax5.legend(fontsize=9, loc="lower right")
    ax5.grid(True, alpha=0.3)

    # ── 6. Stats + latest sample ──────────────────────────────────
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis("off")

    cur = combined[-1]
    pretrain_end = [m for m in combined if m["unified_phase"] == "pretrain"]
    pretrain_final = pretrain_end[-1] if pretrain_end else None

    stats = f"Hamner Training Journey\n{'='*40}\n"
    if pretrain_final:
        stats += (
            f"Pretrain:   {pretrain_final['tokens_billions']:.2f}B tok, "
            f"loss {pretrain_final['loss']:.3f}\n"
        )
    stats += (
        f"Current:    {cur['tokens_billions']:.2f}B tok, "
        f"loss {cur['loss']:.3f}\n"
        f"Phase:      {cur.get('unified_phase', '?')}\n"
        f"PPL:        {cur['perplexity']:.1f}\n"
        f"LR:         {cur['learning_rate']:.2e}\n"
        f"Throughput: {cur['tokens_per_sec']:.0f} tok/s\n"
        f"{'='*40}\n"
    )

    # Latest sample
    if curriculum_samples:
        latest = curriculum_samples[-1]
        stats += f"\nPhase: {latest.get('phase', '?')}, step {latest.get('step', '?')}\n"
        for prompt, output in latest.get("samples", {}).items():
            stats += f"\n\"{output[:200]}...\"\n"
            break

    ax6.text(0.05, 0.95, stats, transform=ax6.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#BDBDBD"))

    plt.suptitle("Hamner — Full Training Journey", fontsize=18, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "full_journey.png"), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_dir}/full_journey.png")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Hamner training metrics")
    parser.add_argument("--save", action="store_true", help="Save plots to logs/plots/")
    parser.add_argument("--tournament", action="store_true", help="Include legacy tournament plots")
    parser.add_argument("--live", action="store_true", help="Auto-refresh every 60s")
    parser.add_argument("--dashboard", action="store_true", help="Single dashboard image only")
    parser.add_argument("--v3", action="store_true", help="Force V3 metrics")
    parser.add_argument("--v2", action="store_true", help="Force V2 metrics")
    args = parser.parse_args()

    save_dir = PLOT_DIR if args.save or args.dashboard else None

    # Auto-detect which run to plot
    if args.v3:
        mode = "v3"
    elif args.v2:
        mode = "v2"
    elif os.path.exists(V3_METRICS_FILE) and os.path.getsize(V3_METRICS_FILE) > 0:
        mode = "v3"
    else:
        mode = "v2"

    while True:
        if mode == "v3":
            # ── V3 plotting ──
            raw_metrics = load_metrics(V3_METRICS_FILE)
            samples_data = load_samples(V3_SAMPLES_FILE)
            metrics = clean_metrics(raw_metrics)

            print(f"V3: {len(raw_metrics)} raw, cleaned to {len(metrics)}")
            if metrics:
                cur = metrics[-1]
                stage = _v3_stage_name(cur.get("phase", ""))
                print(f"  Stage: {stage} | Step: {cur['step']:,} | "
                      f"Loss: {cur['loss']:.4f} | Tokens: {cur['tokens_billions']:.3f}B")

            if args.dashboard:
                plot_v3_dashboard(metrics, samples_data, save_dir)
                break

            # Full V3 plot suite
            plot_v3_dashboard(metrics, samples_data, save_dir or PLOT_DIR)
            plot_v3_stages(metrics, save_dir)
            plot_training_loss(metrics, save_dir)
            plot_perplexity(metrics, save_dir)
            plot_learning_rate(metrics, save_dir)
            plot_throughput(metrics, save_dir)

            if samples_data:
                plot_sample_evolution(samples_data, save_dir)

        else:
            # ── V2 plotting (original behavior) ──
            raw_metrics = load_metrics(V2_METRICS_FILE)
            samples_data = load_samples(V2_SAMPLES_FILE)
            tournament_data = load_tournament(TOURNAMENT_FILE) if args.tournament else []

            metrics = clean_metrics(raw_metrics)
            projection = fit_projection(metrics)

            print(f"Pretrain v2: {len(raw_metrics)} raw, cleaned to {len(metrics)} (from last spike)")
            if projection:
                print(f"Projection: final loss {projection['final_loss']:.2f}, "
                      f"final PPL {projection['final_ppl']:.0f} at {V2_TARGET_TOKENS_B:.1f}B tokens")

            if args.dashboard:
                plot_all_dashboard(metrics, tournament_data, samples_data, save_dir, projection)
                break

            plot_training_loss(metrics, save_dir, projection)
            plot_perplexity(metrics, save_dir, projection)
            plot_learning_rate(metrics, save_dir)
            plot_throughput(metrics, save_dir)

            if args.tournament and tournament_data:
                plot_tournament(tournament_data, save_dir)
                plot_tournament_final(tournament_data, save_dir)

            if samples_data:
                plot_sample_evolution(samples_data, save_dir)

            plot_all_dashboard(metrics, tournament_data, samples_data, save_dir or PLOT_DIR, projection)

        if not args.live:
            break

        print(f"\nRefreshing in 60s... (Ctrl+C to stop)")
        import time
        time.sleep(60)


if __name__ == "__main__":
    main()
