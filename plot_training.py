#!/usr/bin/env python3
"""
Plot training metrics for Hamner.

Usage:
    python plot_training.py                    # show all plots
    python plot_training.py --save             # save to logs/plots/
    python plot_training.py --tournament       # include tournament data
    python plot_training.py --live             # auto-refresh every 60s
"""

import os
import sys
import csv
import json
import argparse
import math
from pathlib import Path

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("matplotlib not installed. Install with: pip install matplotlib")
    sys.exit(1)


METRICS_FILE = "logs/metrics.csv"
TOURNAMENT_FILE = "logs/tournament_metrics.csv"
SAMPLES_FILE = "logs/samples.jsonl"
PLOT_DIR = "logs/plots"

# Notable training events to annotate on plots
# (step, short_label, description)
TRAINING_EVENTS = [
    (49000, "batch 16\u219248, grad ckpt off",
     "Batch size 16\u219248, disabled gradient checkpointing.\n"
     "Doubled VRAM usage (65%\u219292%), ~2x throughput.\n"
     "Loss spike is expected and recovers quickly."),
]


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


def plot_training_loss(metrics, save_dir=None):
    """Plot loss over training steps and tokens."""
    if not metrics:
        print("No training metrics to plot.")
        return

    steps = [m["step"] for m in metrics]
    losses = [m["loss"] for m in metrics]
    tokens_b = [m["tokens_billions"] for m in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss vs steps
    ax1.plot(steps, losses, color="#2196F3", linewidth=0.8, alpha=0.5, label="Raw")
    # Smoothed (rolling avg of 20 points)
    if len(losses) > 20:
        window = 20
        smoothed = [sum(losses[max(0,i-window):i+1]) / len(losses[max(0,i-window):i+1])
                     for i in range(len(losses))]
        ax1.plot(steps, smoothed, color="#F44336", linewidth=2, label="Smoothed")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss vs Steps")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    annotate_events(ax1, metrics, x_key="step")

    # Loss vs tokens
    ax2.plot(tokens_b, losses, color="#2196F3", linewidth=0.8, alpha=0.5, label="Raw")
    if len(losses) > 20:
        ax2.plot(tokens_b, smoothed, color="#F44336", linewidth=2, label="Smoothed")
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


def plot_perplexity(metrics, save_dir=None):
    """Plot perplexity over training."""
    if not metrics:
        return

    steps = [m["step"] for m in metrics]
    ppls = [m["perplexity"] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(steps, ppls, color="#4CAF50", linewidth=0.8, alpha=0.5, label="Raw")
    if len(ppls) > 20:
        window = 20
        smoothed = [sum(ppls[max(0,i-window):i+1]) / len(ppls[max(0,i-window):i+1])
                     for i in range(len(ppls))]
        ax.plot(steps, smoothed, color="#FF9800", linewidth=2, label="Smoothed")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity Over Training")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    annotate_events(ax, metrics, x_key="step")

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


def plot_all_dashboard(metrics, tournament_data, samples_data, save_dir=None):
    """Single dashboard image with key metrics."""
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

    # 1. Loss curve (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(tokens_b, losses, color="#2196F3", linewidth=0.5, alpha=0.3)
    if len(losses) > 20:
        ax1.plot(tokens_b, smooth(losses), color="#F44336", linewidth=2)
    ax1.set_xlabel("Tokens (Billions)")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    annotate_events(ax1, metrics, x_key="tokens_billions")

    # 2. Perplexity (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(tokens_b, ppls, color="#4CAF50", linewidth=0.5, alpha=0.3)
    if len(ppls) > 20:
        ax2.plot(tokens_b, smooth(ppls), color="#FF9800", linewidth=2)
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
    stats_text = (
        f"Model: Hamner v02_dense_medium (164M params)\n"
        f"─────────────────────────────────\n"
        f"Current Step:     {current['step']:,}\n"
        f"Current Loss:     {current['loss']:.4f}\n"
        f"Current PPL:      {current['perplexity']:.1f}\n"
        f"Tokens Processed: {current['tokens_billions']:.2f}B\n"
        f"Training Time:    {current['elapsed_hours']:.1f}h\n"
        f"Avg Throughput:   {avg_tps:.0f} tok/s\n"
        f"─────────────────────────────────\n"
        f"Target: 300K steps / ~4.9B tokens\n"
        f"Progress: {current['step']/300000*100:.1f}%\n"
    )

    if current['tokens_per_sec'] > 0:
        remaining_tokens = 4.9e9 - current['tokens_billions'] * 1e9
        remaining_hours = remaining_tokens / current['tokens_per_sec'] / 3600
        stats_text += f"ETA: ~{remaining_hours:.0f}h ({remaining_hours/24:.1f} days)\n"

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
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


def main():
    parser = argparse.ArgumentParser(description="Plot Hamner training metrics")
    parser.add_argument("--save", action="store_true", help="Save plots to logs/plots/")
    parser.add_argument("--tournament", action="store_true", help="Include tournament plots")
    parser.add_argument("--live", action="store_true", help="Auto-refresh every 60s")
    parser.add_argument("--dashboard", action="store_true", help="Single dashboard image only")
    args = parser.parse_args()

    save_dir = PLOT_DIR if args.save or args.dashboard else None

    while True:
        metrics = load_metrics(METRICS_FILE)
        tournament_data = load_tournament(TOURNAMENT_FILE)
        samples_data = load_samples(SAMPLES_FILE)

        print(f"Loaded {len(metrics)} training metrics, {len(tournament_data)} tournament entries, {len(samples_data)} sample sets")

        if args.dashboard:
            plot_all_dashboard(metrics, tournament_data, samples_data, save_dir)
            break

        plot_training_loss(metrics, save_dir)
        plot_perplexity(metrics, save_dir)
        plot_learning_rate(metrics, save_dir)
        plot_throughput(metrics, save_dir)

        if args.tournament or tournament_data:
            plot_tournament(tournament_data, save_dir)
            plot_tournament_final(tournament_data, save_dir)

        if samples_data:
            plot_sample_evolution(samples_data, save_dir)

        plot_all_dashboard(metrics, tournament_data, samples_data, save_dir or PLOT_DIR)

        if not args.live:
            break

        print(f"\nRefreshing in 60s... (Ctrl+C to stop)")
        import time
        time.sleep(60)


if __name__ == "__main__":
    main()
