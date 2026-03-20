#!/usr/bin/env python3
"""Generate figures from baseline results.

Usage:
    python scripts/visualize_results.py --results_dir results/baseline

Outputs to results/baseline/figures/:
    comparison_bar.png          zero-shot vs linear probe top-k accuracy
    training_curve.png          linear probe loss and accuracy over epochs
    per_class_top.png           best countries by accuracy (both methods)
    per_class_bottom.png        worst countries by accuracy (both methods)
    accuracy_distribution.png   histogram of per-class accuracies
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------


def plot_comparison_bar(zs, lp, out_dir):
    """Bar chart comparing zero-shot vs linear probe."""
    metrics = ["top1_acc", "top5_acc", "top10_acc", "mean_class_acc"]
    labels = ["Top-1", "Top-5", "Top-10", "Mean Class"]

    zs_vals = [zs.get(m, 0) for m in metrics]
    lp_vals = [lp.get(m, 0) for m in metrics]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w / 2, zs_vals, w, label="Zero-Shot", color="#4C72B0")
    bars2 = ax.bar(x + w / 2, lp_vals, w, label="Linear Probe", color="#DD8452")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_ylabel("Accuracy")
    ax.set_title("CLIP ViT-L/14 Geolocation: Zero-Shot vs Linear Probe")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, min(1.0, max(max(zs_vals), max(lp_vals)) * 1.25))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_bar.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'comparison_bar.png'}")


def plot_training_curve(log, out_dir):
    """Loss and accuracy curves during linear probe training."""
    epochs = [e["epoch"] for e in log]
    train_loss = [e["train_loss"] for e in log]
    train_acc = [e["train_acc"] for e in log]
    test_top1 = [e["test_top1"] for e in log]
    test_top5 = [e["test_top5"] for e in log if e["test_top5"] is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_loss, "o-", color="#4C72B0")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Linear Probe Training Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_acc, "o-", label="Train Top-1", color="#4C72B0")
    ax2.plot(epochs, test_top1, "s-", label="Test Top-1", color="#DD8452")
    if test_top5:
        ax2.plot(epochs[:len(test_top5)], test_top5, "^-", label="Test Top-5", color="#55A868")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Linear Probe Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'training_curve.png'}")


def plot_per_class(zs_pc, lp_pc, label_map, out_dir, top_n=15):
    """Show top/bottom countries by accuracy for both methods."""
    # Sort by linear probe accuracy
    lp_sorted = sorted(lp_pc.items(), key=lambda x: x[1], reverse=True)

    top = lp_sorted[:top_n]
    bottom = lp_sorted[-top_n:]

    for subset, name in [(top, "top"), (bottom, "bottom")]:
        class_ids = [c for c, _ in subset]
        countries = [label_map.get(int(c), f"Class {c}") for c in class_ids]
        lp_accs = [lp_pc.get(c, 0) for c in class_ids]
        zs_accs = [zs_pc.get(c, 0) for c in class_ids]

        fig, ax = plt.subplots(figsize=(10, 6))
        y = np.arange(len(countries))
        h = 0.35
        ax.barh(y - h / 2, zs_accs, h, label="Zero-Shot", color="#4C72B0")
        ax.barh(y + h / 2, lp_accs, h, label="Linear Probe", color="#DD8452")
        ax.set_yticks(y)
        ax.set_yticklabels(countries, fontsize=9)
        ax.set_xlabel("Top-1 Accuracy")
        ax.set_title(f"{'Best' if name == 'top' else 'Worst'} {top_n} Countries (by Linear Probe)")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"per_class_{name}.png", dpi=150)
        plt.close(fig)
        print(f"Saved {out_dir / f'per_class_{name}.png'}")


def plot_accuracy_distribution(zs_pc, lp_pc, out_dir):
    """Histogram of per-class accuracies."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(list(zs_pc.values()), bins, alpha=0.6, label="Zero-Shot", color="#4C72B0")
    ax.hist(list(lp_pc.values()), bins, alpha=0.6, label="Linear Probe", color="#DD8452")
    ax.set_xlabel("Per-Class Top-1 Accuracy")
    ax.set_ylabel("Number of Countries")
    ax.set_title("Distribution of Per-Class Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'accuracy_distribution.png'}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results/baseline")
    args = p.parse_args()

    rd = Path(args.results_dir)
    fig_dir = rd / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    zs = load_json(rd / "zero_shot_metrics.json")
    lp = load_json(rd / "linear_probe_metrics.json")
    log = load_json(rd / "training_log.json")
    zs_pc = load_json(rd / "zero_shot_per_class.json")
    lp_pc = load_json(rd / "linear_probe_per_class.json")

    # Build label map from summary
    summary = load_json(rd / "summary.json")
    countries = summary.get("num_countries", [])
    label_map = {str(i): c for i, c in enumerate(countries)}

    plot_comparison_bar(zs, lp, fig_dir)
    plot_training_curve(log, fig_dir)
    plot_per_class(zs_pc, lp_pc, label_map, fig_dir)
    plot_accuracy_distribution(zs_pc, lp_pc, fig_dir)

    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
