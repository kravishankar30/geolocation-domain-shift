#!/usr/bin/env python3
"""Generate figures from LoRA training results.

Usage:
    python scripts/visualize_lora_results.py --results_dir results/lora

Outputs to results/lora/figures/:
    comparison_bar.png          Top-k and mean-class accuracy for LoRA
    training_curve.png          LoRA loss and accuracy over epochs
    per_class_top.png           best countries by LoRA per-class accuracy
    per_class_bottom.png        worst countries by LoRA per-class accuracy
    accuracy_distribution.png   histogram of LoRA per-class accuracies
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_metrics_bar(metrics, out_dir):
    keys = ["top1_acc", "top5_acc", "top10_acc", "mean_class_acc"]
    labels = ["Top-1", "Top-5", "Top-10", "Mean Class"]
    vals = [metrics.get(k, 0) for k in keys]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, vals, color="#55A868")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_ylabel("Accuracy")
    ax.set_title("CLIP ViT-L/14 LoRA Geolocation Metrics")
    ax.set_ylim(0, min(1.0, max(vals + [0.01]) * 1.25))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_bar.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'comparison_bar.png'}")


def plot_training_curve(log, out_dir):
    epochs = [e["epoch"] for e in log]
    train_loss = [e["train_loss"] for e in log]
    train_acc = [e["train_acc"] for e in log]
    test_top1 = [e["test_top1"] for e in log]
    test_top5 = [e["test_top5"] for e in log if e.get("test_top5") is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_loss, "o-", color="#4C72B0")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("LoRA Training Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_acc, "o-", label="Train Top-1", color="#4C72B0")
    ax2.plot(epochs, test_top1, "s-", label="Test Top-1", color="#DD8452")
    if test_top5:
        ax2.plot(epochs[: len(test_top5)], test_top5, "^-", label="Test Top-5", color="#55A868")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("LoRA Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'training_curve.png'}")


def plot_per_class(lora_pc, label_map, out_dir, top_n=15):
    ordered = sorted(lora_pc.items(), key=lambda x: x[1], reverse=True)
    top = ordered[:top_n]
    bottom = ordered[-top_n:]

    for subset, name in [(top, "top"), (bottom, "bottom")]:
        class_ids = [c for c, _ in subset]
        countries = [label_map.get(str(int(c)), f"Class {c}") for c in class_ids]
        accs = [lora_pc.get(c, 0) for c in class_ids]

        fig, ax = plt.subplots(figsize=(10, 6))
        y = np.arange(len(countries))
        ax.barh(y, accs, color="#55A868")
        ax.set_yticks(y)
        ax.set_yticklabels(countries, fontsize=9)
        ax.set_xlabel("Top-1 Accuracy")
        ax.set_title(f"{'Best' if name == 'top' else 'Worst'} {top_n} Countries (LoRA)")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"per_class_{name}.png", dpi=150)
        plt.close(fig)
        print(f"Saved {out_dir / f'per_class_{name}.png'}")


def plot_accuracy_distribution(lora_pc, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(list(lora_pc.values()), bins, alpha=0.8, label="LoRA", color="#55A868")
    ax.set_xlabel("Per-Class Top-1 Accuracy")
    ax.set_ylabel("Number of Countries")
    ax.set_title("Distribution of Per-Class Accuracy (LoRA)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'accuracy_distribution.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results/lora")
    args = p.parse_args()

    rd = Path(args.results_dir)
    fig_dir = rd / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_json(rd / "lora_metrics.json")
    log = load_json(rd / "lora_training_log.json")
    lora_pc = load_json(rd / "lora_per_class.json")
    summary = load_json(rd / "summary.json")

    class_names = summary.get("dataset", {}).get("class_names", [])
    label_map = {str(i): c for i, c in enumerate(class_names)}

    plot_metrics_bar(metrics, fig_dir)
    plot_training_curve(log, fig_dir)
    plot_per_class(lora_pc, label_map, fig_dir)
    plot_accuracy_distribution(lora_pc, fig_dir)

    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()

