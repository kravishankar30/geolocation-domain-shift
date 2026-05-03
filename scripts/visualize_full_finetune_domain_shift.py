#!/usr/bin/env python3
"""Generate figures for full fine-tuning domain shift experiments.

Usage:
    python scripts/visualize_full_finetune_domain_shift.py --results_dir results/full_finetune_domain_shift

Outputs to results/full_finetune_domain_shift/figures/:
    domain_shift_comparison.png
    degradation_chart.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_comparison(results, out_dir):
    conditions = list(results.keys())
    metrics = ["top1_acc", "top5_acc"]
    labels = ["Top-1", "Top-5"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, label in zip(axes, metrics, labels):
        values = [results[condition]["full_finetune"].get(metric, 0) for condition in conditions]
        x = np.arange(len(conditions))
        bars = ax.bar(x, values, color="#4C72B0")

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

        ax.set_ylabel("Accuracy")
        ax.set_title(f"{label} Accuracy Under Domain Shift")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=15, ha="right")
        ax.set_ylim(0, min(1.0, max(values) * 1.3 if values else 1.0))
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "domain_shift_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'domain_shift_comparison.png'}")


def plot_degradation(results, out_dir):
    conditions = [condition for condition in results if condition != "clean"]
    if not conditions:
        return

    clean_top1 = results["clean"]["full_finetune"].get("top1_acc", 0)
    drops = [clean_top1 - results[condition]["full_finetune"].get("top1_acc", 0) for condition in conditions]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(conditions))
    bars = ax.bar(x, drops, color="#C44E52")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.set_ylabel("Top-1 Accuracy Drop (pp)")
    ax.set_title("Full Fine-Tuning Accuracy Degradation Under Domain Shift")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "degradation_chart.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'degradation_chart.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/full_finetune_domain_shift")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figure_dir = results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    results = load_json(results_dir / "comparison.json")
    plot_comparison(results, figure_dir)
    plot_degradation(results, figure_dir)

    print(f"\nAll figures saved to {figure_dir}/")


if __name__ == "__main__":
    main()
