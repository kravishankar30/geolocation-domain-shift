#!/usr/bin/env python3
"""Generate comparison figures for domain shift experiments.

Usage:
    python scripts/visualize_domain_shift.py --results_dir results/domain_shift

Outputs to results/domain_shift/figures/:
    domain_shift_comparison.png     grouped bar chart of all conditions
    degradation_chart.png           accuracy drop from clean baseline
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
    """Grouped bar chart for all available methods in comparison.json."""
    conditions = list(results.keys())
    methods = sorted({m for cond in results.values() for m in cond.keys()})
    if not methods:
        raise ValueError("No methods found in results.")
    metrics = ["top1_acc", "top5_acc"]
    metric_labels = ["Top-1", "Top-5"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    for ax, (metric, mlabel) in zip(axes, zip(metrics, metric_labels)):
        x = np.arange(len(conditions))
        w = 0.8 / len(methods)
        offsets = np.linspace(-0.4 + w / 2, 0.4 - w / 2, len(methods))

        for idx, method in enumerate(methods):
            vals = [results[c].get(method, {}).get(metric, 0) for c in conditions]
            bars = ax.bar(
                x + offsets[idx],
                vals,
                w,
                label=method.replace("_", " ").title(),
                color=colors[idx % len(colors)],
            )
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

        ax.set_ylabel("Accuracy")
        ax.set_title(f"{mlabel} Accuracy Under Domain Shift")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=15, ha="right")
        ax.legend()
        max_y = max(
            [results[c].get(method, {}).get(metric, 0) for c in conditions for method in methods] + [0.01]
        )
        ax.set_ylim(0, min(1.0, max_y * 1.3))
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "domain_shift_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'domain_shift_comparison.png'}")


def plot_degradation(results, out_dir):
    """Show accuracy drop relative to clean baseline."""
    conditions = [c for c in results if c != "clean"]
    if not conditions:
        return

    methods = sorted({m for cond in results.values() for m in cond.keys()})
    if not methods:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(conditions))
    w = 0.8 / len(methods)
    offsets = np.linspace(-0.4 + w / 2, 0.4 - w / 2, len(methods))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    for idx, method in enumerate(methods):
        clean_top1 = results.get("clean", {}).get(method, {}).get("top1_acc", 0)
        drops = [clean_top1 - results[c].get(method, {}).get("top1_acc", 0) for c in conditions]
        bars = ax.bar(
            x + offsets[idx],
            drops,
            w,
            label=method.replace("_", " ").title(),
            color=colors[idx % len(colors)],
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_ylabel("Top-1 Accuracy Drop (pp)")
    ax.set_title("Accuracy Degradation Under Domain Shift (higher = worse)")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "degradation_chart.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'degradation_chart.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results/domain_shift")
    args = p.parse_args()

    rd = Path(args.results_dir)
    fig_dir = rd / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results = load_json(rd / "comparison.json")

    plot_comparison(results, fig_dir)
    plot_degradation(results, fig_dir)

    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
