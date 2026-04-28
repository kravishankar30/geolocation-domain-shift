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
    """Grouped bar chart: all conditions, zero-shot and linear probe side by side."""
    conditions = list(results.keys())
    metrics = ["top1_acc", "top5_acc"]
    metric_labels = ["Top-1", "Top-5"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (metric, mlabel) in zip(axes, zip(metrics, metric_labels)):
        zs_vals = [results[c]["zero_shot"].get(metric, 0) for c in conditions]
        lp_vals = [results[c]["linear_probe"].get(metric, 0) for c in conditions]

        x = np.arange(len(conditions))
        w = 0.35

        bars1 = ax.bar(x - w / 2, zs_vals, w, label="Zero-Shot", color="#4C72B0")
        bars2 = ax.bar(x + w / 2, lp_vals, w, label="Linear Probe", color="#DD8452")

        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

        ax.set_ylabel("Accuracy")
        ax.set_title(f"{mlabel} Accuracy Under Domain Shift")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=15, ha="right")
        ax.legend()
        ax.set_ylim(0, min(1.0, max(max(zs_vals), max(lp_vals)) * 1.3))
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

    clean_zs = results["clean"]["zero_shot"].get("top1_acc", 0)
    clean_lp = results["clean"]["linear_probe"].get("top1_acc", 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(conditions))
    w = 0.35

    # Compute relative degradation (percentage points dropped)
    zs_drops = [clean_zs - results[c]["zero_shot"].get("top1_acc", 0) for c in conditions]
    lp_drops = [clean_lp - results[c]["linear_probe"].get("top1_acc", 0) for c in conditions]

    bars1 = ax.bar(x - w / 2, zs_drops, w, label="Zero-Shot", color="#4C72B0")
    bars2 = ax.bar(x + w / 2, lp_drops, w, label="Linear Probe", color="#DD8452")

    for bars in [bars1, bars2]:
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
