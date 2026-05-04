#!/usr/bin/env python3
"""Create split-aware comparison figures across baseline, LoRA, and full fine-tuning.

This script is intentionally conservative about direct cross-method claims:
- Zero-shot, linear probe, and LoRA are shown together on the pooled-random split.
- Full fine-tuning is shown separately on the official train/val/test split.
- LoRA vs full fine-tuning domain-shift comparison is expressed primarily as
  retention/degradation relative to each method's own clean baseline.

Usage:
    python scripts/visualize_method_comparisons.py

Outputs to results/comparisons/:
    clean_split_aware_comparison.png
    full_finetune_overfitting.png
    lora_vs_full_finetune_domain_shift.png
    summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "comparisons"


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_summary(summary):
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def annotate_bars(ax, bars, fmt="{:.3f}", fontsize=8):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=fontsize,
        )


def plot_clean_split_aware_comparison(zs, lp, lora, full_ft):
    metrics = ["top1_acc", "top5_acc", "top10_acc", "mean_class_acc"]
    labels = ["Top-1", "Top-5", "Top-10", "Mean Class"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    pooled_methods = {
        "Zero-Shot": zs,
        "Linear Probe": lp,
        "LoRA": lora,
    }
    colors = {
        "Zero-Shot": "#4C72B0",
        "Linear Probe": "#DD8452",
        "LoRA": "#55A868",
        "Full FT": "#C44E52",
    }

    x = np.arange(len(labels))
    width = 0.24
    offsets = [-width, 0.0, width]

    ax = axes[0]
    for (name, metrics_dict), offset in zip(pooled_methods.items(), offsets):
        values = [metrics_dict.get(metric, 0) for metric in metrics]
        bars = ax.bar(x + offset, values, width, label=name, color=colors[name])
        annotate_bars(ax, bars)

    ax.set_title("Pooled Random Split: Baseline and LoRA")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    ax = axes[1]
    ft_values = [full_ft.get(metric, 0) for metric in metrics]
    bars = ax.bar(labels, ft_values, color=colors["Full FT"])
    annotate_bars(ax, bars)
    ax.set_title("Official Train/Val/Test Split: Full Fine-Tuning")
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    fig.suptitle(
        "Clean Evaluation Comparison\n"
        "Do not directly rank pooled-split methods against official-split full fine-tuning",
        fontsize=13,
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "clean_split_aware_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_full_finetune_overfitting(training_log, best_epoch):
    epochs = [entry["epoch"] for entry in training_log]
    train_acc = [entry["train_acc"] for entry in training_log]
    val_top1 = [entry["val_top1"] for entry in training_log]
    train_loss = [entry["train_loss"] for entry in training_log]
    val_loss = [entry["val_loss"] for entry in training_log]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(epochs, train_acc, "o-", label="Train Top-1", color="#4C72B0")
    ax.plot(epochs, val_top1, "s-", label="Val Top-1", color="#C44E52")
    ax.axvline(best_epoch, linestyle="--", color="black", alpha=0.6, label=f"Best epoch = {best_epoch}")
    ax.set_title("Full Fine-Tuning Accuracy Trajectory")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, train_loss, "o-", label="Train Loss", color="#4C72B0")
    ax.plot(epochs, val_loss, "s-", label="Val Loss", color="#C44E52")
    ax.axvline(best_epoch, linestyle="--", color="black", alpha=0.6)
    ax.set_title("Full Fine-Tuning Loss Trajectory")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Full Fine-Tuning Overfits Quickly: validation peaks early while train keeps improving",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "full_finetune_overfitting.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lora_vs_full_ft_domain_shift(lora_ds, full_ft_ds):
    conditions = ["clean", "gaussian_blur", "brightness", "occlusion"]
    pretty = {
        "clean": "Clean",
        "gaussian_blur": "Gaussian Blur",
        "brightness": "Brightness",
        "occlusion": "Occlusion",
    }

    lora_clean = lora_ds["clean"]["lora"]["top1_acc"]
    ft_clean = full_ft_ds["clean"]["full_finetune"]["top1_acc"]

    lora_abs = [lora_ds[c]["lora"]["top1_acc"] for c in conditions]
    ft_abs = [full_ft_ds[c]["full_finetune"]["top1_acc"] for c in conditions]
    lora_retention = [value / lora_clean for value in lora_abs]
    ft_retention = [value / ft_clean for value in ft_abs]
    lora_drop = [lora_clean - value for value in lora_abs[1:]]
    ft_drop = [ft_clean - value for value in ft_abs[1:]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x = np.arange(len(conditions))
    width = 0.35
    ax = axes[0]
    bars1 = ax.bar(x - width / 2, lora_abs, width, label="LoRA (pooled split)", color="#55A868")
    bars2 = ax.bar(x + width / 2, ft_abs, width, label="Full FT (official split)", color="#C44E52")
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)
    ax.set_title("Absolute Top-1 Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[c] for c in conditions], rotation=15, ha="right")
    ax.set_ylabel("Top-1 Accuracy")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    bars1 = ax.bar(x - width / 2, lora_retention, width, label="LoRA retention", color="#55A868")
    bars2 = ax.bar(x + width / 2, ft_retention, width, label="Full FT retention", color="#C44E52")
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)
    ax.set_title("Top-1 Retention vs Each Method's Clean Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[c] for c in conditions], rotation=15, ha="right")
    ax.set_ylabel("Retention Ratio")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    x_drop = np.arange(len(conditions) - 1)
    ax = axes[2]
    bars1 = ax.bar(x_drop - width / 2, lora_drop, width, label="LoRA drop", color="#55A868")
    bars2 = ax.bar(x_drop + width / 2, ft_drop, width, label="Full FT drop", color="#C44E52")
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)
    ax.set_title("Top-1 Accuracy Drop Under Corruption")
    ax.set_xticks(x_drop)
    ax.set_xticklabels([pretty[c] for c in conditions[1:]], rotation=15, ha="right")
    ax.set_ylabel("Absolute Top-1 Drop")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "LoRA vs Full Fine-Tuning Under Domain Shift\n"
        "Absolute scores are split-mismatched; retention/drop trends are the safer comparison",
        fontsize=13,
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lora_vs_full_finetune_domain_shift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    zs = load_json(ROOT / "results" / "baseline" / "zero_shot_metrics.json")
    lp = load_json(ROOT / "results" / "baseline" / "linear_probe_metrics.json")
    lora = load_json(ROOT / "results" / "lora" / "lora_metrics.json")
    lora_summary = load_json(ROOT / "results" / "lora" / "summary.json")
    full_ft_summary = load_json(ROOT / "results" / "full_finetune" / "summary.json")
    full_ft = full_ft_summary["test_metrics"]
    full_ft_log = load_json(ROOT / "results" / "full_finetune" / "training_log.json")
    pooled_domain_shift = load_json(ROOT / "results" / "domain_shift" / "comparison.json")
    full_ft_domain_shift = load_json(ROOT / "results" / "full_finetune_domain_shift" / "comparison.json")

    plot_clean_split_aware_comparison(zs, lp, lora, full_ft)
    plot_full_finetune_overfitting(full_ft_log, full_ft_summary["best_epoch"])
    plot_lora_vs_full_ft_domain_shift(pooled_domain_shift, full_ft_domain_shift)

    summary = {
        "notes": [
            "Zero-shot, linear probe, and LoRA use the pooled random split from OSV-5M train.",
            "Full fine-tuning uses the official train/val/test split.",
            "Only split-matched comparisons should be interpreted quantitatively.",
            "LoRA vs full fine-tuning domain-shift plots should emphasize retention/drop trends over absolute score ranking.",
        ],
        "clean_metrics": {
            "pooled_split": {
                "zero_shot": zs,
                "linear_probe": lp,
                "lora": lora,
            },
            "official_split": {
                "full_finetune": full_ft,
            },
        },
        "domain_shift_sources": {
            "pooled_split": pooled_domain_shift,
            "official_split": full_ft_domain_shift,
        },
        "lora_clean_top1": lora_summary["final_metrics"]["top1_acc"],
        "full_finetune_clean_top1": full_ft_summary["test_metrics"]["top1_acc"],
        "full_finetune_best_epoch": full_ft_summary["best_epoch"],
    }
    save_summary(summary)

    print(f"Saved comparison figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()
