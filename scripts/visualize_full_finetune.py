#!/usr/bin/env python3
"""Generate figures from full fine-tuning results.

Usage:
    python scripts/visualize_full_finetune.py --results_dir results/full_finetune

Outputs to results/full_finetune/figures/:
    metrics_bar.png              final Top-1/5/10 and mean-class accuracy
    training_curve.png           train loss and train/validation accuracy
    lr_curve.png                 encoder/head learning-rate schedule
    per_class_top.png            best countries by full-finetune accuracy
    per_class_bottom.png         worst countries by full-finetune accuracy
    accuracy_distribution.png    histogram of per-class accuracies
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
    metric_keys = ["top1_acc", "top5_acc", "top10_acc", "mean_class_acc"]
    labels = ["Top-1", "Top-5", "Top-10", "Mean Class"]
    values = [metrics.get(key, 0.0) for key in metric_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.set_ylabel("Accuracy")
    ax.set_title("Full Fine-Tuning Final Test Metrics")
    ax.set_ylim(0, min(1.0, max(values) * 1.25 if values else 1.0))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_bar.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'metrics_bar.png'}")


def plot_training_curve(log, out_dir):
    epochs = [entry["epoch"] for entry in log]
    train_loss = [entry["train_loss"] for entry in log]
    train_acc = [entry["train_acc"] for entry in log]
    val_top1 = [entry["val_top1"] for entry in log]
    val_top5 = [entry["val_top5"] for entry in log if entry["val_top5"] is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_loss, "o-", color="#4C72B0")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Full Fine-Tuning Training Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_acc, "o-", label="Train Top-1", color="#4C72B0")
    ax2.plot(epochs, val_top1, "s-", label="Val Top-1", color="#DD8452")
    if val_top5:
        ax2.plot(epochs[: len(val_top5)], val_top5, "^-", label="Val Top-5", color="#55A868")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Full Fine-Tuning Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'training_curve.png'}")


def plot_lr_curve(log, out_dir):
    epochs = [entry["epoch"] for entry in log]
    encoder_lr = [entry["encoder_lr"] for entry in log]
    head_lr = [entry["head_lr"] for entry in log]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, encoder_lr, "o-", label="Encoder LR", color="#4C72B0")
    ax.plot(epochs, head_lr, "s-", label="Head LR", color="#DD8452")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Full Fine-Tuning Learning Rate Schedule")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "lr_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'lr_curve.png'}")


def plot_per_class(ft_pc, label_map, out_dir, top_n=15):
    ft_sorted = sorted(ft_pc.items(), key=lambda x: x[1], reverse=True)
    top = ft_sorted[:top_n]
    bottom = ft_sorted[-top_n:]

    for subset, name in [(top, "top"), (bottom, "bottom")]:
        class_ids = [class_id for class_id, _ in subset]
        countries = [label_map.get(str(int(class_id)), f"Class {class_id}") for class_id in class_ids]
        values = [ft_pc.get(class_id, 0) for class_id in class_ids]

        fig, ax = plt.subplots(figsize=(10, 6))
        y = np.arange(len(countries))
        ax.barh(y, values, color="#4C72B0")
        ax.set_yticks(y)
        ax.set_yticklabels(countries, fontsize=9)
        ax.set_xlabel("Top-1 Accuracy")
        ax.set_title(f"{'Best' if name == 'top' else 'Worst'} {top_n} Countries (Full Fine-Tuning)")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"per_class_{name}.png", dpi=150)
        plt.close(fig)
        print(f"Saved {out_dir / f'per_class_{name}.png'}")


def plot_accuracy_distribution(ft_pc, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(list(ft_pc.values()), bins, alpha=0.8, color="#4C72B0")
    ax.set_xlabel("Per-Class Top-1 Accuracy")
    ax.set_ylabel("Number of Countries")
    ax.set_title("Distribution of Per-Class Accuracy")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'accuracy_distribution.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/full_finetune")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figure_dir = results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_json(results_dir / "full_finetune_metrics.json")
    training_log = load_json(results_dir / "training_log.json")
    per_class = load_json(results_dir / "full_finetune_per_class.json")
    summary = load_json(results_dir / "summary.json")

    countries = summary.get("num_countries", [])
    label_map = {str(i): country for i, country in enumerate(countries)}

    plot_metrics_bar(metrics, figure_dir)
    plot_training_curve(training_log, figure_dir)
    plot_lr_curve(training_log, figure_dir)
    plot_per_class(per_class, label_map, figure_dir)
    plot_accuracy_distribution(per_class, figure_dir)

    print(f"\nAll figures saved to {figure_dir}/")


if __name__ == "__main__":
    main()
