#!/usr/bin/env python3
"""Baseline evaluation: zero-shot + linear probe on OSV-5M subset.

Usage:
    python scripts/run_baseline.py --train_size 10000 --test_size 2000 --epochs 10 --batch_size 256

Outputs saved to results/baseline/:
    embeddings/             cached image embeddings (reusable across runs)
    zero_shot_metrics.json
    linear_probe_metrics.json
    training_log.json
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.osv5m_dataset import OSV5MDataset
from src.models.clip_geolocation import GeolocationCLIP


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_size", type=int, default=10_000)
    p.add_argument("--test_size", type=int, default=2_000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, default="results/baseline")
    p.add_argument("--cache_dir", type=str, default="./data/osv5m_cache")
    p.add_argument("--extract_dir", type=str, default="./data/osv5m_images")
    p.add_argument("--max_shards", type=int, default=1, help="Max shards to download (0=all)")
    return p.parse_args()


# ------------------------------------------------------------------
# Embedding extraction
# ------------------------------------------------------------------


@torch.no_grad()
def extract_embeddings(dataset, model, device, batch_size=256):
    """Extract normalized image embeddings and labels from dataset."""
    model.eval()
    all_embs, all_labels = [], []

    # Manual batching since dataset returns dicts with PIL images
    preprocess = model.preprocess
    for start in tqdm(range(0, len(dataset), batch_size), desc="Extracting embeddings"):
        end = min(start + batch_size, len(dataset))
        images, labels = [], []
        for i in range(start, end):
            sample = dataset[i]
            images.append(preprocess(sample["image"]))
            labels.append(sample["label"])

        images = torch.stack(images).to(device)
        embs = model.encode_image(images)
        embs = F.normalize(embs, dim=-1)

        all_embs.append(embs.cpu())
        all_labels.append(torch.tensor(labels))

    return torch.cat(all_embs), torch.cat(all_labels)


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------


def compute_metrics(logits, labels, topk=(1, 5, 10)):
    """Compute top-k accuracy and per-class accuracy."""
    metrics = {}
    for k in topk:
        if k > logits.shape[1]:
            continue
        _, pred = logits.topk(k, dim=1)
        correct = pred.eq(labels.unsqueeze(1)).any(dim=1).float()
        metrics[f"top{k}_acc"] = correct.mean().item()

    # Per-class accuracy (top-1 only)
    preds = logits.argmax(dim=1)
    per_class = {}
    for c in labels.unique().tolist():
        mask = labels == c
        per_class[c] = (preds[mask] == c).float().mean().item()
    metrics["per_class_acc"] = per_class
    metrics["mean_class_acc"] = np.mean(list(per_class.values()))

    return metrics


# ------------------------------------------------------------------
# Zero-shot
# ------------------------------------------------------------------


@torch.no_grad()
def evaluate_zero_shot(embeddings, labels, model, device):
    """Cosine similarity between image embeddings and text class embeddings."""
    print("\n=== Zero-Shot Evaluation ===")
    model.to(device)
    model.set_mode("zero_shot")
    model.build_text_embeddings()

    text_embs = model.text_embeddings  # (C, d)
    logit_scale = model.clip.logit_scale.exp().item()

    # Already normalized from extraction
    logits = logit_scale * (embeddings.to(device) @ text_embs.T)
    logits = logits.cpu()

    metrics = compute_metrics(logits, labels)
    print(f"  Top-1: {metrics['top1_acc']:.4f}")
    print(f"  Top-5: {metrics.get('top5_acc', 'N/A')}")
    print(f"  Top-10: {metrics.get('top10_acc', 'N/A')}")
    print(f"  Mean class acc: {metrics['mean_class_acc']:.4f}")
    return metrics


# ------------------------------------------------------------------
# Linear probe
# ------------------------------------------------------------------


def train_linear_probe(
    train_embs, train_labels, test_embs, test_labels,
    num_classes, embed_dim=768, epochs=10, lr=1e-3, batch_size=256, device="cuda",
):
    """Train a linear classifier on frozen embeddings."""
    print("\n=== Linear Probe Training ===")
    head = nn.Linear(embed_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(train_embs, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    training_log = []

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss, correct, total = 0.0, 0, 0
        for emb_batch, label_batch in train_loader:
            emb_batch, label_batch = emb_batch.to(device), label_batch.to(device)
            logits = head(emb_batch)
            loss = F.cross_entropy(logits, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(label_batch)
            correct += (logits.argmax(1) == label_batch).sum().item()
            total += len(label_batch)

        scheduler.step()
        train_acc = correct / total
        train_loss = total_loss / total

        # Eval on test set
        head.eval()
        with torch.no_grad():
            test_logits = head(test_embs.to(device)).cpu()
        test_metrics = compute_metrics(test_logits, test_labels)

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_top1": test_metrics["top1_acc"],
            "test_top5": test_metrics.get("top5_acc"),
            "test_top10": test_metrics.get("top10_acc"),
            "test_mean_class_acc": test_metrics["mean_class_acc"],
            "lr": scheduler.get_last_lr()[0],
        }
        training_log.append(entry)
        print(f"  Epoch {epoch}/{epochs} - loss: {train_loss:.4f}, "
              f"train: {train_acc:.4f}, test top-1: {test_metrics['top1_acc']:.4f}, "
              f"top-5: {test_metrics.get('top5_acc', 'N/A')}")

    # Final eval
    head.eval()
    with torch.no_grad():
        test_logits = head(test_embs.to(device)).cpu()
    final_metrics = compute_metrics(test_logits, test_labels)

    return final_metrics, training_log


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    emb_dir = Path(args.output_dir) / "embeddings"
    os.makedirs(emb_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load data
    print("\n--- Loading datasets ---")
    max_shards = args.max_shards if args.max_shards > 0 else None
    train_ds = OSV5MDataset("train", args.train_size, args.seed, args.cache_dir, args.extract_dir, max_shards=max_shards)
    test_ds = OSV5MDataset("test", args.test_size, args.seed, args.cache_dir, args.extract_dir, max_shards=max_shards)

    # Harmonize label mapping across splits
    all_countries = sorted(set(train_ds.country_to_label) | set(test_ds.country_to_label))
    mapping = {c: i for i, c in enumerate(all_countries)}
    reverse = {i: c for c, i in mapping.items()}
    for ds in (train_ds, test_ds):
        ds.country_to_label = mapping
        ds.label_to_country = reverse
    num_classes = len(all_countries)
    print(f"Countries: {num_classes}, Train: {len(train_ds)}, Test: {len(test_ds)}")

    # Download images
    print("\n--- Downloading images ---")
    t0 = time.time()
    train_ds.download_images()
    test_ds.download_images()
    print(f"Download took {time.time() - t0:.1f}s")

    # Build model (encoder + text embeddings)
    print("\n--- Loading CLIP ViT-L/14 ---")
    model = GeolocationCLIP(
        num_classes=num_classes,
        class_names=all_countries,
        mode="zero_shot",
    )
    model.to(device)
    model.eval()

    # Extract embeddings (cached)
    train_emb_path = emb_dir / "train_embs.pt"
    test_emb_path = emb_dir / "test_embs.pt"

    if train_emb_path.exists() and test_emb_path.exists():
        print("Loading cached embeddings...")
        train_data = torch.load(train_emb_path, weights_only=True)
        test_data = torch.load(test_emb_path, weights_only=True)
        train_embs, train_labels = train_data["embs"], train_data["labels"]
        test_embs, test_labels = test_data["embs"], test_data["labels"]
    else:
        print("\n--- Extracting embeddings ---")
        t0 = time.time()
        train_embs, train_labels = extract_embeddings(train_ds, model, device, args.batch_size)
        test_embs, test_labels = extract_embeddings(test_ds, model, device, args.batch_size)
        print(f"Extraction took {time.time() - t0:.1f}s")
        torch.save({"embs": train_embs, "labels": train_labels}, train_emb_path)
        torch.save({"embs": test_embs, "labels": test_labels}, test_emb_path)

    print(f"Train embeddings: {train_embs.shape}, Test embeddings: {test_embs.shape}")

    # Zero-shot eval
    zs_metrics = evaluate_zero_shot(test_embs, test_labels, model, device)
    # Save per-class separately (large)
    per_class_zs = zs_metrics.pop("per_class_acc")
    with open(Path(args.output_dir) / "zero_shot_metrics.json", "w") as f:
        json.dump(zs_metrics, f, indent=2)
    with open(Path(args.output_dir) / "zero_shot_per_class.json", "w") as f:
        json.dump({str(k): v for k, v in per_class_zs.items()}, f, indent=2)

    # Linear probe
    lp_metrics, training_log = train_linear_probe(
        train_embs, train_labels, test_embs, test_labels,
        num_classes=num_classes,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=str(device),
    )
    per_class_lp = lp_metrics.pop("per_class_acc")
    with open(Path(args.output_dir) / "linear_probe_metrics.json", "w") as f:
        json.dump(lp_metrics, f, indent=2)
    with open(Path(args.output_dir) / "linear_probe_per_class.json", "w") as f:
        json.dump({str(k): v for k, v in per_class_lp.items()}, f, indent=2)
    with open(Path(args.output_dir) / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    # Summary
    print("\n" + "=" * 50)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 50)
    print(f"{'Metric':<25} {'Zero-Shot':>12} {'Linear Probe':>12}")
    print("-" * 50)
    for k in ["top1_acc", "top5_acc", "top10_acc", "mean_class_acc"]:
        zv = zs_metrics.get(k)
        lv = lp_metrics.get(k)
        zs = f"{zv:.4f}" if zv is not None else "N/A"
        ls = f"{lv:.4f}" if lv is not None else "N/A"
        print(f"{k:<25} {zs:>12} {ls:>12}")

    # Save combined summary
    summary = {
        "config": vars(args),
        "num_classes": num_classes,
        "num_countries": all_countries,
        "zero_shot": zs_metrics,
        "linear_probe": lp_metrics,
    }
    with open(Path(args.output_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()