#!/usr/bin/env python3
"""Evaluate a full-finetuned checkpoint under domain shift corruptions.

Usage:
    python scripts/run_full_finetune_domain_shift.py --full_finetune_dir results/full_finetune

Outputs to results/full_finetune_domain_shift/:
    comparison.json
    summary.json
    <corruption>/full_finetune_metrics.json
    <corruption>/full_finetune_per_class.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.osv5m_dataset import OSV5MDataset
from src.models.clip_geolocation import GeolocationCLIP


def _apply_occlusion(img, fraction=0.3, seed=42):
    rng = np.random.RandomState(seed)
    arr = np.array(img)
    h, w = arr.shape[:2]
    total = int(w * h * fraction)
    blacked = 0
    while blacked < total:
        rw = rng.randint(w // 8, w // 3)
        rh = rng.randint(h // 8, h // 3)
        rx = rng.randint(0, w - rw)
        ry = rng.randint(0, h - rh)
        arr[ry:ry + rh, rx:rx + rw] = 0
        blacked += rw * rh
    return Image.fromarray(arr)


CORRUPTIONS = {
    "gaussian_blur": lambda img: img.filter(ImageFilter.GaussianBlur(radius=5)),
    "brightness": lambda img: ImageEnhance.Brightness(img).enhance(2.5),
    "occlusion": lambda img: _apply_occlusion(img, fraction=0.3, seed=42),
}


class _CorruptedDataset(Dataset):
    def __init__(self, dataset, preprocess, corruption_fn):
        self.dataset = dataset
        self.preprocess = preprocess
        self.corruption_fn = corruption_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.corruption_fn(sample["image"])
        return self.preprocess(image), sample["label"]


class _PreprocessedDataset(Dataset):
    def __init__(self, dataset, preprocess):
        self.dataset = dataset
        self.preprocess = preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return self.preprocess(sample["image"]), sample["label"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_finetune_dir", type=str, default="results/full_finetune")
    parser.add_argument("--output_dir", type=str, default="results/full_finetune_domain_shift")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--extract_dir", type=str, default=None)
    parser.add_argument("--disable_amp", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def compute_metrics(logits, labels, topk=(1, 5, 10)):
    metrics = {}
    for k in topk:
        if k > logits.shape[1]:
            continue
        _, pred = logits.topk(k, dim=1)
        correct = pred.eq(labels.unsqueeze(1)).any(dim=1).float()
        metrics[f"top{k}_acc"] = correct.mean().item()

    preds = logits.argmax(dim=1)
    per_class = {}
    for c in labels.unique().tolist():
        mask = labels == c
        per_class[c] = (preds[mask] == c).float().mean().item()
    metrics["per_class_acc"] = per_class
    metrics["mean_class_acc"] = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return metrics


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled: bool):
    model.eval()
    all_logits = []
    all_labels = []
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return compute_metrics(logits, labels)


def build_test_dataset(config, cache_dir: str, extract_dir: str):
    max_shards_value = config.get("max_test_shards")
    if max_shards_value is None:
        max_shards_value = config.get("max_shards", 1)
    max_shards = max_shards_value if max_shards_value > 0 else None

    test_ds = OSV5MDataset(
        split="test",
        subset_size=config["test_size"],
        seed=config["seed"],
        cache_dir=cache_dir,
        extract_dir=extract_dir,
        max_shards=max_shards,
    )
    test_ds.download_images()
    return test_ds


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ft_dir = Path(args.full_finetune_dir)
    summary = load_json(ft_dir / "summary.json")
    clean_metrics = load_json(ft_dir / "full_finetune_metrics.json")
    config = summary["config"]
    countries = summary["num_countries"]
    num_classes = len(countries)

    cache_dir = args.cache_dir or config.get("cache_dir", "./data/osv5m_cache")
    extract_dir = args.extract_dir or config.get("extract_dir", "./data/osv5m_images")
    device = torch.device(args.device)
    amp_enabled = (not args.disable_amp) and device.type == "cuda"

    print("\n--- Rebuilding clean official test subset ---")
    t0 = time.time()
    test_ds = build_test_dataset(config, cache_dir=cache_dir, extract_dir=extract_dir)
    mapping = {country: i for i, country in enumerate(countries)}
    reverse = {i: country for country, i in mapping.items()}
    test_ds.country_to_label = mapping
    test_ds.label_to_country = reverse
    print(f"Test set ready in {time.time() - t0:.1f}s ({len(test_ds)} images)")

    print("\n--- Loading full-finetuned checkpoint ---")
    model = GeolocationCLIP(
        num_classes=num_classes,
        class_names=countries,
        mode="full_finetune",
    ).to(device)
    checkpoint = torch.load(ft_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_results = {
        "clean": {"full_finetune": clean_metrics},
    }

    for corruption_name, corruption_fn in CORRUPTIONS.items():
        print(f"\n{'=' * 50}")
        print(f"Corruption: {corruption_name}")
        print(f"{'=' * 50}")

        corr_dir = output_dir / corruption_name
        corr_dir.mkdir(parents=True, exist_ok=True)

        corrupted_ds = _CorruptedDataset(test_ds, model.preprocess, corruption_fn)
        loader = DataLoader(
            corrupted_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
        )

        start = time.time()
        metrics = evaluate(model, loader, device, amp_enabled)
        print(
            f"  Full fine-tune top-1: {metrics['top1_acc']:.4f}, "
            f"top-5: {metrics.get('top5_acc', 'N/A')}"
        )
        print(f"  Evaluation time: {time.time() - start:.1f}s")

        per_class = metrics.pop("per_class_acc")
        save_json(metrics, corr_dir / "full_finetune_metrics.json")
        save_json({str(k): v for k, v in per_class.items()}, corr_dir / "full_finetune_per_class.json")
        all_results[corruption_name] = {"full_finetune": metrics}

    save_json(all_results, output_dir / "comparison.json")

    summary_out = {
        "method": "Full fine-tuning",
        "source_run": str(ft_dir),
        "config": config,
        "num_classes": num_classes,
        "num_countries": countries,
        "results": all_results,
    }
    save_json(summary_out, output_dir / "summary.json")

    print(f"\n{'=' * 70}")
    print("FULL FINETUNE DOMAIN SHIFT SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Condition':<20} {'FT Top-1':>10} {'FT Top-5':>10}")
    print("-" * 70)
    for name, result in all_results.items():
        ft = result["full_finetune"]
        print(f"{name:<20} {ft.get('top1_acc', 0):>10.4f} {ft.get('top5_acc', 0):>10.4f}")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
