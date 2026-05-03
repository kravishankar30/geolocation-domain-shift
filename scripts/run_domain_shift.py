#!/usr/bin/env python3
"""Evaluate domain-shift robustness on clean/corrupted test images.

Supported modes:
1) baseline: evaluate zero-shot + linear-probe.
2) lora: evaluate a trained LoRA checkpoint directly.

Usage:
    python scripts/run_domain_shift.py --eval_mode baseline --baseline_dir results/baseline
    python scripts/run_domain_shift.py --eval_mode lora --lora_dir results/lora
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
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.osv5m_dataset import OSV5MDataset
from src.models.clip_geolocation import GeolocationCLIP


# ------------------------------------------------------------------
# Corruption transforms (applied to PIL images before CLIP preprocess)
# ------------------------------------------------------------------

CORRUPTIONS = {
    "gaussian_blur": lambda img: img.filter(ImageFilter.GaussianBlur(radius=5)),
    "brightness": lambda img: ImageEnhance.Brightness(img).enhance(2.5),
    "occlusion": lambda img: _apply_occlusion(img, fraction=0.3, seed=42),
}


def _apply_occlusion(img, fraction=0.3, seed=42):
    """Black out random rectangular patches covering ~fraction of pixels."""
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


# ------------------------------------------------------------------
# Dataset wrapper with corruption
# ------------------------------------------------------------------


class _CorruptedDataset(torch.utils.data.Dataset):
    """Wraps a dataset, applies a corruption to images before preprocessing."""
    def __init__(self, dataset, preprocess, corruption_fn):
        self.dataset = dataset
        self.preprocess = preprocess
        self.corruption_fn = corruption_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = self.corruption_fn(sample["image"])
        return self.preprocess(img), sample["label"]


# ------------------------------------------------------------------
# Helpers (shared with run_baseline.py)
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

    preds = logits.argmax(dim=1)
    per_class = {}
    for c in labels.unique().tolist():
        mask = labels == c
        per_class[c] = (preds[mask] == c).float().mean().item()
    metrics["per_class_acc"] = per_class
    metrics["mean_class_acc"] = np.mean(list(per_class.values()))
    return metrics


@torch.no_grad()
def extract_embeddings(dataset, model, device, batch_size=256):
    """Extract normalized embeddings using a DataLoader for parallel loading."""
    model.eval()
    all_embs, all_labels = [], []
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
    for images, labels in tqdm(loader, desc="Extracting embeddings"):
        images = images.to(device)
        embs = model.encode_image(images)
        embs = F.normalize(embs, dim=-1)
        all_embs.append(embs.cpu())
        all_labels.append(labels)
    return torch.cat(all_embs), torch.cat(all_labels)


@torch.no_grad()
def evaluate_zero_shot(embeddings, labels, model, device):
    """Zero-shot eval via cosine similarity."""
    model.set_mode("zero_shot")
    model.build_text_embeddings()
    text_embs = model.text_embeddings
    logit_scale = model.clip.logit_scale.exp().item()
    logits = logit_scale * (embeddings.to(device) @ text_embs.T)
    return compute_metrics(logits.cpu(), labels)


def train_linear_probe(train_embs, train_labels, num_classes, epochs=10, lr=1e-3, batch_size=256, device="cuda"):
    """Train linear probe, return the trained head."""
    head = nn.Linear(768, num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(train_embs, train_labels)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        head.train()
        for emb_batch, label_batch in loader:
            emb_batch, label_batch = emb_batch.to(device), label_batch.to(device)
            loss = F.cross_entropy(head(emb_batch), label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    head.eval()
    return head


@torch.no_grad()
def evaluate_linear_probe(head, embeddings, labels, device):
    """Evaluate a trained linear probe head on embeddings."""
    logits = head(embeddings.to(device)).cpu()
    return compute_metrics(logits, labels)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_mode", type=str, default="baseline", choices=["baseline", "lora"])
    p.add_argument("--baseline_dir", type=str, default="results/baseline")
    p.add_argument("--lora_dir", type=str, default="results/lora")
    p.add_argument("--lora_checkpoint", type=str, default="results/lora/lora_best_checkpoint.pt")
    p.add_argument("--output_dir", type=str, default="results/domain_shift")
    p.add_argument("--train_size", type=int, default=10_000)
    p.add_argument("--test_size", type=int, default=2_000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache_dir", type=str, default="./data/osv5m_cache")
    p.add_argument("--extract_dir", type=str, default="./data/osv5m_images")
    p.add_argument("--max_shards", type=int, default=1, help="Max shards to download (0=all)")
    return p.parse_args()


@torch.no_grad()
def evaluate_model_logits(dataset, model, device, batch_size=256):
    """Evaluate model outputs directly (used for LoRA mode)."""
    model.eval()
    all_logits, all_labels = [], []
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
    for images, labels in tqdm(loader, desc="Evaluating model"):
        images = images.to(device)
        logits = model(images).cpu()
        all_logits.append(logits)
        all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return compute_metrics(logits, labels)


def _save_metrics(metrics, out_path):
    metrics = dict(metrics)
    metrics.pop("per_class_acc", None)
    json.dump(metrics, open(out_path, "w"), indent=2)
    return metrics


def _build_test_split(args):
    print("\n--- Rebuilding test dataset ---")
    max_shards = args.max_shards if args.max_shards > 0 else None
    total_size = args.train_size + args.test_size
    full_ds = OSV5MDataset("train", total_size, args.seed, args.cache_dir, args.extract_dir, max_shards=max_shards)
    full_ds.download_images()
    n = len(full_ds)
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(n)
    n_test = min(args.test_size, n // 5)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:n_test + args.train_size]
    return full_ds, torch.utils.data.Subset(full_ds, train_idx), torch.utils.data.Subset(full_ds, test_idx)


def run_baseline_mode(args, device):
    baseline_dir = Path(args.baseline_dir)

    train_data = torch.load(baseline_dir / "embeddings" / "train_embs.pt", weights_only=True)
    train_embs, train_labels = train_data["embs"], train_data["labels"]
    print(f"Loaded clean train embeddings: {train_embs.shape}")

    summary = json.load(open(baseline_dir / "summary.json"))
    all_countries = summary["num_countries"]
    num_classes = len(all_countries)

    _, _, test_ds = _build_test_split(args)
    print(f"Test set: {len(test_ds)} images")

    model = GeolocationCLIP(num_classes=num_classes, class_names=all_countries, mode="zero_shot").to(device)
    model.eval()

    print("\n--- Training linear probe on clean data ---")
    head = train_linear_probe(
        train_embs, train_labels, num_classes,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=str(device),
    )

    clean_zs = json.load(open(baseline_dir / "zero_shot_metrics.json"))
    clean_lp = json.load(open(baseline_dir / "linear_probe_metrics.json"))
    all_results = {"clean": {"zero_shot": clean_zs, "linear_probe": clean_lp}}

    for corruption_name, corruption_fn in CORRUPTIONS.items():
        print(f"\n{'='*50}\nCorruption: {corruption_name}\n{'='*50}")
        t0 = time.time()
        corrupted_ds = _CorruptedDataset(test_ds, model.preprocess, corruption_fn)
        corrupted_embs, corrupted_labels = extract_embeddings(corrupted_ds, model, device, args.batch_size)
        print(f"Extraction took {time.time() - t0:.1f}s")

        zs_metrics = evaluate_zero_shot(corrupted_embs, corrupted_labels, model, device)
        lp_metrics = evaluate_linear_probe(head, corrupted_embs, corrupted_labels, device)

        corr_dir = Path(args.output_dir) / corruption_name
        os.makedirs(corr_dir, exist_ok=True)
        zs_metrics = _save_metrics(zs_metrics, corr_dir / "zero_shot_metrics.json")
        lp_metrics = _save_metrics(lp_metrics, corr_dir / "linear_probe_metrics.json")
        all_results[corruption_name] = {"zero_shot": zs_metrics, "linear_probe": lp_metrics}

    return all_results


def run_lora_mode(args, device):
    lora_dir = Path(args.lora_dir)
    summary = json.load(open(lora_dir / "summary.json"))
    class_names = summary["dataset"]["class_names"]
    num_classes = summary["dataset"]["num_classes"]

    # Reuse the same split configuration used during LoRA training by default.
    if "config" in summary:
        args.train_size = int(summary["config"].get("train_size", args.train_size))
        args.test_size = int(summary["config"].get("test_size", args.test_size))
        args.seed = int(summary["config"].get("seed", args.seed))
        args.max_shards = int(summary["config"].get("max_shards", args.max_shards))

    _, _, test_ds = _build_test_split(args)
    print(f"Test set: {len(test_ds)} images")

    ckpt_path = Path(args.lora_checkpoint) if args.lora_checkpoint else (lora_dir / "lora_checkpoint.pt")
    print(f"\n--- Loading LoRA checkpoint: {ckpt_path} ---")
    model = GeolocationCLIP(num_classes=num_classes, class_names=class_names, mode="lora").to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_results = {}

    clean_ds = _CorruptedDataset(test_ds, model.preprocess, lambda x: x)
    clean_metrics = evaluate_model_logits(clean_ds, model, device, args.batch_size)
    clean_dir = Path(args.output_dir) / "clean"
    os.makedirs(clean_dir, exist_ok=True)
    clean_metrics = _save_metrics(clean_metrics, clean_dir / "lora_metrics.json")
    all_results["clean"] = {"lora": clean_metrics}

    for corruption_name, corruption_fn in CORRUPTIONS.items():
        print(f"\n{'='*50}\nCorruption: {corruption_name}\n{'='*50}")
        corrupted_ds = _CorruptedDataset(test_ds, model.preprocess, corruption_fn)
        metrics = evaluate_model_logits(corrupted_ds, model, device, args.batch_size)
        corr_dir = Path(args.output_dir) / corruption_name
        os.makedirs(corr_dir, exist_ok=True)
        metrics = _save_metrics(metrics, corr_dir / "lora_metrics.json")
        all_results[corruption_name] = {"lora": metrics}

    return all_results


def print_summary(all_results):
    methods = sorted({m for cond in all_results.values() for m in cond.keys()})
    print(f"\n{'='*70}\nDOMAIN SHIFT RESULTS SUMMARY\n{'='*70}")
    header = f"{'Condition':<20}" + "".join([f"{m[:8]} T1".rjust(12) for m in methods])
    print(header)
    print("-" * len(header))
    for condition, payload in all_results.items():
        row = f"{condition:<20}"
        for m in methods:
            row += f"{payload.get(m, {}).get('top1_acc', 0):>12.4f}"
        print(row)


def merge_with_existing_results(output_dir, new_results):
    """Merge new method results into existing comparison.json without deleting others."""
    comparison_path = Path(output_dir) / "comparison.json"
    if comparison_path.exists():
        existing = json.load(open(comparison_path))
    else:
        existing = {}

    merged = dict(existing)
    for condition, payload in new_results.items():
        if condition not in merged:
            merged[condition] = {}
        merged[condition].update(payload)
    return merged


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    if args.eval_mode == "baseline":
        new_results = run_baseline_mode(args, device)
    else:
        new_results = run_lora_mode(args, device)

    all_results = merge_with_existing_results(args.output_dir, new_results)
    json.dump(all_results, open(Path(args.output_dir) / "comparison.json", "w"), indent=2)
    print_summary(all_results)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
