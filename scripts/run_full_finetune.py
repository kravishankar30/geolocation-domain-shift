#!/usr/bin/env python3
"""Full fine-tuning of CLIP ViT-L/14 for country-level geolocation.

This script trains the image encoder and classification head end to end on a
subset of OSV-5M, using an in-training validation split for early stopping and
the official OSV-5M test split for final evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.osv5m_dataset import OSV5MDataset
from src.models.clip_geolocation import GeolocationCLIP


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_size", type=int, default=50_000)
    p.add_argument("--val_size", type=int, default=5_000)
    p.add_argument("--test_size", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--encoder_lr", type=float, default=1e-6)
    p.add_argument("--head_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, default="results/full_finetune")
    p.add_argument("--cache_dir", type=str, default="./data/osv5m_cache")
    p.add_argument("--extract_dir", type=str, default="./data/osv5m_images")
    p.add_argument("--max_shards", type=int, default=1, help="Max train shards to download (0=all)")
    p.add_argument("--max_test_shards", type=int, default=None, help="Max test shards to download (defaults to --max_shards)")
    p.add_argument("--save_best", action="store_true", help="Save best checkpoint by validation Top-1 accuracy.")
    p.add_argument("--disable_amp", action="store_true")
    return p.parse_args()


class _PreprocessedDataset(Dataset):
    """Wraps a dataset to apply the CLIP preprocessing transform on the fly."""

    def __init__(self, dataset, preprocess):
        self.dataset = dataset
        self.preprocess = preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return self.preprocess(sample["image"]), sample["label"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def get_system_info(device):
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info.update(
            {
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    return info


def save_json(obj, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def stratified_split_indices(metadata, train_size: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    """Split rows into train/val while preserving country proportions as closely as possible."""
    if train_size <= 0:
        raise ValueError("train_size must be positive")
    if val_size < 0:
        raise ValueError("val_size must be non-negative")
    if train_size + val_size > len(metadata):
        raise ValueError(
            f"Requested train_size + val_size = {train_size + val_size}, "
            f"but only {len(metadata)} train examples are available."
        )

    rng = np.random.RandomState(seed)
    grouped = metadata.groupby("country").indices
    total_examples = len(metadata)
    if val_size == 0:
        all_indices = np.arange(total_examples, dtype=int)
        rng.shuffle(all_indices)
        return all_indices[:train_size].tolist(), []

    # Allocate validation quotas proportionally while guaranteeing that any class
    # contributing validation examples still leaves at least one training example.
    per_country = []
    max_val_capacity = 0
    for country in sorted(grouped):
        idxs = np.array(grouped[country], dtype=int)
        rng.shuffle(idxs)
        count = len(idxs)
        max_val = max(0, count - 1)
        raw_quota = (val_size * count) / total_examples
        base_quota = min(max_val, int(np.floor(raw_quota)))
        per_country.append(
            {
                "country": country,
                "indices": idxs,
                "count": count,
                "max_val": max_val,
                "val_quota": base_quota,
                "remainder": raw_quota - np.floor(raw_quota),
            }
        )
        max_val_capacity += max_val

    if max_val_capacity < val_size:
        raise RuntimeError(
            "Unable to create the requested validation split because too many "
            "classes have only one example. Reduce val_size or increase shard coverage."
        )

    current_val = sum(entry["val_quota"] for entry in per_country)
    if current_val < val_size:
        candidates = [entry for entry in per_country if entry["val_quota"] < entry["max_val"]]
        candidates.sort(
            key=lambda entry: (
                -entry["remainder"],
                -entry["count"],
                entry["country"],
            )
        )
        idx = 0
        while current_val < val_size:
            entry = candidates[idx % len(candidates)]
            if entry["val_quota"] < entry["max_val"]:
                entry["val_quota"] += 1
                current_val += 1
            idx += 1

    elif current_val > val_size:
        candidates = [entry for entry in per_country if entry["val_quota"] > 0]
        candidates.sort(
            key=lambda entry: (
                entry["remainder"],
                entry["count"],
                entry["country"],
            )
        )
        idx = 0
        while current_val > val_size:
            entry = candidates[idx % len(candidates)]
            if entry["val_quota"] > 0:
                entry["val_quota"] -= 1
                current_val -= 1
            idx += 1

    train_idx: list[int] = []
    val_idx: list[int] = []
    for entry in per_country:
        val_quota = entry["val_quota"]
        idxs = entry["indices"]
        val_idx.extend(idxs[:val_quota].tolist())
        train_idx.extend(idxs[val_quota:].tolist())

    if len(val_idx) != val_size or len(train_idx) < train_size:
        raise RuntimeError("Unable to create the requested train/val split from sampled metadata.")

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx[:train_size], val_idx[:val_size]


def build_datasets(args):
    max_train_shards = args.max_shards if args.max_shards > 0 else None
    raw_test_shards = args.max_test_shards if args.max_test_shards is not None else args.max_shards
    max_test_shards = raw_test_shards if raw_test_shards > 0 else None

    train_pool = OSV5MDataset(
        split="train",
        subset_size=args.train_size + args.val_size,
        seed=args.seed,
        cache_dir=args.cache_dir,
        extract_dir=args.extract_dir,
        max_shards=max_train_shards,
    )
    test_ds = OSV5MDataset(
        split="test",
        subset_size=args.test_size,
        seed=args.seed,
        cache_dir=args.cache_dir,
        extract_dir=args.extract_dir,
        max_shards=max_test_shards,
    )

    print("\n--- Downloading and indexing OSV-5M train split ---")
    t0 = time.time()
    train_pool.download_images()
    print(f"Train split ready in {time.time() - t0:.1f}s")

    print("\n--- Downloading and indexing OSV-5M test split ---")
    t0 = time.time()
    test_ds.download_images()
    print(f"Test split ready in {time.time() - t0:.1f}s")

    all_countries = sorted(
        set(train_pool.metadata["country"].dropna().unique())
        | set(test_ds.metadata["country"].dropna().unique())
    )
    mapping = {country: i for i, country in enumerate(all_countries)}
    reverse = {i: country for country, i in mapping.items()}
    for ds in (train_pool, test_ds):
        ds.country_to_label = mapping
        ds.label_to_country = reverse

    train_idx, val_idx = stratified_split_indices(
        train_pool.metadata,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    train_ds = Subset(train_pool, train_idx)
    val_ds = Subset(train_pool, val_idx)
    return train_ds, val_ds, test_ds, all_countries


def build_loaders(train_ds, val_ds, test_ds, preprocess, batch_size: int, num_workers: int):
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    return {
        "train": DataLoader(_PreprocessedDataset(train_ds, preprocess), shuffle=True, drop_last=False, **loader_kwargs),
        "val": DataLoader(_PreprocessedDataset(val_ds, preprocess), shuffle=False, drop_last=False, **loader_kwargs),
        "test": DataLoader(_PreprocessedDataset(test_ds, preprocess), shuffle=False, drop_last=False, **loader_kwargs),
    }


def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int):
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, loader, device, amp_enabled: bool):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = compute_metrics(logits, labels)
    metrics["loss"] = total_loss / max(1, total_examples)
    return metrics


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int, best_val_top1: float, args) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_val_top1": best_val_top1,
            "config": vars(args),
        },
        path,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device)
    amp_enabled = (not args.disable_amp) and device.type == "cuda"
    system_info = get_system_info(device)
    run_start = time.time()
    output_dir = Path(args.output_dir)

    print(f"Device: {device}")
    print(f"AMP enabled: {amp_enabled}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output directory: {output_dir}")

    data_start = time.time()
    train_ds, val_ds, test_ds, all_countries = build_datasets(args)
    data_time = time.time() - data_start
    num_classes = len(all_countries)
    print(
        f"Countries: {num_classes}, Train: {len(train_ds)}, "
        f"Val: {len(val_ds)}, Test: {len(test_ds)}"
    )

    print("\n--- Loading CLIP ViT-L/14 ---")
    model = GeolocationCLIP(
        num_classes=num_classes,
        class_names=all_countries,
        mode="full_finetune",
    ).to(device)
    model_time = time.time() - run_start - data_time
    param_counts = model.parameter_counts()
    print(
        f"Parameters - total: {param_counts['total']:,}, "
        f"trainable: {param_counts['trainable']:,}"
    )
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    save_json(trainable_names, output_dir / "trainable_parameters.json")

    loaders = build_loaders(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        preprocess=model.preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.optimizer_param_groups(
            encoder_lr=args.encoder_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
        )
    )

    updates_per_epoch = math.ceil(len(loaders["train"]) / args.grad_accum_steps)
    total_steps = max(1, args.epochs * updates_per_epoch)
    warmup_steps = min(total_steps - 1, int(total_steps * args.warmup_ratio)) if total_steps > 1 else 0
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
    scaler = GradScaler(device=device.type, enabled=amp_enabled)

    training_log = []
    best_val_top1 = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    output_dir = Path(args.output_dir)
    train_start = time.time()

    print("\n--- Full fine-tuning ---")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        running_correct = 0
        running_examples = 0
        progress = tqdm(loaders["train"], desc=f"Epoch {epoch}/{args.epochs}")

        for step_idx, (images, labels) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = F.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=args.label_smoothing,
                )
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size * args.grad_accum_steps
            running_correct += (logits.detach().argmax(dim=1) == labels).sum().item()
            running_examples += batch_size

            if step_idx % args.grad_accum_steps == 0 or step_idx == len(loaders["train"]):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            progress.set_postfix(
                loss=f"{running_loss / max(1, running_examples):.4f}",
                acc=f"{running_correct / max(1, running_examples):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        train_loss = running_loss / max(1, running_examples)
        train_acc = running_correct / max(1, running_examples)

        val_metrics = evaluate(model, loaders["val"], device, amp_enabled)
        entry = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1_acc"],
            "val_top5": val_metrics.get("top5_acc"),
            "val_top10": val_metrics.get("top10_acc"),
            "val_mean_class_acc": val_metrics["mean_class_acc"],
            "encoder_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[2]["lr"],
        }
        training_log.append(entry)

        print(
            f"  Epoch {epoch}/{args.epochs} - train loss: {train_loss:.4f}, "
            f"train acc: {train_acc:.4f}, val top-1: {val_metrics['top1_acc']:.4f}, "
            f"val top-5: {val_metrics.get('top5_acc', 'N/A')}"
        )

        save_checkpoint(
            output_dir / "last_checkpoint.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_val_top1,
            args,
        )

        if val_metrics["top1_acc"] > best_val_top1:
            best_val_top1 = val_metrics["top1_acc"]
            best_epoch = epoch
            epochs_without_improvement = 0
            if args.save_best:
                save_checkpoint(
                    output_dir / "best_checkpoint.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_val_top1,
                    args,
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    train_time = time.time() - train_start

    if not (output_dir / "best_checkpoint.pt").exists():
        save_checkpoint(
            output_dir / "best_checkpoint.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            best_epoch if best_epoch > 0 else len(training_log),
            best_val_top1,
            args,
        )

    print("\n--- Loading best checkpoint for final evaluation ---")
    eval_start = time.time()
    best_checkpoint = torch.load(output_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_metrics = evaluate(model, loaders["test"], device, amp_enabled)
    eval_time = time.time() - eval_start
    total_time = time.time() - run_start
    per_class_test = test_metrics.pop("per_class_acc")

    save_json(training_log, output_dir / "training_log.json")
    save_json(test_metrics, output_dir / "full_finetune_metrics.json")
    save_json({str(k): v for k, v in per_class_test.items()}, output_dir / "full_finetune_per_class.json")

    summary = {
        "method": "Full fine-tuning",
        "model": "OpenCLIP ViT-L/14",
        "pretraining": "LAION-2B",
        "task": "country-level geolocation classification",
        "config": vars(args),
        "system_info": system_info,
        "num_classes": num_classes,
        "num_countries": all_countries,
        "parameter_counts": param_counts,
        "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "test_metrics": test_metrics,
        "dataset": {
            "source": "OSV-5M",
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
            "num_classes": num_classes,
            "class_names": all_countries,
            "split_seed": args.seed,
        },
        "timing": {
            "data_loading_sec": data_time,
            "model_loading_sec": model_time,
            "training_sec": train_time,
            "final_eval_sec": eval_time,
            "total_run_sec": total_time,
            "training_min": train_time / 60,
            "total_run_min": total_time / 60,
        },
        "best_epoch_log": max(training_log, key=lambda x: x["val_top1"]) if training_log else None,
    }
    save_json(summary, output_dir / "summary.json")

    print("\n" + "=" * 50)
    print("FULL FINETUNE RESULTS SUMMARY")
    print("=" * 50)
    print(f"Best validation epoch: {best_epoch}")
    print(f"Test top-1: {test_metrics['top1_acc']:.4f}")
    print(f"Test top-5: {test_metrics.get('top5_acc', 'N/A')}")
    print(f"Test top-10: {test_metrics.get('top10_acc', 'N/A')}")
    print(f"Test mean class acc: {test_metrics['mean_class_acc']:.4f}")
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
