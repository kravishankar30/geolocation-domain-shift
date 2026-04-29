#!/usr/bin/env python3
"""
Train OpenCLIP ViT-L/14 for geolocation using LoRA.

This script:
1. Loads an OSV-5M subset.
2. Reproduces the same train/test split style as run_baseline.py.
3. Trains only LoRA adapter parameters + classification head.
4. Tracks metrics useful for the final report:
   - Top-1 / Top-5 / Top-10 accuracy
   - Mean class accuracy
   - Per-class accuracy
   - Trainable vs total parameter counts
   - Dataset sizes and number of classes
   - Training time and epoch time
   - Hyperparameters and device info
5. Saves checkpoint, logs, metrics, and summary files.

Example run:
    python scripts/run_lora.py \
        --train_size 10000 \
        --test_size 2000 \
        --epochs 5 \
        --batch_size 32 \
        --lr 1e-4 \
        --output_dir results/lora
"""

import argparse
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.osv5m_dataset import OSV5MDataset
from src.models.clip_geolocation import GeolocationCLIP


class PreprocessedDataset(torch.utils.data.Dataset):
    """
    Wraps an OSV5MDataset or Subset and applies CLIP preprocessing.

    The raw dataset returns PIL images. CLIP expects normalized tensors, so this
    wrapper applies the model's preprocess transform before returning each image.
    """

    def __init__(self, dataset, preprocess):
        self.dataset = dataset
        self.preprocess = preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.preprocess(sample["image"])
        label = sample["label"]
        return image, label


def parse_args():
    """
    Parse command-line arguments for LoRA training.

    These hyperparameters are saved in summary.json so the final report can
    describe the training setup reproducibly.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_size", type=int, default=10_000)
    parser.add_argument("--test_size", type=int, default=2_000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument("--output_dir", type=str, default="results/lora")
    parser.add_argument("--cache_dir", type=str, default="./data/osv5m_cache")
    parser.add_argument("--extract_dir", type=str, default="./data/osv5m_images")
    parser.add_argument("--max_shards", type=int, default=1)

    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--save_best",
        action="store_true",
        help="Save best checkpoint by test Top-1 accuracy.",
    )

    return parser.parse_args()


def set_seed(seed):
    """
    Set random seeds for reproducibility.

    This helps ensure that dataset splitting and training behavior are more
    stable across runs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_system_info(device):
    """
    Collect system information useful for the report and debugging.

    Returns:
        dict with Python, PyTorch, CUDA, GPU, and device details.
    """
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


def compute_metrics(logits, labels, topk=(1, 5, 10)):
    """
    Compute standard geolocation classification metrics.

    Args:
        logits: Tensor of shape (N, C), model predictions before softmax.
        labels: Tensor of shape (N,), ground-truth class indices.
        topk: Tuple of k values for Top-k accuracy.

    Returns:
        Dictionary containing Top-k accuracy, per-class accuracy, and mean
        class accuracy.
    """
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
        per_class[int(c)] = (preds[mask] == c).float().mean().item()

    metrics["per_class_acc"] = per_class
    metrics["mean_class_acc"] = float(np.mean(list(per_class.values())))

    return metrics


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate the model on a DataLoader.

    Args:
        model: GeolocationCLIP model.
        loader: DataLoader returning preprocessed image tensors and labels.
        device: torch.device.

    Returns:
        Metrics dictionary from compute_metrics().
    """
    model.eval()

    all_logits = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)

        logits = model(images).cpu()

        all_logits.append(logits)
        all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    return compute_metrics(logits, labels)


def train_one_epoch(model, train_loader, optimizer, device, epoch, total_epochs):
    """
    Train the model for one epoch.

    Only parameters returned by model.trainable_parameters() are optimized,
    which should be LoRA parameters plus the classification head.

    Returns:
        Dictionary with train loss, train accuracy, and epoch time.
    """
    model.train()

    epoch_start = time.time()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    epoch_time = time.time() - epoch_start

    return {
        "train_loss": total_loss / total,
        "train_acc": correct / total,
        "epoch_time_sec": epoch_time,
    }


def save_json(obj, path):
    """
    Save a Python object as pretty-formatted JSON.
    """
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_checkpoint(model, output_dir, filename, metadata):
    """
    Save model checkpoint with metadata.

    This checkpoint can later be used for domain shift evaluation.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        output_dir / filename,
    )


def main():
    """
    Main training entry point.

    This function loads data, builds the LoRA model, trains it, evaluates it,
    and saves all report-relevant outputs.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    system_info = get_system_info(device)

    print("\n" + "=" * 70)
    print("LoRA Geolocation Training")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output directory: {output_dir}")

    run_start = time.time()

    # ------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------
    print("\n--- Loading OSV-5M subset ---")
    data_start = time.time()

    max_shards = args.max_shards if args.max_shards > 0 else None
    total_size = args.train_size + args.test_size

    full_ds = OSV5MDataset(
        "train",
        total_size,
        args.seed,
        args.cache_dir,
        args.extract_dir,
        max_shards=max_shards,
    )

    full_ds.download_images()

    data_time = time.time() - data_start

    # Same split style as baseline script: randomly permute one pool.
    n = len(full_ds)
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(n)

    n_test = min(args.test_size, n // 5)
    test_idx = indices[:n_test]
    train_idx = indices[n_test : n_test + args.train_size]

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    test_ds = torch.utils.data.Subset(full_ds, test_idx)

    num_classes = full_ds.num_classes
    all_countries = [full_ds.label_to_country[i] for i in range(num_classes)]

    print(f"Dataset loading/downloading time: {data_time:.1f}s")
    print(f"Total available samples: {n}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Number of countries/classes: {num_classes}")

    # ------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------
    print("\n--- Loading OpenCLIP ViT-L/14 with LoRA ---")
    model_start = time.time()

    model = GeolocationCLIP(
        num_classes=num_classes,
        class_names=all_countries,
        mode="lora",
    ).to(device)

    model_time = time.time() - model_start

    parameter_counts = model.parameter_counts()

    trainable_names = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]

    print(f"Model loading time: {model_time:.1f}s")
    print("Parameter counts:")
    print(parameter_counts)
    print(f"Number of trainable tensors: {len(trainable_names)}")
    print("First trainable tensors:")
    for name in trainable_names[:20]:
        print(f"  {name}")

    # Save trainable parameter names for report/debugging.
    save_json(trainable_names, output_dir / "trainable_parameters.json")

    # ------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------
    train_loader = DataLoader(
        PreprocessedDataset(train_ds, model.preprocess),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        PreprocessedDataset(test_ds, model.preprocess),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # ------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    print("\n--- Training LoRA ---")

    training_log = []
    best_top1 = -1.0

    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        test_metrics = evaluate(model, test_loader, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        entry = {
            "epoch": epoch,
            "train_loss": epoch_stats["train_loss"],
            "train_acc": epoch_stats["train_acc"],
            "test_top1": test_metrics["top1_acc"],
            "test_top5": test_metrics.get("top5_acc"),
            "test_top10": test_metrics.get("top10_acc"),
            "test_mean_class_acc": test_metrics["mean_class_acc"],
            "lr": current_lr,
            "epoch_time_sec": epoch_stats["epoch_time_sec"],
        }

        training_log.append(entry)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss={entry['train_loss']:.4f} | "
            f"train_acc={entry['train_acc']:.4f} | "
            f"test_top1={entry['test_top1']:.4f} | "
            f"test_top5={entry['test_top5']:.4f} | "
            f"time={entry['epoch_time_sec']:.1f}s | "
            f"lr={entry['lr']:.2e}"
        )

        if args.save_best and entry["test_top1"] > best_top1:
            best_top1 = entry["test_top1"]
            save_checkpoint(
                model=model,
                output_dir=output_dir,
                filename="lora_best_checkpoint.pt",
                metadata={
                    "epoch": epoch,
                    "best_top1": best_top1,
                    "num_classes": num_classes,
                    "class_names": all_countries,
                    "config": vars(args),
                },
            )

    train_time = time.time() - train_start

    # ------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------
    print("\n--- Final Evaluation ---")
    eval_start = time.time()

    final_metrics = evaluate(model, test_loader, device)

    eval_time = time.time() - eval_start
    total_time = time.time() - run_start

    per_class = final_metrics.pop("per_class_acc")

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    save_json(final_metrics, output_dir / "lora_metrics.json")
    save_json({str(k): v for k, v in per_class.items()}, output_dir / "lora_per_class.json")
    save_json(training_log, output_dir / "lora_training_log.json")

    summary = {
        "method": "LoRA",
        "model": "OpenCLIP ViT-L/14",
        "pretraining": "LAION-2B",
        "task": "country-level geolocation classification",
        "config": vars(args),
        "system_info": system_info,
        "dataset": {
            "source": "OSV-5M",
            "total_available_samples": n,
            "train_samples": len(train_ds),
            "test_samples": len(test_ds),
            "num_classes": num_classes,
            "class_names": all_countries,
            "split_seed": args.seed,
        },
        "parameter_counts": parameter_counts,
        "timing": {
            "data_loading_sec": data_time,
            "model_loading_sec": model_time,
            "training_sec": train_time,
            "final_eval_sec": eval_time,
            "total_run_sec": total_time,
            "training_min": train_time / 60,
            "total_run_min": total_time / 60,
        },
        "final_metrics": final_metrics,
        "best_epoch_by_test_top1": max(training_log, key=lambda x: x["test_top1"]),
    }

    save_json(summary, output_dir / "summary.json")

    save_checkpoint(
        model=model,
        output_dir=output_dir,
        filename="lora_checkpoint.pt",
        metadata={
            "num_classes": num_classes,
            "class_names": all_countries,
            "config": vars(args),
            "final_metrics": final_metrics,
            "parameter_counts": parameter_counts,
        },
    )

    # ------------------------------------------------------------
    # Print final summary
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LORA RESULTS SUMMARY")
    print("=" * 70)

    for key, value in final_metrics.items():
        print(f"{key:<25} {value:.4f}")

    print("\nTiming:")
    print(f"Data loading:       {data_time / 60:.2f} min")
    print(f"Training:           {train_time / 60:.2f} min")
    print(f"Final evaluation:   {eval_time / 60:.2f} min")
    print(f"Total run:          {total_time / 60:.2f} min")

    print("\nParameter counts:")
    for key, value in parameter_counts.items():
        print(f"{key:<25} {value}")

    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()