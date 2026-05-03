# geolocation-domain-shift

Finetuning VLMs for Geolocation under Domain Shift

## Model

ViT-L/14 LAION-2B via [open_clip](https://github.com/mlfoundations/open_clip)

## Adaptation Strategies

**Zero-shot**: image → embedding, text (country name) → embedding, predict by cosine similarity (Top-1, Top-5, etc.)

**Linear probe**: freeze CLIP encoder, train a linear classifier on top

**Full fine-tuning**: unfreeze and fine-tune the full image encoder + classification head

**LoRA**: insert low-rank adapters into transformer layers, train only those + classification head

## Data

OpenStreetView-5M (OSV-5M) — 5.1M geo-referenced street-view images across 225 countries

## Running Baseline Experiments

### Setup for GPU instance

```bash
# On Lambda Labs / RunPod / Vast.ai (A100 or L4)
git clone https://github.com/kravishankar30/geolocation-domain-shift && cd geolocation-domain-shift
git checkout geoclip_model
pip install -e .
```

### Run baseline (zero-shot + linear probe)

```bash
# Small run for quick validation (~20 min on A100)
python scripts/run_baseline.py --train_size 10000 --test_size 2000 --epochs 10 --batch_size 256

# Larger run for final paper results (~1-2 hrs on A100)
python scripts/run_baseline.py --train_size 50000 --test_size 10000 --epochs 20 --batch_size 256
```

### Run full fine-tuning

```bash
# Validation-sized cluster smoke test
python scripts/run_full_finetune.py \
  --train_size 50000 \
  --val_size 5000 \
  --test_size 10000 \
  --epochs 10 \
  --batch_size 32 \
  --encoder_lr 1e-6 \
  --head_lr 1e-4 \
  --weight_decay 1e-4 \
  --patience 3 \
  --save_best

# Larger cluster run
python scripts/run_full_finetune.py \
  --train_size 250000 \
  --val_size 25000 \
  --test_size 25000 \
  --epochs 15 \
  --batch_size 64 \
  --max_shards 0 \
  --max_test_shards 0
```

Notes:
- The script uses the official OSV-5M `train` split for training/validation and the official `test` split for final evaluation.
- Validation Top-1 is used for early stopping.
- The encoder and head use separate learning rates to keep backbone updates conservative.
- The script now mirrors the LoRA branch's cluster-facing outputs: timing metadata, system info, and `trainable_parameters.json`.
- `--disable_amp` is available if your cluster environment has AMP compatibility issues.

### Run full fine-tuning via Slurm

```bash
sbatch scripts/run_full_finetune.slurm
```

### Generate figures

```bash
python scripts/visualize_results.py --results_dir results/baseline
# Outputs: results/baseline/figures/*.png
```

```bash
python scripts/visualize_full_finetune.py --results_dir results/full_finetune
# Outputs: results/full_finetune/figures/*.png
```

```bash
python scripts/run_full_finetune_domain_shift.py --full_finetune_dir results/full_finetune
python scripts/visualize_full_finetune_domain_shift.py --results_dir results/full_finetune_domain_shift
# Outputs: results/full_finetune_domain_shift/figures/*.png
```

### Output structure

```
results/baseline/
├── summary.json                 # Combined results
├── zero_shot_metrics.json       # Top-1/5/10, mean class accuracy
├── linear_probe_metrics.json
├── training_log.json            # Per-epoch loss/accuracy
├── embeddings/                  # Cached (reuse across runs)
└── figures/
    ├── comparison_bar.png       # ZS vs LP side-by-side
    ├── training_curve.png       # Loss + accuracy over epochs
    ├── per_class_top.png        # Best countries
    ├── per_class_bottom.png     # Worst countries
    └── accuracy_distribution.png
```

```
results/full_finetune/
├── best_checkpoint.pt           # Best checkpoint by validation Top-1
├── last_checkpoint.pt           # Latest checkpoint
├── full_finetune_metrics.json   # Final clean test metrics
├── full_finetune_per_class.json
├── training_log.json            # Per-epoch train/validation metrics
├── trainable_parameters.json
├── figures/
│   ├── metrics_bar.png
│   ├── training_curve.png
│   ├── lr_curve.png
│   ├── per_class_top.png
│   ├── per_class_bottom.png
│   └── accuracy_distribution.png
└── summary.json
```

```
results/full_finetune_domain_shift/
├── comparison.json
├── summary.json
├── gaussian_blur/
│   ├── full_finetune_metrics.json
│   └── full_finetune_per_class.json
├── brightness/
│   ├── full_finetune_metrics.json
│   └── full_finetune_per_class.json
├── occlusion/
│   ├── full_finetune_metrics.json
│   └── full_finetune_per_class.json
└── figures/
    ├── domain_shift_comparison.png
    └── degradation_chart.png
```
