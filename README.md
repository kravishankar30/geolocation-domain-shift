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

### Generate figures

```bash
python scripts/visualize_results.py --results_dir results/baseline
# Outputs: results/baseline/figures/*.png
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
