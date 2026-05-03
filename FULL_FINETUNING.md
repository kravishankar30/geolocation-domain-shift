# Full Fine-Tuning Guide

This document explains the full fine-tuning implementation in this repo in enough detail that you can:

- understand what the training script is doing
- trace how data and model outputs flow through the code
- know which hyperparameters to change when training does not behave as expected
- debug common failure modes on a GPU cluster

This guide covers the current implementation in:

- [scripts/run_full_finetune.py](/Users/ayushbaweja/Downloads/geolocation-domain-shift/scripts/run_full_finetune.py:1)
- [scripts/run_full_finetune.slurm](/Users/ayushbaweja/Downloads/geolocation-domain-shift/scripts/run_full_finetune.slurm:1)
- [src/models/clip_geolocation.py](/Users/ayushbaweja/Downloads/geolocation-domain-shift/src/models/clip_geolocation.py:1)
- [src/data/osv5m_dataset.py](/Users/ayushbaweja/Downloads/geolocation-domain-shift/src/data/osv5m_dataset.py:1)

## Goal

Full fine-tuning means training:

- the CLIP image encoder
- the classification head on top of it

end to end for country classification on OSV-5M.

This is different from linear probing:

- linear probe freezes the encoder and only trains the classifier head
- full fine-tuning updates the pretrained backbone itself

That usually gives more task adaptation, but it also increases:

- GPU memory usage
- training time
- overfitting risk
- sensitivity to learning-rate mistakes

## High-Level Training Flow

The script in [scripts/run_full_finetune.py](/Users/ayushbaweja/Downloads/geolocation-domain-shift/scripts/run_full_finetune.py:1) does the following:

1. Parse command-line arguments.
2. Set random seeds.
3. Download and index a subset of OSV-5M train and test images.
4. Build a unified country label mapping across train and test.
5. Split the train subset into train and validation sets.
6. Load OpenCLIP ViT-L/14 LAION-2B and switch it to `full_finetune` mode.
7. Create dataloaders that preprocess PIL images using the CLIP transform.
8. Build an AdamW optimizer with separate learning rates for encoder and head.
9. Train with mixed precision, gradient accumulation, warmup, cosine LR decay, and early stopping.
10. Reload the best checkpoint based on validation Top-1 accuracy.
11. Evaluate on the official OSV-5M test subset.
12. Save metrics, per-class metrics, checkpoints, and a summary JSON.

The script is now also aligned with the `origin/lora` branch operationally, so it saves the same kind of run metadata that is useful on a shared GPU cluster:

- system info
- timing breakdowns
- trainable parameter names
- a matching Slurm launcher script

## Model Structure

The model class is [GeolocationCLIP](/Users/ayushbaweja/Downloads/geolocation-domain-shift/src/models/clip_geolocation.py:30).

It wraps:

- `self.clip`: the pretrained OpenCLIP model
- `self.head`: a linear classifier from CLIP image embedding dimension to number of countries

For this checkpoint:

- model: `ViT-L-14`
- pretrained weights: `laion2b_s32b_b82k`
- embedding dimension: `768`

The forward pass for full fine-tuning is simple:

1. preprocess image
2. encode image with CLIP image encoder
3. pass image embedding into linear head
4. get logits over countries

In code, the full fine-tuning path is:

- [src/models/clip_geolocation.py](/Users/ayushbaweja/Downloads/geolocation-domain-shift/src/models/clip_geolocation.py:138)

For non-zero-shot modes:

```python
img_embs = self.encode_image(images)
return self.head(img_embs)
```

## What `full_finetune` Mode Changes

Mode control happens in:

- [src/models/clip_geolocation.py](/Users/ayushbaweja/Downloads/geolocation-domain-shift/src/models/clip_geolocation.py:82)

When mode is `full_finetune`:

- all CLIP encoder parameters have `requires_grad=True`
- all head parameters have `requires_grad=True`

That is the key difference from `linear_probe`, where only the head is trainable.

## Dataset and Label Pipeline

The dataset class is:

- [src/data/osv5m_dataset.py](/Users/ayushbaweja/Downloads/geolocation-domain-shift/src/data/osv5m_dataset.py:32)

It does several things:

### 1. Metadata download

It downloads `train.csv` or `test.csv` from the Hugging Face dataset repo.

### 2. Image shard selection

OSV-5M images are sharded zip files. The dataset can download:

- only some train shards
- only some test shards

This is controlled by:

- `--max_train_shards`
- `--max_test_shards`

Special behavior:

- `0` means use all available shards

### 3. Disk scan

After extraction, the dataset scans extracted shard folders and builds a map from image ID to file path.

### 4. Metadata filtering

It keeps only rows from the CSV whose image files actually exist on disk.

### 5. Sampling

It samples a subset of examples up to `subset_size` with approximate country balancing.

### 6. Output format

Each item returns:

- `image`: PIL image
- `label`: integer class ID
- `latitude`
- `longitude`
- `country`

## Why the Script Uses Train and Test Separately

The baseline script rebuilt a train/test split from the train pool. The full fine-tuning script is stricter:

- it uses the official `train` split for training and validation
- it uses the official `test` split for final evaluation

That is closer to the intended dataset setup and cleaner for a real fine-tuning run.

## Train/Validation Split

Inside [scripts/run_full_finetune.py](/Users/ayushbaweja/Downloads/geolocation-domain-shift/scripts/run_full_finetune.py:102), the function `stratified_split_indices(...)` builds the train/validation split.

The goal is to preserve country proportions as closely as possible while creating:

- `train_size` examples for training
- `val_size` examples for validation

Important detail:

- validation examples come from the training subset, not from the official test split

This is required for:

- early stopping
- hyperparameter tuning without touching the final test set

This is the main methodological difference from the LoRA branch:

- LoRA evaluates directly on its held-out split each epoch
- full fine-tuning keeps a separate validation split because early stopping matters more when the full backbone is trainable

## Country Label Mapping

One subtle issue in geolocation classification is that train and test can have different country coverage in a sampled subset.

The script resolves this by building the class mapping from the union of countries seen in:

- sampled train metadata
- sampled test metadata

That logic is in:

- [scripts/run_full_finetune.py](/Users/ayushbaweja/Downloads/geolocation-domain-shift/scripts/run_full_finetune.py:194)

This ensures:

- train labels and test labels use the same integer mapping
- evaluation logits line up with the same output class ordering

## Preprocessing and Dataloaders

The model exposes `self.preprocess`, which comes from OpenCLIP.

That transform is applied in `_PreprocessedDataset`:

- [scripts/run_full_finetune.py](/Users/ayushbaweja/Downloads/geolocation-domain-shift/scripts/run_full_finetune.py:60)

This wrapper converts the dataset output from:

- raw PIL images

to:

- model-ready CLIP input tensors

The dataloaders are built in:

- [scripts/run_full_finetune.py](/Users/ayushbaweja/Downloads/geolocation-domain-shift/scripts/run_full_finetune.py:216)

The script creates:

- train loader
- validation loader
- test loader

Key dataloader settings:

- `batch_size`
- `num_workers`
- `pin_memory=True`
- `persistent_workers=True` when workers are enabled

## Loss Function

The training objective is standard multi-class cross-entropy:

```python
loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
```

This matches the PDF-level method description.

Optional regularization:

- `label_smoothing`

Default:

- `0.0`

If training becomes too confident or overfits early, small label smoothing such as `0.05` or `0.1` can help.

## Optimizer Design

The optimizer is AdamW.

The parameter groups are created by:

- [src/models/clip_geolocation.py](/Users/ayushbaweja/Downloads/geolocation-domain-shift/src/models/clip_geolocation.py:169)

Why separate parameter groups?

Because the pretrained encoder should usually move much more slowly than the randomly initialized classification head.

Current parameter grouping:

1. encoder parameters with weight decay
2. encoder parameters without weight decay
3. head parameters with weight decay
4. head parameters without weight decay

Biases and 1D parameters are excluded from weight decay.

This is standard practice because things like:

- bias vectors
- LayerNorm scales

usually should not be decayed.

### Current learning-rate split

Defaults:

- `encoder_lr = 1e-6`
- `head_lr = 1e-4`

This is intentionally conservative.

Reason:

- the head is new and needs larger updates
- the encoder is pretrained and easy to destabilize

## Learning-Rate Schedule

The scheduler is created in:

- [scripts/run_full_finetune.py](/Users/ayushbaweja/Downloads/geolocation-domain-shift/scripts/run_full_finetune.py:230)

It uses:

- linear warmup
- cosine decay

Warmup is helpful because full fine-tuning large pretrained models can diverge early if the first updates are too large.

Relevant arguments:

- `--warmup_ratio`
- `--epochs`
- `--grad_accum_steps`

The total number of optimizer steps is:

```text
epochs * ceil(num_train_batches / grad_accum_steps)
```

Warmup steps are:

```text
total_steps * warmup_ratio
```

## Mixed Precision

By default, AMP is enabled on CUDA:

- [scripts/run_full_finetune.py](/Users/ayushbaweja/Downloads/geolocation-domain-shift/scripts/run_full_finetune.py:278)

This reduces:

- memory usage
- training time

The script uses:

- `autocast(...)`
- `GradScaler(...)`

You can disable it with:

```bash
--disable_amp
```

Use that if:

- your cluster setup has AMP incompatibilities
- you suspect numerical instability that only appears in mixed precision

## Gradient Accumulation

Full fine-tuning ViT-L/14 can exceed GPU memory if the batch size is too high.

To handle that, the script supports:

- `--grad_accum_steps`

This lets you simulate a larger effective batch size by accumulating gradients across multiple forward/backward passes before stepping the optimizer.

Effective batch size:

```text
batch_size * grad_accum_steps
```

Example:

- `batch_size=16`
- `grad_accum_steps=4`

acts roughly like an optimizer batch size of `64`, while keeping per-step memory lower.

## Early Stopping

Early stopping is based on validation Top-1 accuracy.

Relevant args:

- `--patience`
- `--epochs`

Behavior:

- if validation Top-1 improves, save `best_checkpoint.pt`
- if it does not improve for `patience` consecutive epochs, stop training early

This is a practical safeguard against overfitting and wasted cluster time.

## Checkpoints and Output Files

The script writes to:

- `results/full_finetune/` by default

Files:

- `best_checkpoint.pt`: best validation checkpoint
- `last_checkpoint.pt`: latest training state
- `full_finetune_metrics.json`: final test metrics
- `full_finetune_per_class.json`: per-class test accuracy
- `training_log.json`: per-epoch train and validation logs
- `trainable_parameters.json`: names of trainable tensors for auditing/debugging
- `summary.json`: config, parameter counts, best epoch, and final metrics

The `summary.json` file also includes:

- system information
- timing information
- dataset sizes
- best epoch log entry

To generate figures after the run:

```bash
python scripts/visualize_full_finetune.py --results_dir results/full_finetune
```

That script creates:

- `figures/metrics_bar.png`
- `figures/training_curve.png`
- `figures/lr_curve.png`
- `figures/per_class_top.png`
- `figures/per_class_bottom.png`
- `figures/accuracy_distribution.png`

To run domain-shift evaluation on the saved best checkpoint:

```bash
python scripts/run_full_finetune_domain_shift.py --full_finetune_dir results/full_finetune
python scripts/visualize_full_finetune_domain_shift.py --results_dir results/full_finetune_domain_shift
```

That evaluation:

- reloads `best_checkpoint.pt`
- rebuilds the same official test subset using the saved run config
- evaluates the checkpoint under `gaussian_blur`, `brightness`, and `occlusion`
- saves a `comparison.json` plus two summary figures

## Important Command-Line Arguments

This section is the practical part you will likely care about most when running on a cluster.

### Data size arguments

#### `--train_size`

How many examples to use for actual training.

Increase this if:

- training underfits
- validation accuracy is noisy because the train set is too small

Decrease this if:

- runs are too expensive
- you are doing a smoke test

#### `--val_size`

How many training-split examples to reserve for validation.

Increase this if:

- validation metrics are too noisy
- early stopping decisions look unstable

Decrease this if:

- you want more data in training
- total available examples from selected shards are limited

#### `--test_size`

How many official test examples to evaluate on.

This does not affect training, only final evaluation cost and metric stability.

Use smaller values for quick debugging. Use larger values for final reporting.

#### `--max_shards` and `--max_test_shards`

How many OSV-5M zip shards to download.

`--max_shards` now matches the LoRA branch interface and controls train shard download count.

`--max_test_shards` is optional:

- if omitted, it defaults to the same value as `--max_shards`
- if set to `0`, all test shards are used

Use:

- `1` for very small debug runs
- a few shards for development
- `0` for all shards on a real cluster run

If your requested `train_size + val_size` is larger than what the selected shards actually contain, the script will fail. In that case:

- increase shard count
- or reduce requested sizes

### Optimization arguments

#### `--encoder_lr`

Learning rate for the pretrained CLIP encoder.

This is the most sensitive parameter in full fine-tuning.

If training is unstable:

- lower it to `5e-7` or `1e-7`

If training is very stable but barely improves over linear probing:

- try `2e-6` or `5e-6`

Be careful. Too high an encoder LR can destroy pretrained features quickly.

#### `--head_lr`

Learning rate for the classifier head.

If training is slow to start:

- increase it to `2e-4` or `5e-4`

If loss oscillates badly:

- reduce it to `5e-5`

Usually this can be much higher than `encoder_lr`.

#### `--weight_decay`

L2-style regularization through AdamW.

If overfitting is strong:

- increase to `5e-4` or `1e-3`

If optimization looks too constrained:

- reduce to `1e-5`

#### `--warmup_ratio`

Fraction of total training steps used for LR warmup.

If the first epoch is unstable:

- increase to `0.15` or `0.2`

If warmup feels too slow:

- reduce to `0.05`

#### `--label_smoothing`

Regularization on the classification targets.

If the model becomes overconfident:

- try `0.05`
- or `0.1`

If you want pure hard-label cross-entropy:

- keep `0.0`

### Memory and throughput arguments

#### `--batch_size`

Per-device batch size.

If you get CUDA OOM:

- reduce this first

If GPU utilization is low and memory headroom exists:

- increase it

#### `--grad_accum_steps`

Number of mini-batches to accumulate before optimizer step.

If you need a larger effective batch size but cannot fit it in memory:

- increase this

Tradeoff:

- larger accumulation lowers step frequency
- wall-clock time may increase

#### `--num_workers`

DataLoader worker processes.

If GPUs are starving waiting for data:

- increase this

If your cluster filesystem is slow or unstable:

- reducing workers can sometimes help

### Training-control arguments

#### `--epochs`

Maximum number of epochs.

If training is still improving at the end:

- increase this

If early stopping is working, this is just an upper bound.

#### `--patience`

How many non-improving validation epochs to tolerate before stopping.

If validation is noisy:

- increase patience

If you want aggressive stopping:

- reduce patience

#### `--seed`

Controls:

- shard selection
- sampling
- train/val split
- RNG for training

Change this if you want to verify that results are not seed-specific.

#### `--output_dir`

Where outputs are written.

Useful on a cluster when you want:

- separate experiment folders
- job-specific output directories

## How to Run It

Example smoke test:

```bash
python scripts/run_full_finetune.py \
  --train_size 50000 \
  --val_size 5000 \
  --test_size 10000 \
  --batch_size 32 \
  --epochs 10 \
  --encoder_lr 1e-6 \
  --head_lr 1e-4 \
  --weight_decay 1e-4 \
  --patience 3
```

Larger run:

```bash
python scripts/run_full_finetune.py \
  --train_size 250000 \
  --val_size 25000 \
  --test_size 25000 \
  --batch_size 64 \
  --epochs 15 \
  --max_shards 0 \
  --max_test_shards 0
```

Memory-constrained variant:

```bash
python scripts/run_full_finetune.py \
  --train_size 50000 \
  --val_size 5000 \
  --test_size 10000 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --epochs 10
```

Slurm launch, using the same cluster layout pattern as the LoRA branch:

```bash
sbatch scripts/run_full_finetune.slurm
```

## How to Read the Training Log

Each epoch log entry contains:

- `train_loss`
- `train_acc`
- `val_loss`
- `val_top1`
- `val_top5`
- `val_top10`
- `val_mean_class_acc`
- `encoder_lr`
- `head_lr`

How to interpret patterns:

### Case 1: train improves, validation improves

This is normal.

Possible action:

- let it run

### Case 2: train improves, validation plateaus or drops early

This usually means overfitting.

Possible actions:

- lower `encoder_lr`
- increase `weight_decay`
- add `label_smoothing`
- reduce number of epochs
- increase validation size if metrics are noisy

### Case 3: train loss barely moves

This usually means optimization is too conservative.

Possible actions:

- raise `head_lr`
- slightly raise `encoder_lr`
- train longer
- increase train size

### Case 4: loss spikes or becomes NaN

This usually means instability.

Possible actions:

- lower `encoder_lr`
- lower `head_lr`
- increase `warmup_ratio`
- reduce `batch_size`
- try `--disable_amp`

### Case 5: validation metrics jump around a lot

This usually means the validation set is too small or too imbalanced.

Possible actions:

- increase `val_size`
- increase shard count
- raise `patience`

## Common Failure Modes and Fixes

### Out-of-memory errors

Fixes:

- reduce `--batch_size`
- increase `--grad_accum_steps`
- keep AMP enabled
- reduce `--num_workers` only if system RAM is the issue rather than GPU RAM

### Not enough training examples

Symptom:

- script raises because requested `train_size + val_size` exceeds available sampled data

Fixes:

- increase `--max_train_shards`
- or reduce `--train_size` / `--val_size`

### Poor test accuracy despite good train accuracy

Likely causes:

- overfitting
- encoder LR too high
- too little training diversity due to few shards

Fixes:

- lower `encoder_lr`
- increase shard count
- increase `weight_decay`
- add label smoothing
- stop earlier

### Full fine-tune is not beating linear probe

Possible reasons:

- dataset subset too small
- encoder LR too low
- run stopped too early
- head adapts but encoder barely moves

What to try:

- raise `encoder_lr` slightly
- run more epochs
- increase training data
- compare with linear-probe baseline on the same sampled data budget

### Training is too slow

Possible fixes:

- increase `batch_size` if memory allows
- increase `num_workers`
- use more shards only for final runs, not debugging
- keep AMP on

## Suggested Tuning Order

When a run does not work, do not change everything at once.

A reasonable tuning order is:

1. Make sure the run is stable at a small scale.
2. Fix OOM issues using `batch_size` and `grad_accum_steps`.
3. Tune `encoder_lr`.
4. Tune `head_lr`.
5. Tune regularization with `weight_decay` and `label_smoothing`.
6. Tune `patience` and `epochs`.
7. Increase shard count and dataset size once the training recipe looks healthy.

## Practical Recommendations

For first cluster runs:

- keep `encoder_lr` conservative
- use a moderate `head_lr`
- use early stopping
- do not start with all shards
- inspect `training_log.json` after every run

A safe starting point is:

```text
batch_size=32
grad_accum_steps=1
encoder_lr=1e-6
head_lr=1e-4
weight_decay=1e-4
warmup_ratio=0.1
patience=3
label_smoothing=0.0
```

If that is stable but weak, the next thing I would change is:

1. slightly increase `encoder_lr`
2. increase train data and shard coverage
3. add more epochs

## Files to Inspect When Debugging

If something goes wrong, these are the most relevant files:

- [scripts/run_full_finetune.py](/Users/ayushbaweja/Downloads/geolocation-domain-shift/scripts/run_full_finetune.py:1)
- [src/models/clip_geolocation.py](/Users/ayushbaweja/Downloads/geolocation-domain-shift/src/models/clip_geolocation.py:1)
- [src/data/osv5m_dataset.py](/Users/ayushbaweja/Downloads/geolocation-domain-shift/src/data/osv5m_dataset.py:1)
- `results/full_finetune/training_log.json`
- `results/full_finetune/summary.json`

## Bottom Line

The full fine-tuning implementation is intentionally conservative:

- low encoder LR
- higher head LR
- weight decay
- warmup
- cosine decay
- early stopping
- AMP

That is the right default for a large pretrained vision backbone.

If the run does not work as intended, the most important parameters to adjust first are:

1. `batch_size`
2. `grad_accum_steps`
3. `encoder_lr`
4. `head_lr`
5. `weight_decay`
6. `patience`
7. `train_size` and shard count
