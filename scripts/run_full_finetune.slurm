#!/bin/bash
#SBATCH --job-name=full-finetune-geolocation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/full_finetune_%j.out
#SBATCH --error=logs/full_finetune_%j.err

cd ~/scratch/geolocation-domain-shift

source .venv/bin/activate

mkdir -p logs results/full_finetune

python scripts/run_full_finetune.py \
  --train_size 50000 \
  --val_size 5000 \
  --test_size 10000 \
  --epochs 10 \
  --batch_size 32 \
  --encoder_lr 1e-6 \
  --head_lr 1e-4 \
  --weight_decay 1e-4 \
  --warmup_ratio 0.1 \
  --patience 3 \
  --device cuda \
  --num_workers 6 \
  --save_best \
  --output_dir results/full_finetune \
  --cache_dir ~/scratch/osv5m_cache \
  --extract_dir ~/scratch/osv5m_images
