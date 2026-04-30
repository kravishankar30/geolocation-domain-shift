#!/bin/bash
#SBATCH --job-name=lora-geolocation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/lora_%j.out
#SBATCH --error=logs/lora_%j.err

cd ~/scratch/geolocation-domain-shift

source .venv/bin/activate

mkdir -p logs results/lora

python scripts/run_lora.py \
  --train_size 50000 \
  --test_size 10000 \
  --epochs 5 \
  --batch_size 32 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --device cuda \
  --num_workers 6 \
  --save_best \
  --output_dir results/lora_50k \
  --cache_dir ~/scratch/osv5m_cache \
  --extract_dir ~/scratch/osv5m_images
