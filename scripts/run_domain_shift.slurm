#!/bin/bash
#SBATCH --job-name=domain-shift
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/domain_shift_%j.out
#SBATCH --error=logs/domain_shift_%j.err

cd ~/scratch/geolocation-domain-shift

source .venv/bin/activate

mkdir -p logs results/domain_shift

python scripts/run_domain_shift.py \
  --eval_mode lora \
  --output_dir results/domain_shift \
  --train_size 50000 \
  --test_size 10000 \
  --batch_size 128 \
  --seed 42 \
  --device cuda \
  --cache_dir ~/scratch/osv5m_cache \
  --extract_dir ~/scratch/osv5m_images

