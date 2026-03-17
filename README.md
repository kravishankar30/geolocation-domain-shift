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
