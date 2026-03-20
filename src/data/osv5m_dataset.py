"""OSV-5M data pipeline with selective shard downloading and stratified sampling.

The OSV-5M dataset on HuggingFace (osv5m/osv5m) is organized as:
  - train.csv / test.csv  (metadata: id, latitude, longitude, country, ...)
  - images/train/00.zip ... images/train/97.zip  (98 shards, ~50K images each)
  - images/test/00.zip  ... images/test/04.zip   (5 shards)

This pipeline:
  1. Downloads only the CSV metadata
  2. Performs stratified geographic sampling to select a diverse subset
  3. Determines which ZIP shards contain the selected images
  4. Downloads only those shards
  5. Provides a standard PyTorch Dataset/DataLoader interface
"""

import logging
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

REPO_ID = "osv5m/osv5m"
TRAIN_SHARDS = 98
TEST_SHARDS = 5


class OSV5MDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        subset_size: int = 250_000,
        seed: int = 42,
        cache_dir: str = "./data/osv5m_cache",
        extract_dir: str = "./data/osv5m_images",
        max_shards: int | None = None,
    ):
        self.split = split
        self.subset_size = subset_size
        self.seed = seed
        self.cache_dir = cache_dir
        self.extract_dir = Path(extract_dir) / split
        self.country_to_label: dict[str, int] = {}
        self.label_to_country: dict[int, str] = {}
        self._shard_to_ids: dict[int, list[str]] = defaultdict(list)
        self._downloaded_shards: set[int] = set()

        csv_path = hf_hub_download(
            repo_id=REPO_ID, filename=f"{split}.csv", repo_type="dataset", cache_dir=cache_dir
        )
        n_shards = TRAIN_SHARDS if split == "train" else TEST_SHARDS
        df = pd.read_csv(csv_path)

        # Filter to a subset of shards before sampling to avoid downloading everything
        if max_shards is not None:
            rng = np.random.RandomState(seed)
            selected = set(rng.choice(n_shards, size=min(max_shards, n_shards), replace=False).tolist())
            df = df[df["id"].apply(lambda x: int(x) % n_shards in selected)]
            logger.info("Filtered to %d shards (%d images available)", len(selected), len(df))

        self.metadata = self._sample(df)

        countries = sorted(self.metadata["country"].dropna().unique())
        self.country_to_label = {c: i for i, c in enumerate(countries)}
        self.label_to_country = {i: c for c, i in self.country_to_label.items()}

        # shard to image_id map
        for img_id in self.metadata["id"].astype(str):
            self._shard_to_ids[int(img_id) % n_shards].append(img_id)

    def _sample(self, df: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.RandomState(self.seed)
        if len(df) <= self.subset_size:
            return df.sample(frac=1, random_state=rng).reset_index(drop=True)

        groups = df.groupby("country")
        total = len(df)
        # reserving min per country
        min_per = min(10, self.subset_size // groups.ngroups)
        remainder = self.subset_size - min_per * groups.ngroups

        parts, leftovers = [], []
        for _, group in groups:
            # proportional share of remainder
            target = min(len(group), min_per + int(remainder * len(group) / total))
            sampled = group.sample(n=target, random_state=rng)
            parts.append(sampled)
            if len(group) > target:
                leftovers.append(group.drop(sampled.index))

        result = pd.concat(parts, ignore_index=True)

        if len(result) < self.subset_size and leftovers:
            deficit = self.subset_size - len(result)
            pool = pd.concat(leftovers, ignore_index=True)
            result = pd.concat(
                [result, pool.sample(n=min(deficit, len(pool)), random_state=rng)],
                ignore_index=True,
            )

        return result.sample(frac=1, random_state=rng).reset_index(drop=True).head(self.subset_size)

    def download_images(self) -> None:
        os.makedirs(self.extract_dir, exist_ok=True)
        total = len(self._shard_to_ids)
        for i, shard_idx in enumerate(sorted(self._shard_to_ids), 1):
            if shard_idx in self._downloaded_shards:
                continue
            print(f"[{i}/{total}] Downloading shard {shard_idx:02d}...", flush=True)
            zip_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=f"images/{self.split}/{shard_idx:02d}.zip",
                repo_type="dataset",
                cache_dir=self.cache_dir,
            )
            print(f"[{i}/{total}] Extracting shard {shard_idx:02d}...", flush=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(self.extract_dir)
            print(f"[{i}/{total}] Shard {shard_idx:02d} done.", flush=True)
            self._downloaded_shards.add(shard_idx)

    def _find_image(self, image_id: str) -> Path | None:
        for ext in (".jpg", ".jpeg", ".png"):
            p = self.extract_dir / f"{image_id}{ext}"
            if p.exists():
                return p
        return next(self.extract_dir.rglob(f"{image_id}.*"), None)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        image_id = str(row["id"])

        path = self._find_image(image_id)
        if path is None:
            raise FileNotFoundError(f"{image_id} not found — call download_images() first")

        return {
            "image": Image.open(path).convert("RGB"),
            "label": self.country_to_label.get(row["country"], -1),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "country": row["country"],
        }

    @property
    def num_classes(self) -> int:
        return len(self.country_to_label)


class GeographicHoldoutSplitter:
    def __init__(
        self,
        dataset: OSV5MDataset,
        holdout_countries: list[str] | None = None,
        holdout_fraction: float = 0.15,
        seed: int = 42,
    ):
        if holdout_countries is not None:
            self.holdout_countries = set(holdout_countries)
        else:
            rng = np.random.RandomState(seed)
            counts = dataset.metadata["country"].value_counts()
            countries = list(counts.index)
            rng.shuffle(countries)

            self.holdout_countries = set()
            total = 0
            target = int(len(dataset) * holdout_fraction)
            for c in countries:
                if total >= target:
                    break
                self.holdout_countries.add(c)
                total += counts[c]

        self.dataset = dataset

    def split(self) -> tuple[list[int], list[int]]:
        train_idx, holdout_idx = [], []
        for i, country in enumerate(self.dataset.metadata["country"]):
            (holdout_idx if country in self.holdout_countries else train_idx).append(i)
        return train_idx, holdout_idx


def create_dataloaders(
    subset_size: int = 250_000,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    cache_dir: str = "./data/osv5m_cache",
    extract_dir: str = "./data/osv5m_images",
) -> dict:
    train_ds = OSV5MDataset("train", subset_size, seed, cache_dir, extract_dir)
    test_ds = OSV5MDataset("test", max(subset_size // 10, 10_000), seed, cache_dir, extract_dir)

    all_countries = sorted(set(train_ds.country_to_label) | set(test_ds.country_to_label))
    mapping = {c: i for i, c in enumerate(all_countries)}
    reverse = {i: c for c, i in mapping.items()}
    for ds in (train_ds, test_ds):
        ds.country_to_label = mapping
        ds.label_to_country = reverse

    return {
        "train": DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True),
        "test": DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "train_dataset": train_ds,
        "test_dataset": test_ds,
    }
