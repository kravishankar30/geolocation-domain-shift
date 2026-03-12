"""uv run python scripts/verify_data_pipeline.py --test-csv [--subset-size N]"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", action="store_true")
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--cache-dir", default="./data/osv5m_cache")
    args = parser.parse_args()

    if not args.test_csv:
        logger.info("Pass --test-csv to run (requires network).")
        return

    from src.data.osv5m_dataset import OSV5MDataset

    ds = OSV5MDataset(split="test", subset_size=args.subset_size, seed=42, cache_dir=args.cache_dir)

    logger.info("samples=%d  countries=%d", len(ds), ds.num_classes)
    dist = ds.metadata["country"].value_counts()
    for country, count in dist.head(10).items():
        logger.info("  %s: %d", country, count)
    logger.info("shards needed: %s", sorted(ds._shard_to_ids))

    assert len(ds) == args.subset_size
    assert ds.num_classes > 0
    logger.info("OK")


if __name__ == "__main__":
    main()
