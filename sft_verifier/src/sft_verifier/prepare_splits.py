from __future__ import annotations

import argparse
from pathlib import Path

from . import DATA_SCHEMA_VERSION
from .common import DEFAULT_SPIDER_ROOT, load_spider_train, sha256_file, stable_bucket, write_json


def build_manifest(spider_root: Path, seed: int) -> dict:
    examples = load_spider_train(spider_root)
    db_ids = sorted({str(item["db_id"]) for item in examples})
    if len(db_ids) < 3:
        raise ValueError("at least three databases are required for DB-disjoint splits")
    ranked = sorted(db_ids, key=lambda value: (stable_bucket(value, seed), value))
    train_end = min(max(1, int(len(ranked) * 0.8)), len(ranked) - 2)
    validation_count = min(max(1, int(len(ranked) * 0.1)), len(ranked) - train_end - 1)
    validation_end = train_end + validation_count
    split_dbs = {
        "train": sorted(ranked[:train_end]),
        "validation": sorted(ranked[train_end:validation_end]),
        "test": sorted(ranked[validation_end:]),
    }
    db_to_split = {
        db_id: split for split, values in split_dbs.items() for db_id in values
    }
    for item in examples:
        item["split"] = db_to_split[str(item["db_id"])]
    counts = {
        split: sum(item["split"] == split for item in examples)
        for split in split_dbs
    }
    return {
        "schema_version": DATA_SCHEMA_VERSION,
        "seed": seed,
        "policy": "db_id_disjoint_80_10_10",
        "source": {
            name: {
                "path": str((spider_root / name).resolve()),
                "sha256": sha256_file(spider_root / name),
            }
            for name in ("train_spider.json", "train_others.json", "tables.json")
        },
        "database_splits": split_dbs,
        "example_counts": counts,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create DB-disjoint Spider train splits")
    parser.add_argument("--spider-root", type=Path, default=DEFAULT_SPIDER_ROOT)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.spider_root.resolve(), args.seed)
    write_json(args.output, manifest)
    print("wrote %d examples to %s" % (len(manifest["examples"]), args.output))


if __name__ == "__main__":
    main()
