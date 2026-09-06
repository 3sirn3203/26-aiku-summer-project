from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from text2sql.config import ExecutionConfig
from rl_finetune.dataset import write_jsonl
from rl_vm_step.prepared_dataset import (
    build_prepared_manifest,
    load_prepared_dataset,
    write_json,
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(model_id="model", revision="revision"),
        execution=ExecutionConfig(
            timeout_seconds=1.0,
            max_sql_bytes=1_000,
            max_result_rows=100,
            max_result_bytes=10_000,
            worker_memory_limit_bytes=100_000_000,
        ),
    )


class PreparedDatasetTests(unittest.TestCase):
    def test_integrity_checked_round_trip(self) -> None:
        config = _config()
        record = {"example_id": "train:0", "gold_reference": {"status": "ready"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train_dataset.jsonl"
            gold = root / "gold_reference.jsonl"
            rejected = root / "rejected_gold_reference.jsonl"
            write_jsonl(train, [record])
            write_jsonl(gold, [record["gold_reference"]])
            write_jsonl(rejected, [])
            manifest = build_prepared_manifest(
                config=config,
                records=[record],
                rejected=[],
                train_dataset_path=train,
                gold_reference_path=gold,
                rejected_reference_path=rejected,
                selection={},
            )
            write_json(root / "dataset_manifest.json", manifest)
            with patch("rl_vm_step.prepared_dataset.validate_gold_reference"):
                loaded, loaded_manifest = load_prepared_dataset(root, config)
        self.assertEqual(loaded, [record])
        self.assertEqual(loaded_manifest["status"], "ready")

    def test_modified_dataset_is_rejected(self) -> None:
        config = _config()
        record = {"example_id": "train:0", "gold_reference": {"status": "ready"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train_dataset.jsonl"
            gold = root / "gold_reference.jsonl"
            rejected = root / "rejected_gold_reference.jsonl"
            write_jsonl(train, [record])
            write_jsonl(gold, [record["gold_reference"]])
            write_jsonl(rejected, [])
            manifest = build_prepared_manifest(
                config=config,
                records=[record],
                rejected=[],
                train_dataset_path=train,
                gold_reference_path=gold,
                rejected_reference_path=rejected,
                selection={},
            )
            write_json(root / "dataset_manifest.json", manifest)
            train.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_prepared_dataset(root, config)


if __name__ == "__main__":
    unittest.main()
