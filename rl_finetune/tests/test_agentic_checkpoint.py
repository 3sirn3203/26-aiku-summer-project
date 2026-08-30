from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from text2sql.config import ConfigError
from rl_finetune.agentic_grpo_trainer import AgenticGRPOConfig
from rl_finetune.train_agentic_grpo_lora import (
    _checkpoint_metadata,
    _dataset_fingerprint,
    _validate_base_provenance,
    main,
)


class AgenticCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "example_id": "train:0",
                "db_id": "fixture",
                "prompt_sha256": "prompt",
                "schema_sha256": "schema",
                "gold_sql": "SELECT 1",
            }
        ]
        self.config = SimpleNamespace(
            model=SimpleNamespace(model_id="Qwen/Coder", revision="revision-1")
        )
        self.trainer_config = AgenticGRPOConfig(max_steps=3)
        self.rollout_contract = {
            "num_generations": 4,
            "temperature": 0.9,
            "max_draft_tokens": 256,
            "max_final_tokens": 256,
            "max_observation_rows": 20,
        }

    def _checkpoint(self, root: Path, *, revision: str = "revision-1") -> Path:
        checkpoint = root / "checkpoint-1"
        checkpoint.mkdir()
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pt"):
            (checkpoint / name).write_bytes(b"fixture")
        (checkpoint / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "Qwen/Coder",
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.0,
                }
            ),
            encoding="utf-8",
        )
        state = {
            "global_step": 1,
            "record_index": 7,
            "config": {
                **self.trainer_config.__dict__,
                "max_steps": 1,
            },
            "checkpoint_metadata": {
                "schema_version": 1,
                "model": {"id": "Qwen/Coder", "revision": revision},
                "lora": {
                    "r": 8,
                    "alpha": 16,
                    "dropout": 0.0,
                    "target_modules": "all-linear",
                },
                "rollout": self.rollout_contract,
                "dataset_sha256": _dataset_fingerprint(self.records),
            },
        }
        (checkpoint / "trainer_state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
        return checkpoint

    def test_agentic_v1_rejects_unverified_base(self) -> None:
        with self.assertRaisesRegex(ConfigError, "pinned Coder base"):
            _validate_base_provenance(self.config)

    def test_resume_restores_cursor_and_actual_lora_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentic_checkpoint_") as directory:
            checkpoint = self._checkpoint(Path(directory))
            state = _checkpoint_metadata(
                checkpoint,
                config=self.config,
                records=self.records,
                trainer_config=self.trainer_config,
                rollout_contract=self.rollout_contract,
            )
        self.assertEqual(state["global_step"], 1)
        self.assertEqual(state["record_index"], 7)
        self.assertEqual(state["metadata"]["lora"]["r"], 8)

    def test_resume_rejects_model_revision_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentic_checkpoint_") as directory:
            checkpoint = self._checkpoint(Path(directory), revision="wrong-revision")
            with self.assertRaisesRegex(ConfigError, "model provenance"):
                _checkpoint_metadata(
                    checkpoint,
                    config=self.config,
                    records=self.records,
                    trainer_config=self.trainer_config,
                    rollout_contract=self.rollout_contract,
                )

    def test_resume_rejects_dataset_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentic_checkpoint_") as directory:
            checkpoint = self._checkpoint(Path(directory))
            changed_records = [dict(self.records[0], example_id="train:1")]
            with self.assertRaisesRegex(ConfigError, "dataset selection"):
                _checkpoint_metadata(
                    checkpoint,
                    config=self.config,
                    records=changed_records,
                    trainer_config=self.trainer_config,
                    rollout_contract=self.rollout_contract,
                )

    def test_model_initialization_failure_is_recorded_in_manifest(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="agentic_manifest_") as directory:
            with patch(
                "rl_finetune.train_agentic_grpo_lora.check_training_dependencies",
                return_value={},
            ), patch(
                "rl_finetune.train_agentic_grpo_lora._execute_training",
                side_effect=RuntimeError("model initialization OOM"),
            ), self.assertRaisesRegex(RuntimeError, "initialization OOM"):
                main(
                    [
                        "--baseline-config",
                        str(
                            repository_root
                            / "overall_pipeline"
                            / "configs"
                            / "evaluate_dev.json"
                        ),
                        "--output-dir",
                        directory,
                        "--run-name",
                        "failed-run",
                        "--limit",
                        "1",
                    ]
                )
            manifest = json.loads(
                (Path(directory) / "failed-run" / "run_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["error"]["type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
