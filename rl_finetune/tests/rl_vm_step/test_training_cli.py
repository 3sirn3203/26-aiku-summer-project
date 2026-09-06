from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from text2sql.config import ConfigError
from rl_vm_step.train_grpo_lora import (
    VM_STEP_V1_BASE_MODEL_ID,
    VM_STEP_V1_BASE_REVISION,
    _create_grpo_config,
    _parser,
    main,
    _validate_args,
    _validate_base_provenance,
)


class TrainingCLITests(unittest.TestCase):
    def test_defaults_define_vm_only_reward(self) -> None:
        args = _parser().parse_args(
            [
                "--baseline-config",
                "config.json",
                "--output-dir",
                "outputs",
                "--run-name",
                "test",
            ]
        )
        _validate_args(args)
        self.assertEqual(args.vm_weight, 0.1)
        self.assertEqual(args.vm_epsilon_steps, 1_000.0)
        self.assertFalse(hasattr(args, "latency_weight"))
        self.assertTrue(args.fp16)
        self.assertTrue(args.gradient_checkpointing)
        self.assertEqual(args.gradient_accumulation_steps, 4)
        self.assertEqual(args.attention_implementation, "sdpa")

    def test_gradient_checkpointing_uses_non_reentrant_mode(self) -> None:
        args = _parser().parse_args(
            [
                "--baseline-config",
                "config.json",
                "--output-dir",
                "outputs",
                "--run-name",
                "test",
                "--gradient-checkpointing",
            ]
        )
        args.run_dir = Path("outputs/test")
        training_config = _create_grpo_config(args)
        self.assertTrue(training_config.gradient_checkpointing)
        self.assertEqual(
            training_config.gradient_checkpointing_kwargs,
            {"use_reentrant": False},
        )

    def test_model_revision_is_pinned(self) -> None:
        valid = SimpleNamespace(
            model=SimpleNamespace(
                model_id=VM_STEP_V1_BASE_MODEL_ID,
                revision=VM_STEP_V1_BASE_REVISION,
            )
        )
        _validate_base_provenance(valid)
        invalid = SimpleNamespace(
            model=SimpleNamespace(model_id=VM_STEP_V1_BASE_MODEL_ID, revision="main")
        )
        with self.assertRaises(ConfigError):
            _validate_base_provenance(invalid)

    def test_dry_run_writes_vm_step_manifest(self) -> None:
        config = SimpleNamespace(
            model=SimpleNamespace(
                model_id=VM_STEP_V1_BASE_MODEL_ID,
                revision=VM_STEP_V1_BASE_REVISION,
                dtype="float32",
                attention_implementation="eager",
            )
        )
        reference = {"status": "ready", "example_id": "train:0"}
        records = [{"prompt": [], "gold_reference": reference}]
        with tempfile.TemporaryDirectory() as directory, patch(
            "rl_vm_step.train_grpo_lora.load_config", return_value=config
        ), patch(
            "rl_vm_step.train_grpo_lora._training_config", return_value=config
        ), patch(
            "rl_vm_step.train_grpo_lora.build_vm_step_records",
            return_value=(records, []),
        ):
            output_dir = Path(directory) / "outputs"
            result = main(
                [
                    "--baseline-config",
                    "config.json",
                    "--output-dir",
                    str(output_dir),
                    "--run-name",
                    "dry-run",
                    "--dry-run-dataset",
                ]
            )
            manifest = json.loads(
                (output_dir / "dry-run" / "run_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(result, 0)
        self.assertEqual(manifest["run_type"], "single_turn_vm_step_grpo_lora")
        self.assertEqual(manifest["config"]["reward"]["cost_scope"], "final_only")
        self.assertFalse(
            manifest["config"]["grpo"]["gradient_checkpointing_use_reentrant"]
        )
        self.assertNotIn("latency_weight", manifest["config"]["reward"])

    def test_multi_gpu_requires_prepared_dataset(self) -> None:
        argv = [
            "--baseline-config",
            "config.json",
            "--output-dir",
            "outputs",
            "--run-name",
            "test",
        ]
        with patch.dict("os.environ", {"RANK": "0", "WORLD_SIZE": "2"}):
            with self.assertRaises(ConfigError):
                main(argv)


if __name__ == "__main__":
    unittest.main()
