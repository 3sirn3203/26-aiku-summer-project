from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rl_finetune.training_runtime import write_adapter_provenance


class TrainingRuntimeTests(unittest.TestCase):
    def test_adapter_provenance_pins_base_checkpoint(self) -> None:
        config = SimpleNamespace(
            model=SimpleNamespace(
                model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
                revision="a" * 40,
                source="hub",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            adapter.mkdir()
            write_adapter_provenance(
                adapter,
                config,
                training_run_type="rl_grpo_lora",
            )
            payload = json.loads(
                (adapter / "text2sql_adapter_provenance.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(payload["base_model_id"], config.model.model_id)
        self.assertEqual(payload["base_model_revision"], config.model.revision)
        self.assertEqual(payload["base_model_source"], "hub")
        self.assertEqual(payload["training_run_type"], "rl_grpo_lora")


if __name__ == "__main__":
    unittest.main()
