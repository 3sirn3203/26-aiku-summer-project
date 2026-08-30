from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rl_finetune.evaluate_two_turn_base import main


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class EvaluationCliTests(unittest.TestCase):
    def test_two_turn_cli_forwards_distributed_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_result = {
                "run_directory": str(Path(directory) / "run"),
                "summary": {"pipeline_pass": True},
            }
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch(
                "rl_finetune.evaluate_two_turn_base.run_full_evaluation",
                return_value=fake_result,
            ) as runner, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = main(
                    [
                        "--baseline-config",
                        str(
                            REPOSITORY_ROOT
                            / "overall_pipeline"
                            / "configs"
                            / "evaluate_dev.json"
                        ),
                        "--gpus",
                        "0,2",
                        "--selection",
                        "smoke",
                        "--output-dir",
                        directory,
                        "--run-name",
                        "run",
                    ]
                )
        self.assertEqual(exit_code, 0, stderr.getvalue())
        kwargs = runner.call_args.kwargs
        self.assertEqual(kwargs["backend_name"], "two_turn")
        self.assertEqual(kwargs["gpu_ids"], (0, 2))
        self.assertIsNone(kwargs["adapter_dir"])
        self.assertEqual(kwargs["workflow_contract"]["turns"], 2)
        self.assertFalse(kwargs["workflow_contract"]["gold_available_to_generation"])
        self.assertEqual(
            kwargs["workflow_contract"]["seed_policy"],
            "base_seed_plus_spider_index",
        )


if __name__ == "__main__":
    unittest.main()
