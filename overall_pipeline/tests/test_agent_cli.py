from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from text2sql.cli import main


CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = CODE_ROOT / "configs" / "multi_turn_multi_agent_zero_shot.json"


class AgentCliTests(unittest.TestCase):
    def test_agent_evaluate_applies_role_pool_and_output_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_result = {
                "run_directory": str(Path(directory) / "agent-cli"),
                "summary": {"pipeline_pass": True},
            }
            run_mock = mock.Mock(return_value=fake_result)
            runner_module = types.ModuleType("text2sql.multi_turn_agent.runner")
            runner_module.run_agent_evaluation = run_mock
            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch.dict(
                sys.modules,
                {"text2sql.multi_turn_agent.runner": runner_module},
            ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = main(
                    [
                        "agent-evaluate",
                        "--config",
                        str(CONFIG_PATH),
                        "--backend",
                        "mock",
                        "--selection",
                        "smoke",
                        "--planner-gpus",
                        "7",
                        "--coder-gpus",
                        "3,4",
                        "--verifier-gpus",
                        "0,1",
                        "--output-dir",
                        directory,
                        "--run-name",
                        "agent-cli",
                        "--no-progress",
                        "--progress-interval-seconds",
                        "2.5",
                    ]
                )

        self.assertEqual(exit_code, 0, stderr.getvalue())
        self.assertEqual(json.loads(stdout.getvalue()), fake_result)
        effective_config = run_mock.call_args.args[0]
        kwargs = run_mock.call_args.kwargs
        self.assertEqual(effective_config.gpu_pools.planner, (7,))
        self.assertEqual(effective_config.gpu_pools.coder, (3, 4))
        self.assertEqual(effective_config.gpu_pools.verifier, (0, 1))
        self.assertEqual(effective_config.output.directory, Path(directory).resolve())
        self.assertEqual(kwargs["backend_name"], "mock")
        self.assertEqual(kwargs["selection"], "smoke")
        self.assertFalse(kwargs["allow_model_download"])
        self.assertFalse(kwargs["progress"].enabled)
        self.assertEqual(
            kwargs["invocation"]["gpu_pools"],
            {"planner": [7], "coder": [3, 4], "verifier": [0, 1]},
        )
        self.assertEqual(kwargs["invocation"]["worker_logical_device"], "cuda:0")
        self.assertFalse(kwargs["invocation"]["progress_enabled"])
        self.assertEqual(kwargs["invocation"]["progress_interval_seconds"], 2.5)

    def test_agent_evaluate_rejects_cross_role_gpu_overlap(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(
                [
                    "agent-evaluate",
                    "--config",
                    str(CONFIG_PATH),
                    "--backend",
                    "mock",
                    "--planner-gpus",
                    "0",
                    "--coder-gpus",
                    "0",
                    "--verifier-gpus",
                    "7",
                ]
            )

        self.assertEqual(exit_code, 2)
        self.assertIn("must not overlap", stderr.getvalue())

    def test_agent_evaluate_rejects_download_flag_for_mock(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(
                [
                    "agent-evaluate",
                    "--config",
                    str(CONFIG_PATH),
                    "--backend",
                    "mock",
                    "--allow-model-download",
                ]
            )

        self.assertEqual(exit_code, 2)
        self.assertIn("valid only with --backend hf", stderr.getvalue())

    def test_agent_evaluate_rejects_invalid_progress_interval(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(
                [
                    "agent-evaluate",
                    "--config",
                    str(CONFIG_PATH),
                    "--backend",
                    "mock",
                    "--progress-interval-seconds",
                    "nan",
                ]
            )

        self.assertEqual(exit_code, 2)
        self.assertIn("positive and finite", stderr.getvalue())

    def test_agent_doctor_uses_overridden_role_pools(self) -> None:
        fake_report = {"ok": True, "worker_count": 3, "reports": []}
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch(
            "text2sql.multi_turn_agent.doctor.run_agent_doctor",
            return_value=fake_report,
        ) as doctor_mock, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(
                [
                    "agent-doctor",
                    "--config",
                    str(CONFIG_PATH),
                    "--planner-gpus",
                    "2",
                    "--coder-gpus",
                    "4",
                    "--verifier-gpus",
                    "6",
                ]
            )

        self.assertEqual(exit_code, 0, stderr.getvalue())
        self.assertEqual(json.loads(stdout.getvalue()), fake_report)
        effective_config = doctor_mock.call_args.args[0]
        self.assertEqual(effective_config.gpu_pools.planner, (2,))
        self.assertEqual(effective_config.gpu_pools.coder, (4,))
        self.assertEqual(effective_config.gpu_pools.verifier, (6,))
        self.assertEqual(
            doctor_mock.call_args.args[1],
            {"planner": (2,), "coder": (4,), "verifier": (6,)},
        )


if __name__ == "__main__":
    unittest.main()
