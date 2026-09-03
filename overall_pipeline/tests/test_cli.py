from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from text2sql.cli import main
from text2sql.config import load_config


CODE_ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_path = CODE_ROOT / "configs" / "single_turn_zero_shot.json"
        config = load_config(cls.config_path)
        if not config.spider.root.is_dir():
            raise unittest.SkipTest("Spider data is not installed")

    def test_smoke_overrides_are_recorded_as_effective_config(self):
        with tempfile.TemporaryDirectory() as directory:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = main(
                    [
                        "smoke",
                        "--config",
                        str(self.config_path),
                        "--backend",
                        "mock",
                        "--device",
                        "cuda:7",
                        "--output-dir",
                        directory,
                        "--run-name",
                        "cli-overrides",
                    ]
                )

            self.assertEqual(exit_code, 0, stderr.getvalue())
            manifest = json.loads(
                (Path(directory) / "cli-overrides" / "run_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["config"]["model"]["device"], "cuda:7")
            self.assertEqual(
                manifest["config"]["output"]["directory"],
                str(Path(directory).resolve()),
            )
            self.assertNotIn("source_config", manifest)
            self.assertEqual(
                manifest["config_source_path"],
                str(self.config_path),
            )
            self.assertEqual(len(manifest["source_config_sha256"]), 64)
            self.assertEqual(
                manifest["invocation"],
                {
                    "interface": "cli",
                    "command": "smoke",
                    "config": str(self.config_path),
                    "backend": "mock",
                    "device": "cuda:7",
                    "output_directory": str(Path(directory).resolve()),
                    "run_name": "cli-overrides",
                    "allow_model_download": False,
                },
            )

    def test_validate_official_reports_pinned_environment(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(
                ["validate-official", "--config", str(self.config_path)]
            )

        self.assertEqual(exit_code, 0, stderr.getvalue())
        report = json.loads(stdout.getvalue())
        self.assertTrue(report["ok"])
        self.assertEqual(
            report["actual_commit"],
            "e97acc546ecbee8fa27fa8dbf025ef61493a876c",
        )
        self.assertEqual(len(report["databases"]), 8)

    def test_validate_official_can_cover_all_dev_databases(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(
                [
                    "validate-official",
                    "--config",
                    str(self.config_path),
                    "--all-examples",
                ]
            )

        self.assertEqual(exit_code, 0, stderr.getvalue())
        report = json.loads(stdout.getvalue())
        self.assertTrue(report["ok"])
        self.assertEqual(len(report["databases"]), 20)

    def test_full_evaluate_cli_records_physical_gpu_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            stdout = io.StringIO()
            stderr = io.StringIO()
            fake_result = {
                "run_directory": str(Path(directory) / "full-cli"),
                "summary": {"pipeline_pass": True},
            }
            with mock.patch(
                "text2sql.single_turn.evaluation_runner.run_full_evaluation",
                return_value=fake_result,
            ) as run_mock, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = main(
                    [
                        "evaluate",
                        "--config",
                        str(self.config_path),
                        "--backend",
                        "mock",
                        "--gpus",
                        "1,3,7",
                        "--output-dir",
                        directory,
                        "--run-name",
                        "full-cli",
                    ]
                )

            self.assertEqual(exit_code, 0, stderr.getvalue())
            kwargs = run_mock.call_args.kwargs
            effective_config = run_mock.call_args.args[0]
            self.assertEqual(kwargs["gpu_ids"], (1, 3, 7))
            self.assertEqual(kwargs["selection"], "all")
            self.assertTrue(kwargs["progress"].enabled)
            self.assertEqual(effective_config.model.device, "cuda:0")
            self.assertEqual(effective_config.output.directory, Path(directory).resolve())
            self.assertEqual(
                kwargs["invocation"]["physical_gpu_ids"], [1, 3, 7]
            )
            self.assertEqual(
                kwargs["invocation"]["worker_logical_device"], "cuda:0"
            )
            self.assertEqual(kwargs["invocation"]["selection"], "all")
            self.assertTrue(kwargs["invocation"]["progress_enabled"])
            self.assertEqual(
                kwargs["invocation"]["progress_interval_seconds"], 5.0
            )

    def test_full_evaluate_cli_rejects_invalid_progress_interval(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch(
            "text2sql.single_turn.evaluation_runner.run_full_evaluation"
        ) as run_mock, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(
                [
                    "evaluate",
                    "--config",
                    str(self.config_path),
                    "--backend",
                    "mock",
                    "--gpus",
                    "1",
                    "--progress-interval-seconds",
                    "nan",
                ]
            )
        self.assertEqual(exit_code, 2)
        self.assertIn("positive and finite", stderr.getvalue())
        run_mock.assert_not_called()

    def test_peft_evaluate_requires_and_forwards_adapter_directory(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            fake_result = {
                "run_directory": str(Path(directory) / "peft-run"),
                "summary": {"pipeline_pass": True},
            }
            with mock.patch(
                "text2sql.single_turn.evaluation_runner.run_full_evaluation",
                return_value=fake_result,
            ) as run_mock, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = main(
                    [
                        "evaluate",
                        "--config",
                        str(self.config_path),
                        "--backend",
                        "peft",
                        "--adapter-dir",
                        str(adapter),
                        "--gpus",
                        "0,1",
                        "--run-name",
                        "peft-run",
                    ]
                )
            self.assertEqual(exit_code, 0, stderr.getvalue())
            self.assertEqual(run_mock.call_args.kwargs["adapter_dir"], adapter.resolve())

        stderr = io.StringIO()
        with mock.patch(
            "text2sql.single_turn.evaluation_runner.run_full_evaluation"
        ) as run_mock, contextlib.redirect_stderr(stderr):
            exit_code = main(
                [
                    "evaluate",
                    "--config",
                    str(self.config_path),
                    "--backend",
                    "peft",
                    "--gpus",
                    "0",
                ]
            )
        self.assertEqual(exit_code, 2)
        self.assertIn("--adapter-dir is required", stderr.getvalue())
        run_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
