from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from text2sql.multi_turn_agent.config import load_agent_config
from text2sql.multi_turn_agent.doctor import inspect_role_worker, run_agent_doctor


CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = CODE_ROOT / "configs" / "agent_evaluate_dev.json"


def _base_report(free_bytes: int):
    return {
        "ok": True,
        "errors": [],
        "warnings": [],
        "gpu_memory_free_bytes": free_bytes,
        "model_loaded": False,
    }


class AgentDoctorTests(unittest.TestCase):
    def test_planner_and_verifier_enforce_eight_gibibytes(self) -> None:
        with mock.patch(
            "text2sql.multi_turn_agent.doctor.server_doctor",
            side_effect=lambda _device: _base_report(6 * 1024**3),
        ):
            planner = inspect_role_worker(CONFIG_PATH, "planner", 3)
            verifier = inspect_role_worker(CONFIG_PATH, "verifier", 7)

        for report, role, physical_gpu in (
            (planner, "planner", 3),
            (verifier, "verifier", 7),
        ):
            self.assertFalse(report["ok"])
            self.assertEqual(report["role"], role)
            self.assertEqual(report["physical_gpu"], physical_gpu)
            self.assertEqual(report["logical_device"], "cuda:0")
            self.assertEqual(report["role_minimum_free_vram_bytes"], 8 * 1024**3)
            self.assertTrue(any("requires at least" in item for item in report["errors"]))

    def test_coder_accepts_four_gibibyte_threshold_and_records_contract(self) -> None:
        with mock.patch(
            "text2sql.multi_turn_agent.doctor.server_doctor",
            return_value=_base_report(6 * 1024**3),
        ) as doctor_mock:
            report = inspect_role_worker(CONFIG_PATH, "coder", 4)

        doctor_mock.assert_called_once_with("cuda:0")
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["configured_model"], "Qwen/Qwen2.5-Coder-0.5B-Instruct")
        self.assertEqual(
            report["configured_revision"],
            "ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
        )
        self.assertTrue(report["trainable"])
        self.assertEqual(report["role_minimum_free_vram_bytes"], 4 * 1024**3)
        self.assertFalse(report["model_loaded"])

    def test_parent_checks_each_role_gpu_in_isolated_process(self) -> None:
        config = load_agent_config(CONFIG_PATH)
        calls = []

        def completed(command, **kwargs):
            role = command[command.index("--role") + 1]
            gpu = int(command[command.index("--physical-gpu") + 1])
            calls.append((role, gpu, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
            payload = {"role": role, "physical_gpu": gpu, "ok": True, "errors": []}
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps(payload), stderr=""
            )

        with mock.patch(
            "text2sql.multi_turn_agent.doctor.subprocess.run",
            side_effect=completed,
        ):
            report = run_agent_doctor(
                config,
                {"planner": (2,), "coder": (4,), "verifier": (6,)},
            )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["worker_count"], 3)
        self.assertEqual(
            calls,
            [
                ("planner", 2, "2"),
                ("coder", 4, "4"),
                ("verifier", 6, "6"),
            ],
        )

    def test_parent_fails_closed_on_invalid_worker_output(self) -> None:
        config = load_agent_config(CONFIG_PATH)
        invalid = subprocess.CompletedProcess([], 0, stdout="not-json", stderr="")
        with mock.patch(
            "text2sql.multi_turn_agent.doctor.subprocess.run",
            return_value=invalid,
        ):
            report = run_agent_doctor(
                config,
                {"planner": (0,), "coder": (3,), "verifier": (5,)},
            )

        self.assertFalse(report["ok"])
        self.assertEqual(report["worker_count"], 3)
        self.assertTrue(
            all("invalid JSON" in item["errors"][0] for item in report["reports"])
        )


if __name__ == "__main__":
    unittest.main()
