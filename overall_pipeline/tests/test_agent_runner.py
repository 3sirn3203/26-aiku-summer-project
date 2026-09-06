from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
from unittest import mock

from text2sql.core.models import ExecutionResult
from text2sql.core.spider import SpiderDataset
from text2sql.multi_turn_agent.config import (
    AgentGpuPoolsConfig,
    load_agent_config,
)
from text2sql.multi_turn_agent.protocol import RoleTask, RoleTaskResult
from text2sql.multi_turn_agent.runner import (
    _mock_output,
    _multi_turn_cost_metrics,
    run_agent_evaluation,
)
from text2sql.multi_turn_agent.scheduler import WorkerInfrastructureError


CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = CODE_ROOT / "configs" / "multi_turn_multi_agent_zero_shot.json"


def _jsonl(path: Path) -> Sequence[Mapping[str, Any]]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _official_preflight(**_kwargs: Any) -> Dict[str, Any]:
    return {
        "ok": True,
        "errors": [],
        "fixture": "agent-runner-unit-test",
    }


def _official_zero_accuracy(items: Sequence[Any], **_kwargs: Any) -> Dict[str, Any]:
    return {
        "results": [
            {
                "example_id": item.example_id,
                "db_id": item.db_id,
                "exact_set_match": {"status": "scored", "match": False},
                "test_suite": {"status": "scored", "match": False},
            }
            for item in items
        ],
        "policy": {"fixture": "always-wrong"},
        "evaluator": {"fixture": "unit-test"},
    }


class _ScriptedCoordinator:
    """In-process stand-in that preserves the runner/worker protocol boundary."""

    instances = []
    fail_role: Optional[str] = None

    def __init__(
        self,
        specs_by_role: Mapping[str, Sequence[Any]],
        runtime_dir: Path,
        event_callback: Optional[Any] = None,
    ) -> None:
        self.specs_by_role = specs_by_role
        self.runtime_dir = runtime_dir
        self.event_callback = event_callback
        self.started = False
        self.stopped = False
        self.terminated = False
        self.tasks = []
        self._lock = threading.Lock()
        type(self).instances.append(self)

    @classmethod
    def reset(cls, *, fail_role: Optional[str] = None) -> None:
        cls.instances = []
        cls.fail_role = fail_role

    @classmethod
    def workers_are_stopped(cls) -> bool:
        return bool(cls.instances) and all(
            instance.stopped or instance.terminated for instance in cls.instances
        )

    def start(self) -> None:
        self.started = True

    def submit(self, role: str, task: RoleTask) -> Future:
        future: Future = Future()
        with self._lock:
            self.tasks.append((role, task))
        if role == type(self).fail_role:
            future.set_exception(
                WorkerInfrastructureError("scripted %s worker failure" % role)
            )
            return future
        future.set_result(
            RoleTaskResult(
                role=role,
                task_id=task.task_id,
                example_id=task.example_id,
                iteration=task.iteration,
                prompt_sha256=task.prompt_sha256 or "",
                generation={
                    "status": "success",
                    "raw_output": task.mock_output or "",
                    "elapsed_seconds": 0.001,
                    "model_id": "scripted-%s" % role,
                },
                infrastructure_retries=0,
            )
        )
        return future

    def close(self, wait: bool = True) -> None:
        if not wait:
            raise AssertionError("runner must drain role workers before evaluation")
        self.stopped = True

    def terminate(self) -> None:
        self.terminated = True

    def metadata(self) -> Dict[str, Any]:
        state = "stopped" if self.stopped else "terminated" if self.terminated else "ready"
        return {
            "startup_order": ["planner", "coder", "verifier"],
            "roles": {
                role: {
                    "role": role,
                    "worker_count": len(specs),
                    "workers": [
                        {
                            "worker_id": spec.worker_id,
                            "physical_gpu": spec.physical_gpu,
                            "logical_device": "cuda:0",
                            "state": state,
                            "backend_selector": spec.backend,
                            "backend": {
                                "backend": spec.backend,
                                "resolved_revision": None,
                            },
                        }
                        for spec in specs
                    ],
                }
                for role, specs in self.specs_by_role.items()
            },
        }


class AgentRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_config = load_agent_config(CONFIG_PATH)
        if not cls.base_config.spider.root.is_dir():
            raise unittest.SkipTest("Spider data is not installed")

    def _config(
        self,
        output_directory: Path,
        *,
        sample_count: int = 3,
        reverse_samples: bool = True,
    ):
        samples = tuple(self.base_config.smoke.samples[:sample_count])
        if reverse_samples:
            samples = tuple(reversed(samples))
        return replace(
            self.base_config,
            smoke=replace(
                self.base_config.smoke,
                samples=samples,
                minimum_executable=1,
            ),
            gpu_pools=AgentGpuPoolsConfig(
                planner=(0,),
                coder=(1,),
                verifier=(2,),
            ),
            output=replace(
                self.base_config.output,
                directory=output_directory,
            ),
        )

    @staticmethod
    def _execution_probe(events: list):
        def execute(_db_path: Path, sql: str, **_kwargs: Any) -> ExecutionResult:
            events.append(
                {
                    "sql": sql,
                    "role_workers_stopped": _ScriptedCoordinator.workers_are_stopped(),
                }
            )
            rows = [["x" * 300, index] for index in range(20)]
            return ExecutionResult(
                status="success",
                rows=rows,
                columns=["payload", "ordinal"],
                elapsed_seconds=0.000004,
                query_elapsed_ns=1_000,
                worker_elapsed_ns=2_000,
                parent_elapsed_ns=4_000,
                vm_steps_lower_bound=5_000,
                vm_steps_upper_bound_exclusive=6_000,
                vm_step_progress_interval=1_000,
                vm_step_measurement_complete=True,
            )

        return execute

    def test_multi_turn_cost_metrics_average_per_episode_cumulative_work(self) -> None:
        trajectories = [
            {
                "iterations": [
                    {
                        "execution_observation": {
                            "vm_steps_lower_bound": 1_000,
                            "query_elapsed_ns": 1_000_000,
                        }
                    }
                ]
            },
            {
                "iterations": [
                    {
                        "execution_observation": {
                            "vm_steps_lower_bound": 2_000,
                            "query_elapsed_ns": 2_000_000,
                        }
                    },
                    {
                        "execution_observation": {
                            "vm_steps_lower_bound": None,
                            "query_elapsed_ns": None,
                        }
                    },
                    {
                        "execution_observation": {
                            "vm_steps_lower_bound": 3_000,
                            "query_elapsed_ns": 3_000_000,
                        }
                    },
                ]
            },
        ]

        self.assertEqual(
            _multi_turn_cost_metrics(trajectories),
            {
                "mean_iterations_used": 2.0,
                "mean_cumulative_tool_vm_steps_lower_bound": 3_000.0,
                "mean_cumulative_tool_query_latency_ms": 3.0,
            },
        )

    def test_scripted_mock_end_to_end_writes_ordered_compact_safe_artifacts(self) -> None:
        _ScriptedCoordinator.reset()
        execution_events = []
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            selected_indices = sorted(sample.index for sample in config.smoke.samples)
            dataset = SpiderDataset(config.spider)
            selected_gold = {
                dataset.get_example(index).gold_sql for index in selected_indices
            }
            before_hashes = {
                dataset.get_example(index).db_id: dataset.database_path(
                    dataset.get_example(index).db_id
                ).read_bytes()
                for index in selected_indices
            }

            with (
                mock.patch(
                    "text2sql.multi_turn_agent.runner.AgentWorkerCoordinator",
                    _ScriptedCoordinator,
                ),
                mock.patch(
                    "text2sql.multi_turn_agent.runner.validate_official_environment",
                    side_effect=_official_preflight,
                ),
                mock.patch(
                    "text2sql.multi_turn_agent.runner.evaluate_official",
                    side_effect=_official_zero_accuracy,
                ),
                mock.patch(
                    "text2sql.multi_turn_agent.runner.execute_sql",
                    side_effect=self._execution_probe(execution_events),
                ),
            ):
                result = run_agent_evaluation(
                    config,
                    backend_name="mock",
                    selection="smoke",
                    run_name="scripted-agent",
                )

            run_dir = Path(result["run_directory"])
            trajectories = _jsonl(run_dir / "trajectories.jsonl")
            records = _jsonl(run_dir / "records.jsonl")
            manifest = json.loads(
                (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            persisted_summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            summary = result["summary"]

            self.assertEqual([item["index"] for item in trajectories], selected_indices)
            self.assertEqual([item["index"] for item in records], selected_indices)
            self.assertEqual(len(trajectories), len(selected_indices))
            self.assertTrue(summary["pipeline_pass"])
            self.assertFalse(summary["accuracy_measurement"])
            self.assertTrue(summary["official_evaluation_ok"])
            self.assertEqual(summary["exact_set_match_accuracy"], 0.0)
            self.assertEqual(summary["test_suite_accuracy"], 0.0)
            self.assertEqual(summary["termination_reasons"], {"verifier_stop": 3})
            self.assertEqual(summary["iterations_used_distribution"], {"1": 3})
            self.assertTrue(summary["selected_databases_unchanged"])
            self.assertIn("page cache", summary["primary_timing_cache_caveat"])
            self.assertEqual(
                persisted_summary,
                {
                    "test_suite_accuracy": 0.0,
                    "exact_set_match_accuracy": 0.0,
                    "result_match_accuracy": 1.0,
                    "mean_vm_steps": 5_000.0,
                    "mean_latency_ms": 0.001,
                    "mean_iterations_used": 1.0,
                    "mean_cumulative_tool_vm_steps_lower_bound": 5_000.0,
                    "mean_cumulative_tool_query_latency_ms": 0.001,
                },
            )
            self.assertEqual(manifest["schema_version"], 2)
            self.assertNotIn("source_config", manifest)
            self.assertNotIn("experiment_config", manifest)
            self.assertNotIn("local_checkpoints", manifest)

            # Each episode has one tool execution while workers are live.  The
            # fresh prediction and gold executions happen only after close().
            self.assertEqual(
                sum(not event["role_workers_stopped"] for event in execution_events),
                len(selected_indices),
            )
            self.assertEqual(
                sum(event["role_workers_stopped"] for event in execution_events),
                2 * len(selected_indices),
            )
            first_final = next(
                index
                for index, event in enumerate(execution_events)
                if event["role_workers_stopped"]
            )
            self.assertTrue(
                all(
                    event["role_workers_stopped"]
                    for event in execution_events[first_final:]
                )
            )

            forbidden_trajectory_keys = {
                "messages",
                "prompt_messages",
                "gold_sql",
                "worker_id",
                "physical_gpu",
                "logical_device",
                "gpu",
            }

            def walk(value: Any):
                if isinstance(value, Mapping):
                    for key, child in value.items():
                        yield key
                        yield from walk(child)
                elif isinstance(value, list):
                    for child in value:
                        yield from walk(child)

            for trajectory in trajectories:
                self.assertTrue(
                    forbidden_trajectory_keys.isdisjoint(set(walk(trajectory)))
                )
                serialized = json.dumps(trajectory, ensure_ascii=False)
                for gold_sql in selected_gold:
                    self.assertNotIn(gold_sql, serialized)
                observation = trajectory["iterations"][0]["execution_observation"]
                self.assertEqual(observation["vm_steps_lower_bound"], 5_000)
                self.assertLessEqual(len(observation["rows"]), 5)
                self.assertLessEqual(
                    len(json.dumps(observation, ensure_ascii=False).encode("utf-8")),
                    4096,
                )

            for record in records:
                predicted = record["predicted_execution"]
                self.assertNotIn("rows", predicted)
                self.assertEqual(predicted["row_count"], 20)
                self.assertEqual(predicted["query_elapsed_ns"], 1_000)
                self.assertEqual(predicted["query_elapsed_ms"], 0.001)
                self.assertEqual(predicted["vm_steps_lower_bound"], 5_000)
                self.assertEqual(
                    predicted["vm_steps_upper_bound_exclusive"], 6_000
                )
                self.assertEqual(len(predicted["result_hash"]), 64)

            self.assertEqual(summary["prediction_vm_steps"]["complete_count"], 3)
            self.assertEqual(summary["gold_vm_steps"]["complete_count"], 3)
            self.assertEqual(summary["tool_vm_steps"]["complete_count"], 3)

            self.assertTrue(
                manifest["dataset"]["selected_databases_unchanged"]
            )
            for db_id, before in before_hashes.items():
                self.assertEqual(dataset.database_path(db_id).read_bytes(), before)
            self.assertEqual(
                manifest["agent_contract"]["final_evaluation"][
                    "starts_after_all_role_workers_exit"
                ],
                True,
            )

            submitted_tasks = [
                task
                for instance in _ScriptedCoordinator.instances
                for _role, task in instance.tasks
            ]
            self.assertEqual(len(submitted_tasks), 3 * len(selected_indices))
            for task in submitted_tasks:
                payload = task.to_payload()
                self.assertNotIn("gold_sql", payload)
                self.assertNotIn("official_evaluation", payload)
                rendered_messages = json.dumps(task.messages, ensure_ascii=False)
                for gold_sql in selected_gold:
                    self.assertNotIn(gold_sql, rendered_messages)

    def test_real_role_subprocesses_retry_verifier_format_end_to_end(self) -> None:
        verifier_calls = 0

        def output(role, iteration, serialized_schema=None):
            nonlocal verifier_calls
            if role == "verifier":
                verifier_calls += 1
                if verifier_calls == 1:
                    return "not json"
            return _mock_output(role, iteration, serialized_schema)

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(
                Path(directory), sample_count=1, reverse_samples=False
            )
            with (
                mock.patch(
                    "text2sql.multi_turn_agent.runner._mock_output",
                    side_effect=output,
                ),
                mock.patch(
                    "text2sql.multi_turn_agent.runner.validate_official_environment",
                    side_effect=_official_preflight,
                ),
                mock.patch(
                    "text2sql.multi_turn_agent.runner.evaluate_official",
                    side_effect=_official_zero_accuracy,
                ),
            ):
                result = run_agent_evaluation(
                    config,
                    backend_name="mock",
                    selection="smoke",
                    run_name="real-mock-role-processes",
                )

            run_dir = Path(result["run_directory"])
            trajectories = _jsonl(run_dir / "trajectories.jsonl")
            records = _jsonl(run_dir / "records.jsonl")
            manifest = json.loads(
                (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(verifier_calls, 2)
            self.assertEqual(len(trajectories[0]["iterations"]), 1)
            self.assertEqual(trajectories[0]["termination_reason"], "verifier_stop")
            attempt = trajectories[0]["iterations"][0]["verifier_initial_attempt"]
            self.assertEqual(attempt["generation"]["raw_output"], "not json")
            self.assertIsNotNone(attempt["contract_error"])
            self.assertEqual(len(trajectories), 1)
            self.assertEqual(len(records), 1)
            self.assertEqual(trajectories[0]["final_sql"], "SELECT 1")
            self.assertIsInstance(
                records[0]["predicted_execution"]["query_elapsed_ns"], int
            )
            self.assertTrue(result["summary"]["pipeline_pass"])
            self.assertFalse(result["summary"]["accuracy_measurement"])
            self.assertEqual(result["summary"]["exact_set_match_accuracy"], 0.0)
            workers = [
                worker
                for role in manifest["worker_runtime"]["roles"].values()
                for worker in role["workers"]
            ]
            self.assertEqual(len(workers), 3)
            self.assertTrue(all(worker["state"] == "stopped" for worker in workers))
            self.assertTrue(all(worker["last_return_code"] == 0 for worker in workers))

    def test_resume_from_coder_stage_skips_completed_planner_call(self) -> None:
        execution_events = []
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(
                Path(directory), sample_count=1, reverse_samples=False
            )
            _ScriptedCoordinator.reset(fail_role="coder")
            common_patches = (
                mock.patch(
                    "text2sql.multi_turn_agent.runner.validate_official_environment",
                    side_effect=_official_preflight,
                ),
                mock.patch(
                    "text2sql.multi_turn_agent.runner.evaluate_official",
                    side_effect=_official_zero_accuracy,
                ),
                mock.patch(
                    "text2sql.multi_turn_agent.runner.execute_sql",
                    side_effect=self._execution_probe(execution_events),
                ),
            )
            with (
                mock.patch(
                    "text2sql.multi_turn_agent.runner.AgentWorkerCoordinator",
                    _ScriptedCoordinator,
                ),
                common_patches[0],
                common_patches[1],
                common_patches[2],
            ):
                with self.assertRaisesRegex(RuntimeError, "resume artifacts"):
                    run_agent_evaluation(
                        config,
                        backend_name="mock",
                        selection="smoke",
                        run_name="resume-coder-stage",
                    )

            first_instance = _ScriptedCoordinator.instances[0]
            self.assertEqual(
                [role for role, _task in first_instance.tasks],
                ["planner", "coder"],
            )
            checkpoint_path = (
                Path(directory)
                / "resume-coder-stage"
                / "episodes"
                / ("%05d.json" % config.smoke.samples[0].index)
            )
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["state"]["next_stage"], "coder")
            self.assertIsNotNone(checkpoint["state"]["current_iteration"])

            _ScriptedCoordinator.reset()
            execution_events.clear()
            with (
                mock.patch(
                    "text2sql.multi_turn_agent.runner.AgentWorkerCoordinator",
                    _ScriptedCoordinator,
                ),
                mock.patch(
                    "text2sql.multi_turn_agent.runner.validate_official_environment",
                    side_effect=_official_preflight,
                ),
                mock.patch(
                    "text2sql.multi_turn_agent.runner.evaluate_official",
                    side_effect=_official_zero_accuracy,
                ),
                mock.patch(
                    "text2sql.multi_turn_agent.runner.execute_sql",
                    side_effect=self._execution_probe(execution_events),
                ),
            ):
                resumed = run_agent_evaluation(
                    config,
                    backend_name="mock",
                    selection="smoke",
                    resume_run="resume-coder-stage",
                )

            resumed_instance = _ScriptedCoordinator.instances[0]
            self.assertEqual(
                [role for role, _task in resumed_instance.tasks],
                ["coder", "verifier"],
            )
            self.assertTrue(resumed["summary"]["pipeline_pass"])
            final_checkpoint = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )
            self.assertEqual(final_checkpoint["state"]["next_stage"], "complete")
            retry_journal = final_checkpoint["infrastructure_retries"]
            self.assertEqual(retry_journal["dev:864:coder:iteration-1"], 1)
            self.assertEqual(retry_journal["dev:864:planner:iteration-1"], 0)
            self.assertEqual(retry_journal["dev:864:verifier:iteration-1"], 0)

    def test_importing_parent_runner_never_imports_model_libraries(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(CODE_ROOT / "src")
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import json,sys; "
                    "import text2sql.multi_turn_agent.runner; "
                    "print(json.dumps({name: name in sys.modules for name in "
                    "['torch','transformers','huggingface_hub']}))"
                ),
            ],
            cwd=str(CODE_ROOT),
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads(completed.stdout),
            {"torch": False, "transformers": False, "huggingface_hub": False},
        )

    def test_mock_path_does_not_attempt_dynamic_model_imports(self) -> None:
        _ScriptedCoordinator.reset()
        blocked = {"torch", "transformers", "huggingface_hub"}
        imported = []
        original_import = builtins.__import__

        def guarded_import(name: str, *args: Any, **kwargs: Any):
            if name.split(".", 1)[0] in blocked:
                imported.append(name)
                raise AssertionError("mock runner imported %s" % name)
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(
                Path(directory), sample_count=1, reverse_samples=False
            )
            with (
                mock.patch(
                    "text2sql.multi_turn_agent.runner.AgentWorkerCoordinator",
                    _ScriptedCoordinator,
                ),
                mock.patch(
                    "text2sql.multi_turn_agent.runner.validate_official_environment",
                    side_effect=_official_preflight,
                ),
                mock.patch(
                    "text2sql.multi_turn_agent.runner.evaluate_official",
                    side_effect=_official_zero_accuracy,
                ),
                mock.patch(
                    "text2sql.multi_turn_agent.runner.execute_sql",
                    side_effect=self._execution_probe([]),
                ),
            ):
                builtins.__import__ = guarded_import
                try:
                    result = run_agent_evaluation(
                        config,
                        backend_name="mock",
                        selection="smoke",
                        run_name="lazy-import-mock",
                    )
                finally:
                    builtins.__import__ = original_import

        self.assertEqual(imported, [])
        self.assertTrue(result["summary"]["pipeline_pass"])


if __name__ == "__main__":
    unittest.main()
