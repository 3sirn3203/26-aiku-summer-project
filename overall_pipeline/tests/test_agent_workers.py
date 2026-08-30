from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from text2sql.multi_turn_agent.protocol import (
    GIB,
    AgentWorkerProtocolError,
    RoleTask,
    RoleWorkerSpec,
)
from text2sql.multi_turn_agent.scheduler import (
    AgentWorkerCoordinator,
    WorkerInfrastructureError,
    _execute_role_task,
)


CODE_ROOT = Path(__file__).resolve().parents[1]


def _messages(label: str = "question"):
    return (
        {"role": "system", "content": "Return one test response."},
        {"role": "user", "content": label},
    )


def _wait_for_worker_state(
    coordinator: AgentWorkerCoordinator,
    role: str,
    state: str,
    launch_count: int,
    timeout_seconds: float = 5.0,
):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        worker = coordinator.metadata()["roles"][role]["workers"][0]
        if worker["state"] == state and worker["launch_count"] == launch_count:
            return worker
        time.sleep(0.01)
    raise AssertionError(
        "worker did not reach state=%s launch_count=%d" % (state, launch_count)
    )


class AgentWorkerProtocolTests(unittest.TestCase):
    def test_role_defaults_and_parent_local_retry_contract(self) -> None:
        planner = RoleWorkerSpec(
            role="planner",
            worker_id="planner-00",
            physical_gpu=0,
            backend="mock",
        )
        coder = RoleWorkerSpec(
            role="coder",
            worker_id="coder-00",
            physical_gpu=1,
            backend="mock",
        )
        self.assertEqual(planner.minimum_free_vram_bytes, 8 * GIB)
        self.assertEqual(coder.minimum_free_vram_bytes, 4 * GIB)

        task = RoleTask(
            task_id="dev-0-planner-1",
            example_id="dev:0",
            iteration=1,
            messages=_messages(),
            infrastructure_retry_limit=0,
        )
        payload = task.to_payload()
        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "type",
                "task_id",
                "example_id",
                "iteration",
                "messages",
                "prompt_sha256",
            },
        )
        self.assertNotIn("infrastructure_retry_limit", payload)
        self.assertFalse({"gold_sql", "official_evaluation"} & set(payload))

        with self.assertRaises(AgentWorkerProtocolError):
            RoleTask(
                task_id="bad",
                example_id="dev:0",
                iteration=1,
                messages=_messages(),
                prompt_sha256="0" * 64,
            )
        with self.assertRaises(AgentWorkerProtocolError):
            RoleTask(
                task_id="bad-retry",
                example_id="dev:0",
                iteration=1,
                messages=_messages(),
                infrastructure_retry_limit=2,
            )

    def test_importing_parent_scheduler_does_not_import_model_libraries(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(CODE_ROOT / "src")
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import json,sys; "
                    "import text2sql.multi_turn_agent.scheduler; "
                    "print(json.dumps({'torch': 'torch' in sys.modules, "
                    "'transformers': 'transformers' in sys.modules}))"
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
            {"torch": False, "transformers": False},
        )


class AgentWorkerSchedulerTests(unittest.TestCase):
    def test_backend_generation_failures_are_infrastructure_retryable(self) -> None:
        task = RoleTask("backend-error", "dev:0", 1, _messages())

        class FakeWorker:
            worker_id = "fake-worker"

            def __init__(self, error_type: str) -> None:
                self.error_type = error_type

            def execute(self, _task: RoleTask):
                return {
                    "status": "error",
                    "error_type": self.error_type,
                    "error_message": "fixture",
                }

        for error_type in ("cuda_out_of_memory", "generation_error"):
            with self.subTest(error_type=error_type), self.assertRaises(
                WorkerInfrastructureError
            ):
                _execute_role_task(FakeWorker(error_type), task)
        non_retryable = _execute_role_task(FakeWorker("input_too_long"), task)
        self.assertEqual(non_retryable["error_type"], "input_too_long")

    @unittest.skipUnless(hasattr(signal, "SIGTERM"), "requires POSIX signals")
    def test_parent_sigterm_terminates_busy_role_worker(self) -> None:
        script = textwrap.dedent(
            """
            import json, time
            from pathlib import Path
            from text2sql.multi_turn_agent.protocol import RoleTask, RoleWorkerSpec
            from text2sql.multi_turn_agent.runner import _install_worker_cleanup_handlers
            from text2sql.multi_turn_agent.scheduler import AgentWorkerCoordinator

            runtime = Path(__import__('sys').argv[1])
            responses = {'slow': {'output': 'done', 'delay_seconds': 30.0}}
            specs = {'coder': [RoleWorkerSpec(
                'coder', 'coder-00', 0, 'mock', mock_responses=responses,
                response_timeout_seconds=40.0,
            )]}
            coordinator = AgentWorkerCoordinator(specs, runtime)
            coordinator.start()
            _install_worker_cleanup_handlers(coordinator)
            coordinator.submit(
                'coder', RoleTask('slow', 'dev:0', 1, (
                    {'role': 'system', 'content': 'Return one value.'},
                    {'role': 'user', 'content': 'test'},
                ))
            )
            deadline = time.monotonic() + 5.0
            while True:
                worker = coordinator.metadata()['roles']['coder']['workers'][0]
                if worker['state'] == 'busy':
                    print(json.dumps({'worker_pid': worker['pid']}), flush=True)
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError('worker never became busy')
                time.sleep(0.01)
            time.sleep(30.0)
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(CODE_ROOT / "src")
            parent = subprocess.Popen(
                [sys.executable, "-B", "-c", script, directory],
                cwd=str(CODE_ROOT),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert parent.stdout is not None
            readable, _, _ = select.select([parent.stdout], [], [], 10.0)
            if not readable:
                parent.kill()
                _stdout, stderr = parent.communicate(timeout=5.0)
                self.fail("signal-test parent did not become ready: %s" % stderr)
            ready = json.loads(parent.stdout.readline())
            worker_pid = int(ready["worker_pid"])
            parent.send_signal(signal.SIGTERM)
            parent.wait(timeout=10.0)
            parent.communicate(timeout=1.0)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    os.kill(worker_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                self.fail("role worker survived parent SIGTERM")

    def test_role_pools_start_sequentially_and_keep_gpu_metadata_run_level(self) -> None:
        specs = {
            "planner": [
                RoleWorkerSpec("planner", "planner-00", 0, "mock")
            ],
            "coder": [RoleWorkerSpec("coder", "coder-00", 3, "mock")],
            "verifier": [
                RoleWorkerSpec("verifier", "verifier-00", 5, "mock")
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            coordinator = AgentWorkerCoordinator(specs, Path(directory))
            coordinator.start()
            try:
                metadata = coordinator.metadata()
                self.assertEqual(
                    metadata["startup_order"], ["planner", "coder", "verifier"]
                )
                ready_times = []
                for role, gpu in (("planner", 0), ("coder", 3), ("verifier", 5)):
                    worker = metadata["roles"][role]["workers"][0]
                    self.assertEqual(worker["state"], "ready")
                    self.assertEqual(worker["physical_gpu"], gpu)
                    self.assertEqual(worker["logical_device"], "cuda:0")
                    self.assertEqual(worker["cuda_visible_devices"], str(gpu))
                    self.assertNotIn("torch", sys.modules)
                    ready_times.append(worker["ready_at_ns"])
                    log_path = Path(worker["log_paths"][0])
                    self.assertTrue(log_path.is_file())
                    self.assertIn("ready for role", log_path.read_text(encoding="utf-8"))
                self.assertEqual(ready_times, sorted(ready_times))

                futures = [
                    coordinator.submit(
                        role,
                        RoleTask(
                            task_id="dev-0-%s-1" % role,
                            example_id="dev:0",
                            iteration=1,
                            messages=_messages(role),
                            mock_output=role + " output",
                        ),
                    )
                    for role in ("planner", "coder", "verifier")
                ]
                results = [future.result(timeout=5.0) for future in futures]
                self.assertEqual(
                    [result.generation["raw_output"] for result in results],
                    ["planner output", "coder output", "verifier output"],
                )
                self.assertTrue(all(result.infrastructure_retries == 0 for result in results))
            finally:
                coordinator.close(wait=True)
            stopped = coordinator.metadata()
            self.assertTrue(
                all(
                    worker["state"] == "stopped"
                    and worker["last_return_code"] == 0
                    for role in stopped["roles"].values()
                    for worker in role["workers"]
                )
            )

    def test_two_workers_dispatch_tasks_concurrently(self) -> None:
        responses = {
            "task-a": {"output": "A", "delay_seconds": 0.35},
            "task-b": {"output": "B", "delay_seconds": 0.35},
        }
        specs = {
            "coder": [
                RoleWorkerSpec(
                    "coder", "coder-00", 3, "mock", mock_responses=responses
                ),
                RoleWorkerSpec(
                    "coder", "coder-01", 4, "mock", mock_responses=responses
                ),
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            with AgentWorkerCoordinator(specs, Path(directory)) as coordinator:
                started = time.monotonic()
                first = coordinator.submit(
                    "coder", RoleTask("task-a", "dev:0", 1, _messages("a"))
                )
                second = coordinator.submit(
                    "coder", RoleTask("task-b", "dev:1", 1, _messages("b"))
                )
                outputs = {
                    first.result(timeout=5).generation["raw_output"],
                    second.result(timeout=5).generation["raw_output"],
                }
                elapsed = time.monotonic() - started
                self.assertEqual(outputs, {"A", "B"})
                self.assertLess(elapsed, 0.62)
                workers = coordinator.metadata()["roles"]["coder"]["workers"]
                self.assertEqual([worker["tasks_completed"] for worker in workers], [1, 1])

    def test_graceful_close_drains_already_submitted_tasks(self) -> None:
        responses = {
            "first": {"output": "one", "delay_seconds": 0.1},
            "second": {"output": "two", "delay_seconds": 0.1},
        }
        specs = {
            "coder": [
                RoleWorkerSpec(
                    "coder", "coder-00", 3, "mock", mock_responses=responses
                )
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            coordinator = AgentWorkerCoordinator(specs, Path(directory))
            coordinator.start()
            first = coordinator.submit(
                "coder", RoleTask("first", "dev:0", 1, _messages("first"))
            )
            second = coordinator.submit(
                "coder", RoleTask("second", "dev:1", 1, _messages("second"))
            )
            coordinator.close(wait=True)
            self.assertEqual(first.result().generation["raw_output"], "one")
            self.assertEqual(second.result().generation["raw_output"], "two")
            worker = coordinator.metadata()["roles"]["coder"]["workers"][0]
            self.assertEqual(worker["tasks_completed"], 2)
            self.assertEqual(worker["state"], "stopped")
            self.assertEqual(worker["last_return_code"], 0)

    def test_new_coordinator_preserves_previous_worker_logs_on_resume(self) -> None:
        specs = {
            "coder": [RoleWorkerSpec("coder", "coder-00", 3, "mock")]
        }
        with tempfile.TemporaryDirectory() as directory:
            runtime_dir = Path(directory)
            first = AgentWorkerCoordinator(specs, runtime_dir)
            first.start()
            first.close(wait=True)
            first_log = (
                runtime_dir
                / "worker_logs"
                / "coder"
                / "coder-00.attempt-00.log"
            )
            original = first_log.read_text(encoding="utf-8")

            second = AgentWorkerCoordinator(specs, runtime_dir)
            second.start()
            second.close(wait=True)
            second_log = (
                runtime_dir
                / "worker_logs"
                / "coder"
                / "coder-00.attempt-01.log"
            )
            self.assertTrue(second_log.is_file())
            self.assertEqual(first_log.read_text(encoding="utf-8"), original)
            worker = second.metadata()["roles"]["coder"]["workers"][0]
            self.assertEqual(worker["launch_count"], 2)
            self.assertEqual(
                worker["log_paths"],
                [str(first_log.resolve()), str(second_log.resolve())],
            )

    @unittest.skipUnless(hasattr(signal, "SIGKILL"), "requires POSIX process signals")
    def test_worker_crash_restarts_once_and_replays_the_same_task(self) -> None:
        responses = {
            "crash-once": {"output": "recovered", "delay_seconds": 0.35}
        }
        specs = {
            "planner": [
                RoleWorkerSpec(
                    "planner",
                    "planner-00",
                    0,
                    "mock",
                    mock_responses=responses,
                    response_timeout_seconds=2.0,
                )
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            with AgentWorkerCoordinator(specs, Path(directory)) as coordinator:
                future = coordinator.submit(
                    "planner",
                    RoleTask("crash-once", "dev:0", 1, _messages()),
                )
                busy = _wait_for_worker_state(coordinator, "planner", "busy", 1)
                os.kill(busy["pid"], signal.SIGKILL)
                result = future.result(timeout=5)
                self.assertEqual(result.generation["raw_output"], "recovered")
                self.assertEqual(result.infrastructure_retries, 1)
                worker = coordinator.metadata()["roles"]["planner"]["workers"][0]
                self.assertEqual(worker["launch_count"], 2)
                self.assertEqual(worker["tasks_started"], 2)
                self.assertEqual(worker["tasks_completed"], 1)
                self.assertEqual(worker["infrastructure_failures"], 1)

    @unittest.skipUnless(hasattr(signal, "SIGKILL"), "requires POSIX process signals")
    def test_second_worker_crash_fails_without_a_third_launch(self) -> None:
        responses = {
            "crash-twice": {"output": "never returned", "delay_seconds": 0.5}
        }
        specs = {
            "planner": [
                RoleWorkerSpec(
                    "planner",
                    "planner-00",
                    0,
                    "mock",
                    mock_responses=responses,
                    response_timeout_seconds=2.0,
                )
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            coordinator = AgentWorkerCoordinator(specs, Path(directory))
            coordinator.start()
            future = coordinator.submit(
                "planner", RoleTask("crash-twice", "dev:0", 1, _messages())
            )
            first = _wait_for_worker_state(coordinator, "planner", "busy", 1)
            os.kill(first["pid"], signal.SIGKILL)
            second = _wait_for_worker_state(coordinator, "planner", "busy", 2)
            os.kill(second["pid"], signal.SIGKILL)
            with self.assertRaises(WorkerInfrastructureError):
                future.result(timeout=5)
            worker = coordinator.metadata()["roles"]["planner"]["workers"][0]
            self.assertEqual(worker["launch_count"], 2)
            self.assertEqual(worker["tasks_started"], 2)
            self.assertEqual(worker["infrastructure_failures"], 2)
            coordinator.close(wait=True)

    @unittest.skipUnless(hasattr(signal, "SIGKILL"), "requires POSIX process signals")
    def test_resume_retry_limit_zero_does_not_restart_a_crashed_worker(self) -> None:
        responses = {"no-retry": {"output": "unused", "delay_seconds": 0.5}}
        specs = {
            "verifier": [
                RoleWorkerSpec(
                    "verifier",
                    "verifier-00",
                    5,
                    "mock",
                    mock_responses=responses,
                    response_timeout_seconds=2.0,
                )
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            coordinator = AgentWorkerCoordinator(specs, Path(directory))
            coordinator.start()
            future = coordinator.submit(
                "verifier",
                RoleTask(
                    "no-retry",
                    "dev:0",
                    1,
                    _messages(),
                    infrastructure_retry_limit=0,
                ),
            )
            busy = _wait_for_worker_state(coordinator, "verifier", "busy", 1)
            os.kill(busy["pid"], signal.SIGKILL)
            with self.assertRaises(WorkerInfrastructureError):
                future.result(timeout=5)
            worker = coordinator.metadata()["roles"]["verifier"]["workers"][0]
            self.assertEqual(worker["launch_count"], 1)
            coordinator.close(wait=True)


if __name__ == "__main__":
    unittest.main()
