from __future__ import annotations

import json
import os
import re
import selectors
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Mapping, Optional, Sequence, Set

from text2sql.multi_turn_agent.protocol import (
    PROTOCOL_SCHEMA_VERSION,
    RoleTask,
    RoleTaskResult,
    RoleWorkerSpec,
)


class WorkerInfrastructureError(RuntimeError):
    """A worker process, pipe, startup, or timeout failure."""


class RolePoolClosedError(RuntimeError):
    """A task was submitted to a stopped or unavailable role pool."""


EventCallback = Callable[[Mapping[str, Any]], None]

_RETRYABLE_BACKEND_ERROR_TYPES = {
    "cuda_out_of_memory",
    "generation_error",
}


def _execute_role_task(
    worker: "_RoleWorkerProcess", task: RoleTask
) -> Mapping[str, Any]:
    generation = worker.execute(task)
    if (
        generation.get("status") == "error"
        and generation.get("error_type") in _RETRYABLE_BACKEND_ERROR_TYPES
    ):
        raise WorkerInfrastructureError(
            "%s backend generation failure: %s"
            % (worker.worker_id, generation.get("error_type"))
        )
    return generation


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _safe_emit(callback: Optional[EventCallback], payload: Mapping[str, Any]) -> None:
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        # Progress/telemetry observers must never alter orchestration semantics.
        return


class _RoleWorkerProcess:
    def __init__(
        self,
        spec: RoleWorkerSpec,
        runtime_dir: Path,
        event_callback: Optional[EventCallback],
    ):
        self.spec = spec
        self.runtime_dir = runtime_dir
        self.event_callback = event_callback
        self._process: Optional[subprocess.Popen] = None
        self._stderr_handle: Any = None
        self._io_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._state = "created"
        existing_logs = sorted(
            (
                self.runtime_dir
                / "worker_logs"
                / self.spec.role
            ).glob("%s.attempt-*.log" % self.spec.worker_id)
        )
        existing_attempts = []
        attempt_pattern = re.compile(
            r"^%s\.attempt-([0-9]+)\.log$" % re.escape(self.spec.worker_id)
        )
        for path in existing_logs:
            match = attempt_pattern.fullmatch(path.name)
            if match is not None:
                existing_attempts.append(int(match.group(1)))
        self._launch_count = max(existing_attempts, default=-1) + 1
        self._tasks_started = 0
        self._tasks_completed = 0
        self._infrastructure_failures = 0
        self._ready_event: Optional[Dict[str, Any]] = None
        self._log_paths = [str(path) for path in existing_logs]
        self._last_return_code: Optional[int] = None

    @property
    def worker_id(self) -> str:
        return self.spec.worker_id

    @property
    def healthy(self) -> bool:
        process = self._process
        return self._state == "ready" and process is not None and process.poll() is None

    def _set_state(self, state: str) -> None:
        with self._state_lock:
            self._state = state

    def _read_event(self, timeout_seconds: float) -> Dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise WorkerInfrastructureError("worker stdout is unavailable")
        deadline = time.monotonic() + timeout_seconds
        selector = selectors.DefaultSelector()
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WorkerInfrastructureError(
                        "%s did not respond within %.1f seconds"
                        % (self.worker_id, timeout_seconds)
                    )
                if process.poll() is not None:
                    # There may still be one buffered error/result line after exit.
                    remaining = min(remaining, 0.1)
                events = selector.select(remaining)
                if not events:
                    if process.poll() is not None:
                        raise WorkerInfrastructureError(
                            "%s exited with code %s"
                            % (self.worker_id, process.returncode)
                        )
                    continue
                line = process.stdout.readline()
                if not line:
                    raise WorkerInfrastructureError(
                        "%s closed its stdout pipe (exit=%s)"
                        % (self.worker_id, process.poll())
                    )
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise WorkerInfrastructureError(
                        "%s emitted invalid JSONL protocol output" % self.worker_id
                    ) from exc
                if not isinstance(payload, dict):
                    raise WorkerInfrastructureError(
                        "%s emitted a non-object protocol message" % self.worker_id
                    )
                return payload
        finally:
            selector.close()

    def start(self) -> None:
        with self._io_lock:
            if self.healthy:
                return
            self._close_process_streams()
            launch_attempt = self._launch_count
            spec_path = (
                self.runtime_dir
                / "worker_specs"
                / self.spec.role
                / ("%s.attempt-%02d.json" % (self.worker_id, launch_attempt))
            )
            log_path = (
                self.runtime_dir
                / "worker_logs"
                / self.spec.role
                / ("%s.attempt-%02d.log" % (self.worker_id, launch_attempt))
            )
            _atomic_json(spec_path, self.spec.to_payload(launch_attempt))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._stderr_handle = log_path.open("w", encoding="utf-8")
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(self.spec.physical_gpu),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                }
            )
            self._set_state("starting")
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "text2sql.multi_turn_agent.worker",
                    "--spec",
                    str(spec_path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr_handle,
                env=environment,
                text=True,
                bufsize=1,
            )
            self._launch_count += 1
            self._log_paths.append(str(log_path))
            _safe_emit(
                self.event_callback,
                {
                    "event": "worker_starting",
                    "role": self.spec.role,
                    "worker_id": self.worker_id,
                    "physical_gpu": self.spec.physical_gpu,
                    "logical_device": "cuda:0",
                    "launch_attempt": launch_attempt,
                },
            )
            try:
                event = self._read_event(self.spec.startup_timeout_seconds)
                if event.get("type") == "startup_error":
                    error = event.get("error", {})
                    raise WorkerInfrastructureError(
                        "%s failed during startup: %s"
                        % (self.worker_id, error.get("message", "unknown error"))
                    )
                expected = {
                    "type": "ready",
                    "role": self.spec.role,
                    "worker_id": self.worker_id,
                    "physical_gpu": self.spec.physical_gpu,
                    "logical_device": "cuda:0",
                    "cuda_visible_devices": str(self.spec.physical_gpu),
                }
                mismatched = [
                    key for key, value in expected.items() if event.get(key) != value
                ]
                if event.get("schema_version") != PROTOCOL_SCHEMA_VERSION or mismatched:
                    raise WorkerInfrastructureError(
                        "%s ready handshake mismatch: %s"
                        % (self.worker_id, ", ".join(mismatched) or "schema_version")
                    )
                self._ready_event = dict(event)
                self._set_state("ready")
                _safe_emit(
                    self.event_callback,
                    {
                        "event": "worker_ready",
                        "role": self.spec.role,
                        "worker_id": self.worker_id,
                        "physical_gpu": self.spec.physical_gpu,
                        "logical_device": "cuda:0",
                        "launch_attempt": launch_attempt,
                    },
                )
            except BaseException as exc:
                self._set_state("failed")
                self._terminate_process()
                self._close_process_streams()
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                if isinstance(exc, WorkerInfrastructureError):
                    raise
                raise WorkerInfrastructureError(
                    "%s could not start: %s" % (self.worker_id, exc)
                ) from exc

    def execute(self, task: RoleTask) -> Mapping[str, Any]:
        with self._io_lock:
            process = self._process
            if not self.healthy or process is None or process.stdin is None:
                raise WorkerInfrastructureError("%s is not ready" % self.worker_id)
            self._set_state("busy")
            self._tasks_started += 1
            try:
                process.stdin.write(
                    json.dumps(task.to_payload(), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
                process.stdin.flush()
                event = self._read_event(self.spec.response_timeout_seconds)
                if event.get("type") != "generation_result":
                    if event.get("type") == "protocol_error":
                        error = event.get("error", {})
                        raise WorkerInfrastructureError(
                            "%s rejected a scheduler request: %s"
                            % (self.worker_id, error.get("message", "protocol error"))
                        )
                    raise WorkerInfrastructureError(
                        "%s returned unexpected event %r"
                        % (self.worker_id, event.get("type"))
                    )
                expected = {
                    "task_id": task.task_id,
                    "example_id": task.example_id,
                    "iteration": task.iteration,
                    "prompt_sha256": task.prompt_sha256,
                }
                mismatched = [
                    key for key, value in expected.items() if event.get(key) != value
                ]
                generation = event.get("generation")
                if mismatched or not isinstance(generation, Mapping):
                    raise WorkerInfrastructureError(
                        "%s returned a mismatched task result: %s"
                        % (self.worker_id, ", ".join(mismatched) or "generation")
                    )
                self._tasks_completed += 1
                self._set_state("ready")
                return dict(generation)
            except Exception as exc:
                self._infrastructure_failures += 1
                self._set_state("failed")
                self._terminate_process()
                raise WorkerInfrastructureError(
                    "%s infrastructure failure: %s" % (self.worker_id, exc)
                ) from exc

    def restart(self) -> None:
        self.stop(graceful=False)
        self.start()

    def stop(self, graceful: bool = True) -> None:
        with self._io_lock:
            process = self._process
            if process is None:
                self._set_state("stopped")
                self._close_process_streams()
                return
            if process.poll() is None and graceful and process.stdin is not None:
                try:
                    process.stdin.write(
                        json.dumps(
                            {
                                "schema_version": PROTOCOL_SCHEMA_VERSION,
                                "type": "shutdown",
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    process.stdin.flush()
                    event = self._read_event(10.0)
                    if event.get("type") != "shutdown_ack":
                        raise WorkerInfrastructureError("shutdown was not acknowledged")
                    process.wait(timeout=10.0)
                except Exception:
                    self._terminate_process()
            elif process.poll() is None:
                self._terminate_process()
            self._last_return_code = process.poll()
            self._set_state("stopped")
            self._close_process_streams()

    def force_terminate(self) -> None:
        # Used by cancellation/SIGTERM paths.  It intentionally does not wait for
        # the I/O lock held by a task blocked on a dead or hung child.
        self._terminate_process()
        self._set_state("terminated")

    def finalize_termination(self) -> None:
        # Called only after role-pool task threads have observed child EOF and
        # exited, so closing their pipe objects cannot strand a selector wait.
        self._close_process_streams()

    def _terminate_process(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass
        self._last_return_code = process.poll()

    def _close_process_streams(self) -> None:
        process = self._process
        if process is not None:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        if self._stderr_handle is not None:
            try:
                self._stderr_handle.close()
            except OSError:
                pass
        self._stderr_handle = None
        self._process = None

    def metadata(self) -> Dict[str, Any]:
        process = self._process
        ready = dict(self._ready_event or {})
        return {
            "role": self.spec.role,
            "worker_id": self.worker_id,
            "physical_gpu": self.spec.physical_gpu,
            "logical_device": "cuda:0",
            "backend_selector": self.spec.backend,
            "minimum_free_vram_bytes": self.spec.minimum_free_vram_bytes,
            "state": self._state,
            "pid": process.pid if process is not None else None,
            "launch_count": self._launch_count,
            "tasks_started": self._tasks_started,
            "tasks_completed": self._tasks_completed,
            "infrastructure_failures": self._infrastructure_failures,
            "last_return_code": self._last_return_code,
            "log_paths": list(self._log_paths),
            "backend": ready.get("backend"),
            "ready_at_ns": ready.get("ready_at_ns"),
            "cuda_visible_devices": ready.get("cuda_visible_devices"),
        }


class RoleWorkerPool:
    """Asynchronous one-task-per-worker pool for one model role."""

    def __init__(
        self,
        specs: Sequence[RoleWorkerSpec],
        runtime_dir: Path,
        event_callback: Optional[EventCallback] = None,
    ):
        if not specs:
            raise ValueError("a role pool requires at least one worker")
        roles = {spec.role for spec in specs}
        if len(roles) != 1:
            raise ValueError("all worker specs in a pool must have one role")
        worker_ids = [spec.worker_id for spec in specs]
        if len(set(worker_ids)) != len(worker_ids):
            raise ValueError("worker_id values must be unique within a role pool")
        self.role = specs[0].role
        self._workers = [
            _RoleWorkerProcess(spec, runtime_dir, event_callback) for spec in specs
        ]
        self._event_callback = event_callback
        self._condition = threading.Condition()
        self._idle: Deque[_RoleWorkerProcess] = deque()
        self._usable: Set[_RoleWorkerProcess] = set()
        self._pending_task_ids: Set[str] = set()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._started = False
        self._accepting = False
        self._closed = False

    def start(self) -> None:
        if self._started:
            return
        if self._closed:
            raise RolePoolClosedError("role pool is closed")
        started = []
        try:
            # Blocking ready handshake deliberately serializes GPU model loads.
            for worker in self._workers:
                worker.start()
                started.append(worker)
            with self._condition:
                self._idle.extend(started)
                self._usable.update(started)
                self._executor = ThreadPoolExecutor(
                    max_workers=len(started),
                    thread_name_prefix="agent-%s" % self.role,
                )
                self._started = True
                self._accepting = True
                self._condition.notify_all()
        except BaseException:
            for worker in started:
                worker.stop(graceful=True)
            raise

    def submit(self, task: RoleTask) -> Future:
        with self._condition:
            if not self._started or not self._accepting or self._executor is None:
                raise RolePoolClosedError("role pool %s is not running" % self.role)
            if task.task_id in self._pending_task_ids:
                raise ValueError("task_id is already pending: %s" % task.task_id)
            self._pending_task_ids.add(task.task_id)
            executor = self._executor
            try:
                future = executor.submit(self._run_task, task)
            except BaseException:
                self._pending_task_ids.discard(task.task_id)
                raise

        def clear_pending(_future: Future) -> None:
            with self._condition:
                self._pending_task_ids.discard(task.task_id)
                self._condition.notify_all()

        future.add_done_callback(clear_pending)
        return future

    def _acquire_worker(self) -> _RoleWorkerProcess:
        with self._condition:
            while not self._idle:
                if self._closed:
                    raise RolePoolClosedError("role pool %s is closing" % self.role)
                if not self._usable:
                    raise WorkerInfrastructureError(
                        "role pool %s has no usable workers" % self.role
                    )
                self._condition.wait(timeout=0.2)
            return self._idle.popleft()

    def _release_worker(self, worker: _RoleWorkerProcess) -> None:
        with self._condition:
            if worker in self._usable and not self._closed:
                self._idle.append(worker)
            self._condition.notify_all()

    def _remove_worker(self, worker: _RoleWorkerProcess) -> None:
        with self._condition:
            self._usable.discard(worker)
            try:
                self._idle.remove(worker)
            except ValueError:
                pass
            self._condition.notify_all()

    def _run_task(self, task: RoleTask) -> RoleTaskResult:
        worker = self._acquire_worker()
        retries = 0
        keep_worker = True
        try:
            try:
                generation = _execute_role_task(worker, task)
            except WorkerInfrastructureError as first_error:
                with self._condition:
                    cancelling = self._closed
                if cancelling:
                    keep_worker = False
                    self._remove_worker(worker)
                    raise
                if task.infrastructure_retry_limit == 0:
                    keep_worker = False
                    self._remove_worker(worker)
                    raise
                retries = 1
                _safe_emit(
                    self._event_callback,
                    {
                        "event": "worker_retry",
                        "role": self.role,
                        "worker_id": worker.worker_id,
                        "task_id": task.task_id,
                        "retry": 1,
                        "reason": str(first_error),
                    },
                )
                try:
                    worker.restart()
                    generation = _execute_role_task(worker, task)
                except WorkerInfrastructureError:
                    keep_worker = False
                    self._remove_worker(worker)
                    raise
            return RoleTaskResult(
                role=self.role,
                task_id=task.task_id,
                example_id=task.example_id,
                iteration=task.iteration,
                prompt_sha256=task.prompt_sha256 or "",
                generation=generation,
                infrastructure_retries=retries,
            )
        finally:
            if keep_worker:
                self._release_worker(worker)

    def close(self, wait: bool = True) -> None:
        if not wait:
            self.terminate()
            return
        with self._condition:
            if self._closed:
                return
            # Reject new submissions, but allow already queued tasks to acquire
            # workers and drain before the JSON shutdown handshake.
            self._accepting = False
            executor = self._executor
            self._condition.notify_all()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        for worker in self._workers:
            worker.stop(graceful=True)
        with self._condition:
            self._idle.clear()
            self._usable.clear()

    def terminate(self) -> None:
        with self._condition:
            self._accepting = False
            self._closed = True
            self._condition.notify_all()
        for worker in self._workers:
            worker.force_terminate()
        executor = self._executor
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for worker in self._workers:
            worker.finalize_termination()

    def metadata(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "worker_count": len(self._workers),
            "workers": [worker.metadata() for worker in self._workers],
        }


class AgentWorkerCoordinator:
    """Own all role pools while keeping model libraries out of the parent."""

    _DEFAULT_ROLE_ORDER = ("planner", "coder", "verifier")

    def __init__(
        self,
        specs_by_role: Mapping[str, Sequence[RoleWorkerSpec]],
        runtime_dir: Path,
        event_callback: Optional[EventCallback] = None,
    ):
        if not specs_by_role:
            raise ValueError("at least one role pool is required")
        all_worker_ids = [
            spec.worker_id for specs in specs_by_role.values() for spec in specs
        ]
        if len(set(all_worker_ids)) != len(all_worker_ids):
            raise ValueError("worker_id values must be globally unique")
        for role, specs in specs_by_role.items():
            if not specs or any(spec.role != role for spec in specs):
                raise ValueError("role pool key and worker specs must match")
        self.runtime_dir = Path(runtime_dir).expanduser().resolve()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        role_order = [
            role for role in self._DEFAULT_ROLE_ORDER if role in specs_by_role
        ]
        role_order.extend(
            sorted(role for role in specs_by_role if role not in role_order)
        )
        self._role_order = tuple(role_order)
        self._pools = {
            role: RoleWorkerPool(
                specs_by_role[role], self.runtime_dir, event_callback=event_callback
            )
            for role in self._role_order
        }
        self._started = False
        self._closed = False

    def start(self) -> None:
        if self._started:
            return
        if self._closed:
            raise RolePoolClosedError("coordinator is closed")
        started = []
        try:
            # Pool.start itself is sequential, and roles are also started in a
            # stable order, so only one model is loading at any instant.
            for role in self._role_order:
                pool = self._pools[role]
                pool.start()
                started.append(pool)
            self._started = True
        except BaseException:
            for pool in reversed(started):
                pool.close(wait=True)
            raise

    def submit(self, role: str, task: RoleTask) -> Future:
        if not self._started or self._closed:
            raise RolePoolClosedError("coordinator is not running")
        try:
            pool = self._pools[role]
        except KeyError as exc:
            raise ValueError("unknown role: %s" % role) from exc
        return pool.submit(task)

    def close(self, wait: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        for role in reversed(self._role_order):
            self._pools[role].close(wait=wait)

    def terminate(self) -> None:
        # Do not return merely because graceful close has begun.  A SIGTERM can
        # arrive while close(wait=True) is draining a task; cancellation must
        # still force every child process down immediately.
        self._closed = True
        for role in reversed(self._role_order):
            self._pools[role].terminate()

    def metadata(self) -> Dict[str, Any]:
        return {
            "startup_order": list(self._role_order),
            "roles": {
                role: self._pools[role].metadata() for role in self._role_order
            },
        }

    def __enter__(self) -> "AgentWorkerCoordinator":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback_value: Any) -> None:
        if exc_type is None:
            self.close(wait=True)
        else:
            self.terminate()
