from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import signal
import sqlite3
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from text2sql.core.evaluation import compare_results
from text2sql.core.executor import execute_sql
from text2sql.core.models import ExecutionResult, GenerationRequest, GenerationResult, SpiderExample
from text2sql.core.model_source import (
    expected_model_identity,
    prepare_model_config,
    validate_download_policy,
)
from text2sql.core.official_eval import (
    OfficialEvaluationError,
    OfficialEvaluationItem,
    evaluate_official,
    validate_official_environment,
)
from text2sql.core.progress import ProgressReporter
from text2sql.core.reporting import (
    compact_run_manifest,
    empty_metric_summary,
    metric_summary,
)
from text2sql.core.schema import serialize_schema
from text2sql.core.spider import SpiderDataError, SpiderDataset
from text2sql.multi_turn_agent.artifacts import (
    SAFE_RUN_NAME,
    atomic_json,
    atomic_jsonl,
    effective_config_payload,
    execution_payload,
    json_bytes,
    query_timing_summary,
    run_directory,
    sha256_bytes,
    sha256_file,
    source_snapshot,
    vm_step_summary,
)
from text2sql.multi_turn_agent.config import AgentAppConfig
from text2sql.multi_turn_agent.protocol import RoleTask, RoleWorkerSpec
from text2sql.multi_turn_agent.scheduler import AgentWorkerCoordinator
from text2sql.multi_turn_agent.workflow import (
    AgentEpisodeRequest,
    EpisodeCheckpoint,
    EpisodeResult,
    run_episode,
)


_INFRASTRUCTURE_OFFICIAL_STATUSES = {
    "evaluator_error",
    "evaluator_timeout",
    "gold_error",
}
_TASK_ITERATION = re.compile(r":iteration-([1-3])$")


class AgentRunInterrupted(RuntimeError):
    """Raised after a parent signal has synchronously terminated role workers."""


def _install_worker_cleanup_handlers(
    coordinator: AgentWorkerCoordinator,
) -> Mapping[int, Any]:
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous: Dict[int, Any] = {}

    def handle(signum: int, _frame: Any) -> None:
        coordinator.terminate()
        raise AgentRunInterrupted(
            "agent run interrupted by signal %d after terminating role workers"
            % signum
        )

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handle)
    return previous


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    if threading.current_thread() is not threading.main_thread():
        return
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _not_run(error_type: str, message: str) -> ExecutionResult:
    return ExecutionResult(
        status="not_run",
        error_type=error_type,
        error_message=message,
    )


def _execution_args(config: AgentAppConfig) -> Dict[str, Any]:
    return {
        "timeout_seconds": config.execution.timeout_seconds,
        "max_sql_bytes": config.execution.max_sql_bytes,
        "max_result_rows": config.execution.max_result_rows,
        "max_result_bytes": config.execution.max_result_bytes,
        "worker_memory_limit_bytes": config.execution.worker_memory_limit_bytes,
    }


def _selected_examples(
    dataset: SpiderDataset,
    config: AgentAppConfig,
    selection: str,
) -> List[SpiderExample]:
    if selection == "all":
        return list(dataset.examples)
    if selection != "smoke":
        raise ValueError("selection must be 'smoke' or 'all'")
    selected: List[SpiderExample] = []
    for sample in config.smoke.samples:
        example = dataset.get_example(sample.index)
        if example.db_id != sample.db_id:
            raise SpiderDataError(
                "Agent smoke manifest db_id mismatch at %s:%d"
                % (example.split, example.index)
            )
        if _sha256_text(example.question) != sample.question_sha256:
            raise SpiderDataError(
                "Agent smoke question hash mismatch at %s:%d"
                % (example.split, example.index)
            )
        if _sha256_text(example.gold_sql) != sample.gold_sql_sha256:
            raise SpiderDataError(
                "Agent smoke gold SQL hash mismatch at %s:%d"
                % (example.split, example.index)
            )
        selected.append(example)
    return sorted(selected, key=lambda example: example.index)


def _experiment_config_payload(config: AgentAppConfig) -> Dict[str, Any]:
    """Return the semantic contract, excluding operational pool/output overrides."""

    payload = effective_config_payload(config)
    payload.pop("gpu_pools", None)
    payload.pop("output", None)
    return payload


def _agent_contract(config: AgentAppConfig, selection: str) -> Dict[str, Any]:
    payload = {
        "schema_version": 1,
        "workflow": "planner_coder_execute_verifier",
        "selection": selection,
        "max_iterations": config.workflow.max_iterations,
        "role_order": ["planner", "coder", "verifier"],
        "checkpoint_stages": ["planner", "coder", "execution", "verifier"],
        "planner_approach": ["direct", "iterative"],
        "verifier_decision": ["stop", "continue"],
        "json_repair_generation": False,
        "verifier_has_stop_authority_after_execution_error": True,
        "third_continue_policy": "evaluate_latest_candidate",
        "gold_available_to_roles": False,
        "official_results_available_to_roles": False,
        "database_values_in_role_context": (
            "bounded tool observation after the first candidate execution"
        ),
        "observation": {
            "max_rows": config.workflow.observation_max_rows,
            "max_bytes": config.workflow.observation_max_bytes,
        },
        "role_processes": {
            "one_model_per_worker": True,
            "logical_device": "cuda:0",
            "parent_imports_model_libraries": False,
            "one_outstanding_task_per_worker": True,
            "model_load_order": "sequential",
            "infrastructure_retry_limit_per_stage": (
                config.workflow.infrastructure_retry_limit
            ),
        },
        "tool_execution": {
            "maximum_concurrency": config.workflow.execution_concurrency,
            "purpose": "agent observation and diagnostic timing only",
        },
        "final_evaluation": {
            "starts_after_all_role_workers_exit": True,
            "owner": "single CPU parent process",
            "execution_order_per_example": ["prediction", "gold"],
            "official_evaluation_after_local_execution": True,
        },
        "result_collection": {
            "mode": "full_cursor_stream",
            "retained_rows": "bounded_prefix",
            "comparison": "row_count_and_streaming_fingerprints",
            "row_fingerprint_version": 1,
        },
        "prediction_sql_execution_time": {
            "primary_observation": "predicted_execution.query_elapsed_ns",
            "clock": "time.perf_counter_ns (monotonic)",
            "query_interval_start": "immediately_before_sqlite_connection_execute",
            "query_interval_end": "after full cursor consumption or query error",
            "tool_timing_is_primary": False,
            "model_generation_runtime_included": False,
            "official_evaluator_runtime_included": False,
            "cache_policy": (
                "no explicit cache reset; agent tool executions may warm the DB/OS "
                "page cache before the fresh final prediction execution"
            ),
            "reward_design_in_scope": False,
        },
        "vm_step_measurement": {
            "method": "sqlite_progress_handler_bounds",
            "exact_stmt_status": False,
            "progress_interval": 1000,
            "lower_bound_field": "vm_steps_lower_bound",
            "upper_bound_exclusive_for_completed_field": (
                "vm_steps_upper_bound_exclusive"
            ),
            "role_prompt_input": False,
        },
        "artifacts": {
            "full_prompt_messages_saved": False,
            "per_record_worker_or_gpu_saved": False,
            "role_prompt_sha256_saved": True,
            "stage_checkpoint_atomic": True,
            "canonical_episode_state": "episodes/<index>.json",
        },
    }
    return {
        "payload": payload,
        "sha256": sha256_bytes(json_bytes(payload)),
    }


def _worker_specs(
    config: AgentAppConfig,
    backend_name: str,
    allow_model_download: bool,
) -> Dict[str, Sequence[RoleWorkerSpec]]:
    pools = {
        "planner": config.gpu_pools.planner,
        "coder": config.gpu_pools.coder,
        "verifier": config.gpu_pools.verifier,
    }
    result: Dict[str, Sequence[RoleWorkerSpec]] = {}
    for role in ("planner", "coder", "verifier"):
        role_config = config.roles[role]
        model = {
            "source": role_config.model.source,
            "id": role_config.model.model_id,
            "revision": role_config.model.revision,
            "checkpoint_identity": role_config.model.checkpoint_identity,
            "dtype": role_config.model.dtype,
            "device": "cuda:0",
            "attention_implementation": role_config.model.attention_implementation,
            "trust_remote_code": role_config.model.trust_remote_code,
            "cache_dir": (
                str(role_config.model.cache_dir)
                if role_config.model.cache_dir is not None
                else None
            ),
        }
        generation = {
            "do_sample": role_config.generation.do_sample,
            "num_beams": role_config.generation.num_beams,
            "repetition_penalty": role_config.generation.repetition_penalty,
            "max_time_seconds": role_config.generation.max_time_seconds,
            "max_input_tokens": role_config.generation.max_input_tokens,
            "max_new_tokens": role_config.generation.max_new_tokens,
            "batch_size": role_config.generation.batch_size,
        }
        result[role] = tuple(
            RoleWorkerSpec(
                role=role,
                worker_id="%s-%02d" % (role, position),
                physical_gpu=physical_gpu,
                backend=backend_name,
                model=model,
                generation=generation,
                allow_model_download=allow_model_download,
                mock_responses={},
                minimum_free_vram_bytes=role_config.minimum_free_vram_bytes,
                startup_timeout_seconds=900.0,
                response_timeout_seconds=(
                    role_config.generation.max_time_seconds + 90.0
                ),
            )
            for position, physical_gpu in enumerate(pools[role])
        )
    return result


def _mock_output(role: str, iteration: int) -> str:
    # This scripted path validates orchestration without using gold SQL.  SELECT
    # 1 is intentionally independent of the question and therefore cannot be
    # reported as model accuracy.
    if role == "planner":
        return json.dumps(
            {
                "iteration": iteration,
                "approach": "direct",
                "plan": ["Translate the question using only the supplied schema."],
                "coder_instruction": "Return one bounded read-only SQLite query.",
            },
            separators=(",", ":"),
        )
    if role == "coder":
        return "SELECT 1"
    if role == "verifier":
        return json.dumps(
            {
                "iteration": iteration,
                "decision": "stop",
                "reason": "Scripted orchestration check completed.",
                "feedback": "",
            },
            separators=(",", ":"),
        )
    raise ValueError("unknown role: %s" % role)


def _generation_result(payload: Mapping[str, Any]) -> GenerationResult:
    return GenerationResult(
        status=str(payload.get("status", "error")),
        raw_output=str(payload.get("raw_output", "")),
        elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
        error_type=(
            str(payload["error_type"])
            if payload.get("error_type") is not None
            else None
        ),
        error_message=(
            str(payload["error_message"])
            if payload.get("error_message") is not None
            else None
        ),
        input_tokens=(
            int(payload["input_tokens"])
            if payload.get("input_tokens") is not None
            else None
        ),
        output_tokens=(
            int(payload["output_tokens"])
            if payload.get("output_tokens") is not None
            else None
        ),
        model_id=(
            str(payload["model_id"])
            if payload.get("model_id") is not None
            else None
        ),
        requested_revision=(
            str(payload["requested_revision"])
            if payload.get("requested_revision") is not None
            else None
        ),
        resolved_revision=(
            str(payload["resolved_revision"])
            if payload.get("resolved_revision") is not None
            else None
        ),
    )


def _initial_episode_state(example_id: str) -> Dict[str, Any]:
    return EpisodeCheckpoint(
        example_id=example_id,
        next_iteration=1,
        next_stage="planner",
        completed_iterations=(),
        current_iteration=None,
        last_iteration=None,
        last_raw_output=None,
        last_sql_parsing=None,
        result=None,
    ).to_dict()


def _episode_checkpoint_path(run_dir: Path, index: int) -> Path:
    return run_dir / "episodes" / ("%05d.json" % index)


def _checkpoint_payload(
    *,
    run_id: str,
    example: SpiderExample,
    contract_sha256: str,
    state: Mapping[str, Any],
    infrastructure_retries: Mapping[str, int],
    last_event: str,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "example_id": "%s:%d" % (example.split, example.index),
        "split": example.split,
        "index": example.index,
        "db_id": example.db_id,
        "question_sha256": _sha256_text(example.question),
        "episode_contract_sha256": contract_sha256,
        "last_event": last_event,
        "infrastructure_retries": dict(infrastructure_retries),
        "state": dict(state),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_episode_checkpoint(
    path: Path,
    *,
    run_id: str,
    example: SpiderExample,
    contract_sha256: str,
) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("invalid episode checkpoint: %s" % path) from exc
    expected = {
        "schema_version": 1,
        "run_id": run_id,
        "example_id": "%s:%d" % (example.split, example.index),
        "split": example.split,
        "index": example.index,
        "db_id": example.db_id,
        "question_sha256": _sha256_text(example.question),
        "episode_contract_sha256": contract_sha256,
    }
    mismatched = [key for key, value in expected.items() if payload.get(key) != value]
    state = payload.get("state")
    retries = payload.get("infrastructure_retries", {})
    if mismatched or not isinstance(state, Mapping) or not isinstance(retries, Mapping):
        raise RuntimeError(
            "episode checkpoint contract mismatch at %s: %s"
            % (path, ", ".join(mismatched) or "state/retries")
        )
    EpisodeCheckpoint.from_dict(state)
    normalized_retries: Dict[str, int] = {}
    for task_id, count in retries.items():
        if (
            not isinstance(task_id, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise RuntimeError("invalid retry journal in %s" % path)
        normalized_retries[task_id] = count
    payload["infrastructure_retries"] = normalized_retries
    return payload


def _role_resolved_revisions(
    metadata: Mapping[str, Any],
) -> Dict[str, Sequence[str]]:
    result: Dict[str, Sequence[str]] = {}
    roles = metadata.get("roles")
    if not isinstance(roles, Mapping):
        return result
    for role in ("planner", "coder", "verifier"):
        role_metadata = roles.get(role)
        workers = role_metadata.get("workers") if isinstance(role_metadata, Mapping) else None
        revisions = []
        if isinstance(workers, list):
            for worker in workers:
                backend = worker.get("backend") if isinstance(worker, Mapping) else None
                revision = backend.get("resolved_revision") if isinstance(backend, Mapping) else None
                if isinstance(revision, str):
                    revisions.append(revision)
        result[role] = tuple(sorted(set(revisions)))
    return result


def _validate_resolved_revisions(
    config: AgentAppConfig,
    backend_name: str,
    metadata: Mapping[str, Any],
) -> Dict[str, Sequence[str]]:
    revisions = _role_resolved_revisions(metadata)
    if backend_name == "mock":
        return revisions
    for role in ("planner", "coder", "verifier"):
        actual = revisions.get(role, ())
        expected = expected_model_identity(config.roles[role].model)
        if tuple(actual) != (expected,):
            raise RuntimeError(
                "%s workers did not resolve exactly the configured model identity %s: %s"
                % (role, expected, list(actual))
            )
    return revisions


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    records: List[Dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    repaired = False
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            if line_number == len(lines):
                repaired = True
                break
            raise RuntimeError("invalid JSONL at %s:%d" % (path, line_number)) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("JSONL record is not an object at %s:%d" % (path, line_number))
        records.append(payload)
    if repaired:
        atomic_jsonl(path, records)
    return records


def _official_metric_summary(
    records: Sequence[Mapping[str, Any]], metric: str
) -> Dict[str, Any]:
    results = [record["official_evaluation"][metric] for record in records]
    statuses = Counter(result["status"] for result in results)
    infrastructure_failures = sum(
        result["status"] in _INFRASTRUCTURE_OFFICIAL_STATUSES
        for result in results
    )
    valid = infrastructure_failures == 0 and len(results) == len(records)
    matches = sum(bool(result.get("match")) for result in results) if valid else None
    return {
        "valid": valid,
        "classified_examples": len(results) - infrastructure_failures,
        "infrastructure_failures": infrastructure_failures,
        "statuses": dict(statuses),
        "matches": matches,
        "accuracy": (
            matches / len(records)
            if records and matches is not None
            else None
        ),
    }


def _tool_timing_summary(trajectories: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    observations = []
    for trajectory in trajectories:
        for iteration in trajectory.get("iterations", []):
            observation = iteration.get("execution_observation")
            if isinstance(observation, Mapping):
                observations.append(observation)
    values = [
        observation["query_elapsed_ns"]
        for observation in observations
        if observation.get("status") == "success"
        and isinstance(observation.get("query_elapsed_ns"), int)
        and not isinstance(observation["query_elapsed_ns"], bool)
    ]
    if values:
        ordered = sorted(values)
        middle = len(ordered) // 2
        median = (
            float(ordered[middle])
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0
        )
        aggregate: Mapping[str, Optional[float]] = {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "median": median,
        }
    else:
        aggregate = {"min": None, "max": None, "mean": None, "median": None}
    return {
        "diagnostic_only": True,
        "total_tool_executions": len(observations),
        "successful_tool_executions": sum(
            observation.get("status") == "success" for observation in observations
        ),
        "count": len(values),
        "statuses": dict(Counter(str(item.get("status")) for item in observations)),
        "query_elapsed_ns": dict(aggregate),
        "query_elapsed_ms": {
            key: (value / 1_000_000 if value is not None else None)
            for key, value in aggregate.items()
        },
    }


def _tool_vm_step_summary(
    trajectories: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    observations = []
    for trajectory in trajectories:
        for iteration in trajectory.get("iterations", []):
            observation = iteration.get("execution_observation")
            if isinstance(observation, Mapping):
                observations.append({"execution": observation})
    summary = vm_step_summary(observations, "execution")
    summary["diagnostic_only"] = True
    return summary


def _trajectory_payload(
    run_id: str,
    example: SpiderExample,
    schema_text: str,
    result: EpisodeResult,
) -> Dict[str, Any]:
    payload = result.to_dict()
    return {
        "schema_version": 1,
        "run_id": run_id,
        "example_id": "%s:%d" % (example.split, example.index),
        "split": example.split,
        "index": example.index,
        "db_id": example.db_id,
        "question": example.question,
        "schema_sha256": _sha256_text(schema_text),
        "status": payload["status"],
        "termination_reason": payload["termination_reason"],
        "final_iteration": payload["final_iteration"],
        "final_raw_output": payload["final_raw_output"],
        "final_sql": payload["final_sql"],
        "final_sql_parsing": payload["final_sql_parsing"],
        "iterations": payload["iterations"],
    }


def _failure_payload(stage: str, exc: BaseException) -> Dict[str, str]:
    return {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc)[:2000],
    }


def _write_failure(
    *,
    run_dir: Path,
    manifest: Dict[str, Any],
    stage: str,
    exc: BaseException,
    started_wall: datetime,
    started_monotonic: float,
    processed_examples: int,
    total_examples: int,
) -> None:
    failure = _failure_payload(stage, exc)
    finished = datetime.now(timezone.utc)
    summary = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "run_type": manifest["run_type"],
        "pipeline_pass": False,
        "total_examples": total_examples,
        "processed_examples": processed_examples,
        "failure": failure,
        "resume_supported": True,
        "started_at": started_wall.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
    }
    atomic_json(run_dir / "summary.json", empty_metric_summary())
    manifest.update(
        {
            "status": "failed",
            "failure": failure,
            "finished_at": finished.isoformat(),
            "elapsed_seconds": summary["elapsed_seconds"],
            "artifacts": {
                "episode_checkpoints": "episodes/",
                "trajectories": "trajectories.jsonl",
                "records": "records.jsonl",
                "summary": "summary.json",
                "worker_runtime": "worker_runtime/",
            },
        }
    )
    atomic_json(run_dir / "run_manifest.json", compact_run_manifest(manifest))


def _validate_resume_manifest(
    manifest: Mapping[str, Any],
    *,
    backend_name: str,
    selection: str,
    total: int,
    experiment_config_sha256: str,
    source_tree_sha256: str,
    agent_contract_sha256: str,
    dataset_contract_sha256: str,
    official_preflight_sha256: Optional[str],
) -> None:
    expected = {
        "backend_selector": backend_name,
        "experiment_config_sha256": experiment_config_sha256,
        "agent_contract_sha256": agent_contract_sha256,
    }
    mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
    source = manifest.get("source")
    dataset = manifest.get("dataset")
    official = manifest.get("official_evaluation")
    if not isinstance(source, Mapping) or source.get("python_tree_sha256") != source_tree_sha256:
        mismatched.append("source.python_tree_sha256")
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("selection") != selection
        or dataset.get("total_examples") != total
        or dataset.get("contract_sha256") != dataset_contract_sha256
    ):
        mismatched.append("dataset")
    if (
        not isinstance(official, Mapping)
        or official.get("preflight_sha256") != official_preflight_sha256
    ):
        mismatched.append("official_evaluation.preflight_sha256")
    if mismatched:
        raise RuntimeError(
            "refusing to resume because the agent run contract changed: %s"
            % ", ".join(mismatched)
        )


def _run_episode_tasks(
    *,
    config: AgentAppConfig,
    run_id: str,
    run_dir: Path,
    selected_examples: Sequence[SpiderExample],
    dataset: SpiderDataset,
    coordinator: AgentWorkerCoordinator,
    backend_name: str,
    contract_sha256: str,
    loaded_checkpoints: Mapping[int, Mapping[str, Any]],
    completed_results: Mapping[int, EpisodeResult],
    reporter: ProgressReporter,
) -> Dict[int, EpisodeResult]:
    results = dict(completed_results)
    execution_limit = threading.BoundedSemaphore(config.workflow.execution_concurrency)
    execution_kwargs = _execution_args(config)
    retry_lock = threading.Lock()

    def run_one(example: SpiderExample) -> Tuple[int, EpisodeResult]:
        example_id = "%s:%d" % (example.split, example.index)
        checkpoint_path = _episode_checkpoint_path(run_dir, example.index)
        loaded = loaded_checkpoints.get(example.index)
        state: Mapping[str, Any] = (
            loaded["state"] if loaded is not None else _initial_episode_state(example_id)
        )
        retries: Dict[str, int] = dict(
            loaded.get("infrastructure_retries", {}) if loaded is not None else {}
        )
        state_holder: Dict[str, Mapping[str, Any]] = {"state": state}

        def persist(event: str, new_state: Optional[Mapping[str, Any]] = None) -> None:
            if new_state is not None:
                state_holder["state"] = dict(new_state)
            atomic_json(
                checkpoint_path,
                _checkpoint_payload(
                    run_id=run_id,
                    example=example,
                    contract_sha256=contract_sha256,
                    state=state_holder["state"],
                    infrastructure_retries=retries,
                    last_event=event,
                ),
            )

        if loaded is None:
            persist("episode_initialized")

        def generator(role: str):
            def generate(request: GenerationRequest) -> GenerationResult:
                match = _TASK_ITERATION.search(request.example_id)
                if match is None:
                    raise RuntimeError("workflow emitted an invalid role task ID")
                iteration = int(match.group(1))
                task_id = request.example_id
                with retry_lock:
                    prior_retries = retries.get(task_id, 0)
                task_kwargs: Dict[str, Any] = {
                    "task_id": task_id,
                    "example_id": example_id,
                    "iteration": iteration,
                    "messages": request.messages,
                    "mock_output": (
                        _mock_output(role, iteration)
                        if backend_name == "mock"
                        else None
                    ),
                }
                # Newer protocol versions let the run-level retry journal cap a
                # resumed stage.  Keep compatibility with the initial dataclass
                # while still preserving the journal in the checkpoint.
                if "infrastructure_retry_limit" in getattr(RoleTask, "__dataclass_fields__", {}):
                    task_kwargs["infrastructure_retry_limit"] = max(
                        0,
                        config.workflow.infrastructure_retry_limit - prior_retries,
                    )
                task = RoleTask(**task_kwargs)
                try:
                    task_result = coordinator.submit(role, task).result()
                except BaseException:
                    with retry_lock:
                        retries[task_id] = max(
                            retries.get(task_id, 0),
                            config.workflow.infrastructure_retry_limit,
                        )
                        persist("%s_infrastructure_failure" % role)
                    raise
                with retry_lock:
                    retries[task_id] = prior_retries + int(
                        task_result.infrastructure_retries
                    )
                return _generation_result(task_result.generation)

            return generate

        def execute_candidate(sql: str) -> ExecutionResult:
            with execution_limit:
                return execute_sql(
                    dataset.database_path(example.db_id), sql, **execution_kwargs
                )

        schema_text = serialize_schema(dataset.get_schema(example.db_id))
        result = run_episode(
            AgentEpisodeRequest(
                example_id=example_id,
                question=example.question,
                serialized_schema=schema_text,
            ),
            planner_generate=generator("planner"),
            coder_generate=generator("coder"),
            verifier_generate=generator("verifier"),
            execute_candidate=execute_candidate,
            stage_callback=lambda event, callback_state: persist(event, callback_state),
            resume_state=state,
        )
        return example.index, result

    pending = [example for example in selected_examples if example.index not in results]
    reporter.start(
        "agent_episodes",
        "Agent episodes",
        len(selected_examples),
        initial=len(results),
    )
    max_in_flight = max(
        1,
        2
        * sum(
            len(pool)
            for pool in (
                config.gpu_pools.planner,
                config.gpu_pools.coder,
                config.gpu_pools.verifier,
            )
        ),
    )
    iterator = iter(pending)
    active: Dict[Future, SpiderExample] = {}
    with ThreadPoolExecutor(
        max_workers=max_in_flight,
        thread_name_prefix="agent-episode",
    ) as executor:
        for example in _take(iterator, max_in_flight):
            active[executor.submit(run_one, example)] = example
        while active:
            done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in done:
                example = active.pop(future)
                index, result = future.result()
                if index != example.index or result.example_id != "%s:%d" % (
                    example.split,
                    example.index,
                ):
                    raise RuntimeError("episode worker changed example identity")
                results[index] = result
                reporter.update(len(results))
                next_example = next(iterator, None)
                if next_example is not None:
                    active[executor.submit(run_one, next_example)] = next_example
    reporter.update(len(selected_examples))
    reporter.finish()
    return results


def _take(iterator: Iterable[SpiderExample], count: int) -> List[SpiderExample]:
    result = []
    source = iter(iterator)
    for _ in range(count):
        item = next(source, None)
        if item is None:
            break
        result.append(item)
    return result


def run_agent_evaluation(
    config: AgentAppConfig,
    *,
    backend_name: str,
    allow_model_download: bool = False,
    run_name: Optional[str] = None,
    resume_run: Optional[str] = None,
    selection: str = "all",
    progress: Optional[ProgressReporter] = None,
    invocation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run independent planner-coder-verifier episodes and final Spider evaluation."""

    if backend_name not in {"mock", "hf"}:
        raise ValueError("backend_name must be mock or hf")
    if run_name is not None and resume_run is not None:
        raise ValueError("run_name and resume_run are mutually exclusive")
    local_checkpoints: Dict[str, Any] = {}
    inspection_cache: Dict[str, Any] = {}
    if backend_name == "hf":
        role_names = ("planner", "coder", "verifier")
        validate_download_policy(
            tuple(config.roles[role].model for role in role_names),
            allow_model_download,
        )
        prepared_roles = dict(config.roles)
        prepared_path_identities: Dict[str, str] = {}
        for role in role_names:
            role_model = config.roles[role].model
            cached_identity = prepared_path_identities.get(role_model.model_id)
            if role_model.source == "local" and cached_identity is not None:
                prepared_model = replace(
                    role_model, checkpoint_identity=cached_identity
                )
                inspection = inspection_cache[cached_identity]
            else:
                prepared_model, inspection = prepare_model_config(role_model)
            prepared_roles[role] = replace(
                config.roles[role], model=prepared_model
            )
            if inspection is not None:
                identity = str(inspection["identity"])
                inspection_cache.setdefault(identity, inspection)
                prepared_path_identities[role_model.model_id] = identity
                local_checkpoints[role] = {
                    "identity": identity,
                    "path": inspection["path"],
                }
        config = replace(config, roles=prepared_roles)
    started_wall = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    reporter = progress if progress is not None else ProgressReporter(enabled=False)
    dataset = SpiderDataset(config.spider)
    validation = dataset.validate()
    if not validation.ok:
        raise SpiderDataError(
            "Spider validation failed:\n- " + "\n- ".join(validation.errors)
        )
    selected = _selected_examples(dataset, config, selection)
    if not selected:
        raise SpiderDataError("selected Spider split is empty")
    total = len(selected)
    selected_ids = ["%s:%d" % (example.split, example.index) for example in selected]

    official_preflight: Optional[Dict[str, Any]] = None
    if config.official_evaluation.enabled:
        official_preflight = validate_official_environment(
            evaluator_root=config.official_evaluation.evaluator_root,
            database_root=config.official_evaluation.test_suite_database_root,
            tables_path=dataset.tables_path,
            expected_commit=config.official_evaluation.upstream_commit,
            nltk_data_dir=config.official_evaluation.nltk_data_dir,
            db_ids=[example.db_id for example in selected],
        )
        if not official_preflight["ok"]:
            raise OfficialEvaluationError(
                "Official evaluator preflight failed:\n- "
                + "\n- ".join(official_preflight["errors"])
            )

    source = source_snapshot()
    prompt_and_schema_contract = {
        "prompt_builder_sha256": source["python_files_sha256"][
            "multi_turn_agent/prompts.py"
        ],
        "planner_verifier_schema_sha256": source["python_files_sha256"][
            "multi_turn_agent/contracts.py"
        ],
        "schema_serializer_sha256": source["python_files_sha256"]["core/schema.py"],
    }
    prompt_and_schema_contract["sha256"] = sha256_bytes(
        json_bytes(prompt_and_schema_contract)
    )
    effective_config = effective_config_payload(config)
    experiment_config = _experiment_config_payload(config)
    experiment_config_sha256 = sha256_bytes(json_bytes(experiment_config))
    agent_contract = _agent_contract(config, selection)
    official_preflight_sha256 = (
        sha256_bytes(json_bytes(official_preflight))
        if official_preflight is not None
        else None
    )
    dataset_contract = {
        "split": config.spider.split,
        "selection": selection,
        "example_file_sha256": sha256_file(dataset.examples_path),
        "tables_file_sha256": sha256_file(dataset.tables_path),
        "example_ids": selected_ids,
    }
    dataset_contract_sha256 = sha256_bytes(json_bytes(dataset_contract))
    db_paths = {
        db_id: dataset.database_path(db_id)
        for db_id in sorted({example.db_id for example in selected})
    }

    if resume_run is not None:
        if SAFE_RUN_NAME.fullmatch(resume_run) is None:
            raise ValueError("resume run contains unsafe characters")
        run_dir = (config.output.directory.resolve() / resume_run).resolve()
        try:
            run_dir.relative_to(config.output.directory.resolve())
        except ValueError as exc:
            raise ValueError("resume run directory escapes the output root") from exc
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("resume manifest does not exist: %s" % manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise RuntimeError("resume manifest is not an object")
        run_id = str(manifest.get("run_id"))
        _validate_resume_manifest(
            manifest,
            backend_name=backend_name,
            selection=selection,
            total=total,
            experiment_config_sha256=experiment_config_sha256,
            source_tree_sha256=source["python_tree_sha256"],
            agent_contract_sha256=agent_contract["sha256"],
            dataset_contract_sha256=dataset_contract_sha256,
            official_preflight_sha256=official_preflight_sha256,
        )
        manifest.setdefault("resume_invocations", []).append(dict(invocation or {}))
        manifest.setdefault("gpu_pool_history", []).append(
            {
                "resumed_at": started_wall.isoformat(),
                "gpu_pools": {
                    "planner": list(config.gpu_pools.planner),
                    "coder": list(config.gpu_pools.coder),
                    "verifier": list(config.gpu_pools.verifier),
                },
            }
        )
        database_hashes_before = manifest["dataset"]["selected_database_sha256_before"]
        current_database_hashes = {
            db_id: sha256_file(path) for db_id, path in db_paths.items()
        }
        if current_database_hashes != database_hashes_before:
            raise RuntimeError(
                "refusing to resume because one or more original Spider databases changed"
            )
    else:
        run_id, run_dir = run_directory(config.output.directory, run_name)
        database_hashes_before = {
            db_id: sha256_file(path) for db_id, path in db_paths.items()
        }
        manifest = {
            "schema_version": 2,
            "run_id": run_id,
            "run_type": (
                "agent_smoke" if selection == "smoke" else "agent_evaluation"
            ),
            "status": "initializing",
            "started_at": started_wall.isoformat(),
            "invocation": dict(invocation or {"interface": "python_api"}),
            "backend_selector": backend_name,
            "config_source_path": str(config.source_path),
            "source_config": config.raw,
            "config": effective_config,
            "experiment_config": experiment_config,
            "experiment_config_sha256": experiment_config_sha256,
            "agent_contract": agent_contract["payload"],
            "agent_contract_sha256": agent_contract["sha256"],
            "prompt_and_schema_contract": prompt_and_schema_contract,
            "gpu_pool_history": [
                {
                    "started_at": started_wall.isoformat(),
                    "gpu_pools": {
                        "planner": list(config.gpu_pools.planner),
                        "coder": list(config.gpu_pools.coder),
                        "verifier": list(config.gpu_pools.verifier),
                    },
                }
            ],
            "runtime": {
                "python_version": os.sys.version.split()[0],
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "sqlite_version": sqlite3.sqlite_version,
            },
            "source": source,
            "local_checkpoints": {
                "by_role": local_checkpoints,
                "unique": list(inspection_cache.values()),
            },
            "roles": {
                role: {
                    "model_source": config.roles[role].model.source,
                    "model_id": config.roles[role].model.model_id,
                    "requested_revision": config.roles[role].model.revision,
                    "checkpoint_identity": (
                        config.roles[role].model.checkpoint_identity
                    ),
                    "trainable": config.roles[role].trainable,
                    "resolved_revisions": [],
                }
                for role in ("planner", "coder", "verifier")
            },
            "official_evaluation": {
                "enabled": config.official_evaluation.enabled,
                "preflight": official_preflight,
                "preflight_sha256": official_preflight_sha256,
            },
            "dataset": {
                "split": config.spider.split,
                "selection": selection,
                "total_examples": total,
                "contract": dataset_contract,
                "contract_sha256": dataset_contract_sha256,
                "validation": validation.to_dict(),
                "selected_database_sha256_before": database_hashes_before,
            },
        }
    manifest["status"] = "agent_inference"
    atomic_json(run_dir / "run_manifest.json", compact_run_manifest(manifest))

    loaded_checkpoints: Dict[int, Mapping[str, Any]] = {}
    completed_results: Dict[int, EpisodeResult] = {}
    for example in selected:
        checkpoint = _load_episode_checkpoint(
            _episode_checkpoint_path(run_dir, example.index),
            run_id=run_id,
            example=example,
            contract_sha256=agent_contract["sha256"],
        )
        if checkpoint is None:
            continue
        loaded_checkpoints[example.index] = checkpoint
        state = EpisodeCheckpoint.from_dict(checkpoint["state"])
        if state.next_stage == "complete":
            assert state.result is not None
            completed_results[example.index] = state.result

    coordinator: Optional[AgentWorkerCoordinator] = None
    signal_handlers: Mapping[int, Any] = {}
    worker_metadata: Mapping[str, Any] = manifest.get("worker_runtime", {})
    episode_results: Dict[int, EpisodeResult] = dict(completed_results)
    try:
        if len(completed_results) != total:
            specs = _worker_specs(config, backend_name, allow_model_download)

            def worker_event(event: Mapping[str, Any]) -> None:
                event_name = event.get("event")
                if event_name in {"worker_ready", "worker_retry", "worker_starting"}:
                    reporter.message(
                        "%s role=%s worker=%s gpu=%s"
                        % (
                            event_name,
                            event.get("role"),
                            event.get("worker_id"),
                            event.get("physical_gpu", "-"),
                        )
                    )

            coordinator = AgentWorkerCoordinator(
                specs,
                run_dir / "worker_runtime",
                event_callback=worker_event,
            )
            signal_handlers = _install_worker_cleanup_handlers(coordinator)
            manifest["status"] = "loading_role_workers"
            atomic_json(run_dir / "run_manifest.json", compact_run_manifest(manifest))
            coordinator.start()
            startup_metadata = coordinator.metadata()
            startup_resolved = _validate_resolved_revisions(
                config, backend_name, startup_metadata
            )
            manifest["status"] = "agent_inference"
            manifest["worker_runtime"] = startup_metadata
            for role in ("planner", "coder", "verifier"):
                manifest["roles"][role]["resolved_revisions"] = list(
                    startup_resolved.get(role, ())
                )
            atomic_json(run_dir / "run_manifest.json", compact_run_manifest(manifest))
            episode_results = _run_episode_tasks(
                config=config,
                run_id=run_id,
                run_dir=run_dir,
                selected_examples=selected,
                dataset=dataset,
                coordinator=coordinator,
                backend_name=backend_name,
                contract_sha256=agent_contract["sha256"],
                loaded_checkpoints=loaded_checkpoints,
                completed_results=completed_results,
                reporter=reporter,
            )
            # Primary SQL timing is forbidden until every role process exits.
            coordinator.close(wait=True)
            worker_metadata = coordinator.metadata()
            _restore_signal_handlers(signal_handlers)
            signal_handlers = {}
            coordinator = None
            resolved = _validate_resolved_revisions(
                config, backend_name, worker_metadata
            )
            manifest["worker_runtime"] = worker_metadata
            for role in ("planner", "coder", "verifier"):
                manifest["roles"][role]["resolved_revisions"] = list(
                    resolved.get(role, ())
                )
            atomic_json(run_dir / "run_manifest.json", compact_run_manifest(manifest))
        if len(episode_results) != total:
            raise RuntimeError("not every selected example has a terminal trajectory")
    except BaseException as exc:
        if coordinator is not None:
            try:
                coordinator.terminate()
                worker_metadata = coordinator.metadata()
                manifest["worker_runtime"] = worker_metadata
            except Exception:
                pass
        _restore_signal_handlers(signal_handlers)
        signal_handlers = {}
        reporter.finish()
        _write_failure(
            run_dir=run_dir,
            manifest=manifest,
            stage="agent_inference",
            exc=exc,
            started_wall=started_wall,
            started_monotonic=started_monotonic,
            processed_examples=len(episode_results),
            total_examples=total,
        )
        raise RuntimeError(
            "Agent inference failed; resume artifacts are in %s: %s"
            % (run_dir, exc)
        ) from exc

    trajectories = []
    for example in selected:
        result = episode_results[example.index]
        schema_text = serialize_schema(dataset.get_schema(example.db_id))
        trajectories.append(
            _trajectory_payload(run_id, example, schema_text, result)
        )
    atomic_jsonl(run_dir / "trajectories.jsonl", trajectories)

    # No model process remains alive below this line.  Prediction and gold are
    # executed serially in the parent-defined order for comparable primary timing.
    manifest["status"] = "final_sql_execution"
    atomic_json(run_dir / "run_manifest.json", compact_run_manifest(manifest))
    records_path = run_dir / "records.jsonl"
    records = _read_jsonl(records_path)
    if len(records) > total:
        raise RuntimeError("records.jsonl contains too many rows")
    for position, record in enumerate(records):
        expected = selected[position]
        trajectory = trajectories[position]
        expected_fields = {
            "example_id": "%s:%d" % (expected.split, expected.index),
            "split": expected.split,
            "index": expected.index,
            "db_id": expected.db_id,
            "question": expected.question,
            "gold_sql": expected.gold_sql,
            "final_sql": trajectory.get("final_sql"),
            "termination_reason": trajectory.get("termination_reason"),
            "iterations_used": len(trajectory.get("iterations", [])),
        }
        mismatched = [
            key for key, value in expected_fields.items() if record.get(key) != value
        ]
        if mismatched:
            raise RuntimeError(
                "records.jsonl does not match dataset/trajectory at position %d: %s"
                % (position, ", ".join(mismatched))
            )
        official = record.get("official_evaluation")
        if isinstance(official, Mapping) and (
            official.get("example_id") != expected_fields["example_id"]
            or official.get("db_id") != expected.db_id
        ):
            raise RuntimeError(
                "records.jsonl contains mismatched official evaluation identity"
            )
    reporter.start(
        "final_sql_execution",
        "Final prediction and gold SQL execution",
        total,
        initial=len(records),
    )
    execution_kwargs = _execution_args(config)
    try:
        with records_path.open("a", encoding="utf-8") as handle:
            for position in range(len(records), total):
                example = selected[position]
                trajectory = trajectories[position]
                final_sql = trajectory.get("final_sql")
                db_path = dataset.database_path(example.db_id)
                if isinstance(final_sql, str) and final_sql.strip():
                    predicted = execute_sql(db_path, final_sql, **execution_kwargs)
                else:
                    final_parsing = trajectory.get("final_sql_parsing")
                    predicted = _not_run(
                        str(
                            final_parsing.get("error_type", "prediction_unavailable")
                            if isinstance(final_parsing, Mapping)
                            else "prediction_unavailable"
                        ),
                        str(
                            final_parsing.get("error_message", "No final SQL candidate")
                            if isinstance(final_parsing, Mapping)
                            else "No final SQL candidate"
                        ),
                    )
                gold = execute_sql(db_path, example.gold_sql, **execution_kwargs)
                comparison = compare_results(
                    predicted,
                    gold,
                    order_sensitive=bool(example.parsed_sql.get("orderBy")),
                )
                comparison.update(
                    {
                        "metric": "local_single_database_result_match",
                        "official_spider_metric": False,
                    }
                )
                record = {
                    "schema_version": 1,
                    "run_id": run_id,
                    "example_id": "%s:%d" % (example.split, example.index),
                    "split": example.split,
                    "index": example.index,
                    "db_id": example.db_id,
                    "question": example.question,
                    "termination_reason": trajectory["termination_reason"],
                    "iterations_used": len(trajectory["iterations"]),
                    "final_raw_output": trajectory["final_raw_output"],
                    "final_sql": final_sql,
                    "final_sql_parsing": trajectory["final_sql_parsing"],
                    "gold_sql": example.gold_sql,
                    "predicted_execution": execution_payload(predicted),
                    "gold_execution": execution_payload(gold),
                    "comparison": comparison,
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                handle.flush()
                records.append(record)
                reporter.update(len(records))
        reporter.update(total)
        reporter.finish()
    except BaseException as exc:
        reporter.finish()
        _write_failure(
            run_dir=run_dir,
            manifest=manifest,
            stage="final_sql_execution",
            exc=exc,
            started_wall=started_wall,
            started_monotonic=started_monotonic,
            processed_examples=len(records),
            total_examples=total,
        )
        raise RuntimeError(
            "Final SQL execution failed; resume artifacts are in %s: %s"
            % (run_dir, exc)
        ) from exc

    manifest["status"] = "official_evaluation"
    atomic_json(run_dir / "run_manifest.json", compact_run_manifest(manifest))
    try:
        if config.official_evaluation.enabled:
            if resume_run is not None:
                for record in records:
                    current = record.get("official_evaluation")
                    if not isinstance(current, Mapping):
                        continue
                    exact_status = current.get("exact_set_match", {}).get("status")
                    suite_status = current.get("test_suite", {}).get("status")
                    if (
                        exact_status in _INFRASTRUCTURE_OFFICIAL_STATUSES
                        or suite_status in _INFRASTRUCTURE_OFFICIAL_STATUSES
                    ):
                        record.pop("official_evaluation", None)
            pending_positions = [
                position
                for position, record in enumerate(records)
                if "official_evaluation" not in record
            ]
            reporter.start(
                "official_evaluation",
                "Official Spider evaluation",
                total,
                initial=total - len(pending_positions),
            )
            previous_result = manifest.get("official_evaluation", {}).get(
                "result", {}
            )
            official_policy: Optional[Mapping[str, Any]] = (
                previous_result.get("policy")
                if isinstance(previous_result, Mapping)
                else None
            )
            official_evaluator: Optional[Mapping[str, Any]] = (
                previous_result.get("evaluator")
                if isinstance(previous_result, Mapping)
                else None
            )
            for start in range(0, len(pending_positions), 8):
                positions = pending_positions[start : start + 8]
                batch = evaluate_official(
                    [
                        OfficialEvaluationItem(
                            example_id=records[position]["example_id"],
                            db_id=records[position]["db_id"],
                            gold_sql=records[position]["gold_sql"],
                            predicted_sql=(
                                records[position]["final_sql"]
                                if isinstance(records[position].get("final_sql"), str)
                                else None
                            ),
                        )
                        for position in positions
                    ],
                    evaluator_root=config.official_evaluation.evaluator_root,
                    database_root=config.official_evaluation.test_suite_database_root,
                    tables_path=dataset.tables_path,
                    expected_commit=config.official_evaluation.upstream_commit,
                    nltk_data_dir=config.official_evaluation.nltk_data_dir,
                    timeout_seconds=config.official_evaluation.timeout_seconds,
                    max_sql_bytes=config.execution.max_sql_bytes,
                    worker_memory_limit_bytes=config.execution.worker_memory_limit_bytes,
                    plug_value=config.official_evaluation.plug_value,
                    keep_distinct=config.official_evaluation.keep_distinct,
                    preflight_report=official_preflight,
                )
                if len(batch["results"]) != len(positions):
                    raise OfficialEvaluationError(
                        "official evaluator returned the wrong batch size"
                    )
                for position, official_result in zip(positions, batch["results"]):
                    if official_result["example_id"] != records[position]["example_id"]:
                        raise OfficialEvaluationError(
                            "official evaluator changed example ordering"
                        )
                    records[position]["official_evaluation"] = official_result
                official_policy = batch["policy"]
                official_evaluator = batch["evaluator"]
                atomic_jsonl(records_path, records)
                reporter.update(
                    sum("official_evaluation" in record for record in records)
                )
            reporter.update(total)
            reporter.finish()
            manifest["official_evaluation"]["result"] = {
                "processed_examples": len(records),
                "batch_size": 8,
                "policy": official_policy,
                "evaluator": official_evaluator,
            }
        else:
            for record in records:
                record["official_evaluation"] = {
                    "status": "disabled",
                    "exact_set_match": None,
                    "test_suite": None,
                }
            atomic_jsonl(records_path, records)
    except BaseException as exc:
        reporter.finish()
        _write_failure(
            run_dir=run_dir,
            manifest=manifest,
            stage="official_evaluation",
            exc=exc,
            started_wall=started_wall,
            started_monotonic=started_monotonic,
            processed_examples=len(records),
            total_examples=total,
        )
        raise RuntimeError(
            "Official evaluation failed; resume artifacts are in %s: %s"
            % (run_dir, exc)
        ) from exc

    database_hashes_after = {
        db_id: sha256_file(path) for db_id, path in db_paths.items()
    }
    databases_unchanged = database_hashes_before == database_hashes_after
    termination_reasons = Counter(
        trajectory["termination_reason"] for trajectory in trajectories
    )
    iteration_distribution = Counter(
        str(len(trajectory["iterations"])) for trajectory in trajectories
    )
    final_parsing_statuses = Counter(
        (
            trajectory["final_sql_parsing"].get("status")
            if isinstance(trajectory.get("final_sql_parsing"), Mapping)
            else "unavailable"
        )
        for trajectory in trajectories
    )
    role_generation_statuses = {
        role: Counter() for role in ("planner", "coder", "verifier")
    }
    role_generation_error_types = {
        role: Counter() for role in ("planner", "coder", "verifier")
    }
    role_contract_error_counts = Counter()
    iteration_sql_parsing_statuses = Counter()
    for trajectory in trajectories:
        for iteration in trajectory["iterations"]:
            parsing = iteration.get("sql_parsing")
            if isinstance(parsing, Mapping):
                iteration_sql_parsing_statuses[str(parsing.get("status"))] += 1
            for role in ("planner", "coder", "verifier"):
                trace = iteration.get(role)
                if not isinstance(trace, Mapping):
                    continue
                generation = trace.get("generation")
                if isinstance(generation, Mapping):
                    role_generation_statuses[role][
                        str(generation.get("status"))
                    ] += 1
                    if generation.get("error_type") is not None:
                        role_generation_error_types[role][
                            str(generation.get("error_type"))
                        ] += 1
                if trace.get("contract_error") is not None:
                    role_contract_error_counts[role] += 1
    prediction_statuses = Counter(
        record["predicted_execution"]["status"] for record in records
    )
    gold_statuses = Counter(record["gold_execution"]["status"] for record in records)
    local_matches = sum(
        bool(record["comparison"].get("result_match")) for record in records
    )
    if config.official_evaluation.enabled:
        exact = _official_metric_summary(records, "exact_set_match")
        test_suite = _official_metric_summary(records, "test_suite")
        official_ok: Optional[bool] = exact["valid"] and test_suite["valid"]
    else:
        exact = {
            "valid": None,
            "classified_examples": None,
            "infrastructure_failures": None,
            "statuses": {},
            "matches": None,
            "accuracy": None,
        }
        test_suite = dict(exact)
        official_ok = None
    minimum_executable = (
        config.smoke.minimum_executable if selection == "smoke" else 0
    )
    pipeline_pass = (
        len(records) == total
        and len(trajectories) == total
        and gold_statuses.get("success", 0) == total
        and prediction_statuses.get("success", 0) >= minimum_executable
        and databases_unchanged
        and official_ok is not False
    )
    finished_wall = datetime.now(timezone.utc)
    prediction_query_timing = query_timing_summary(
        records, "predicted_execution"
    )
    prediction_vm_steps = vm_step_summary(records, "predicted_execution")
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "run_type": manifest["run_type"],
        "split": config.spider.split,
        "selection": selection,
        "backend": backend_name,
        "pipeline_pass": pipeline_pass,
        "accuracy_measurement": backend_name != "mock",
        "total_examples": total,
        "processed_examples": len(records),
        "terminal_trajectories": len(trajectories),
        "termination_reasons": dict(termination_reasons),
        "iterations_used_distribution": dict(iteration_distribution),
        "role_generation_statuses": {
            role: dict(counter)
            for role, counter in role_generation_statuses.items()
        },
        "role_generation_error_types": {
            role: dict(counter)
            for role, counter in role_generation_error_types.items()
        },
        "role_contract_error_counts": dict(role_contract_error_counts),
        "iteration_sql_parsing_statuses": dict(iteration_sql_parsing_statuses),
        "final_sql_parsing_statuses": dict(final_parsing_statuses),
        "prediction_execution_statuses": dict(prediction_statuses),
        "gold_execution_statuses": dict(gold_statuses),
        "executable_predictions": prediction_statuses.get("success", 0),
        "minimum_executable_required": minimum_executable,
        "official_spider_metric": config.official_evaluation.enabled,
        "official_evaluation_ok": official_ok,
        "exact_set_match_valid": exact["valid"],
        "exact_set_match_classified_examples": exact["classified_examples"],
        "exact_set_match_infrastructure_failures": exact["infrastructure_failures"],
        "exact_set_match_statuses": exact["statuses"],
        "exact_set_matches": exact["matches"],
        "exact_set_match_accuracy": exact["accuracy"],
        "test_suite_accuracy_valid": test_suite["valid"],
        "test_suite_classified_examples": test_suite["classified_examples"],
        "test_suite_infrastructure_failures": test_suite["infrastructure_failures"],
        "test_suite_statuses": test_suite["statuses"],
        "test_suite_matches": test_suite["matches"],
        "test_suite_accuracy": test_suite["accuracy"],
        "local_result_matches": local_matches,
        "local_result_match_rate": local_matches / total,
        "local_result_match_policy": (
            "single original Spider DB; diagnostic only; not an official metric"
        ),
        "tool_query_timing": _tool_timing_summary(trajectories),
        "tool_vm_steps": _tool_vm_step_summary(trajectories),
        "prediction_query_timing": prediction_query_timing,
        "gold_query_timing": query_timing_summary(records, "gold_execution"),
        "prediction_vm_steps": prediction_vm_steps,
        "gold_vm_steps": vm_step_summary(records, "gold_execution"),
        "primary_timing_cache_caveat": (
            "Agent tool executions may have warmed the database/OS page cache; "
            "no explicit cache reset was performed."
        ),
        "selected_databases_unchanged": databases_unchanged,
        "started_at": started_wall.isoformat(),
        "finished_at": finished_wall.isoformat(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
    }
    if backend_name == "mock":
        summary["notice"] = (
            "Scripted role outputs validate orchestration only and are not model accuracy."
        )
    atomic_json(
        run_dir / "summary.json",
        metric_summary(
            test_suite_accuracy=test_suite["accuracy"],
            exact_set_match_accuracy=exact["accuracy"],
            result_match_accuracy=local_matches / total,
            prediction_vm_steps=prediction_vm_steps,
            prediction_query_timing=prediction_query_timing,
        ),
    )
    manifest["status"] = "completed" if pipeline_pass else "failed"
    manifest["finished_at"] = finished_wall.isoformat()
    manifest["elapsed_seconds"] = summary["elapsed_seconds"]
    manifest["dataset"]["selected_database_sha256_after"] = database_hashes_after
    manifest["dataset"]["selected_databases_unchanged"] = databases_unchanged
    manifest["artifacts"] = {
        "episode_checkpoints": "episodes/",
        "trajectories": "trajectories.jsonl",
        "records": "records.jsonl",
        "summary": "summary.json",
        "worker_runtime": "worker_runtime/",
    }
    atomic_json(run_dir / "run_manifest.json", compact_run_manifest(manifest))
    return {"run_directory": str(run_dir), "summary": summary}
