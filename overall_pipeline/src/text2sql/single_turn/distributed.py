from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from text2sql.config import AppConfig, load_config
from text2sql.core.models import GenerationRequest, GenerationResult, ParseResult
from text2sql.core.model_source import LOCAL_IDENTITY_PREFIX
from text2sql.single_turn.prompt import build_messages
from text2sql.single_turn.smoke_runner import (
    _atomic_json,
    _create_backend,
    _json_bytes,
    _sha256_bytes,
    _sha256_file,
    _source_snapshot,
)
from text2sql.core.schema import serialize_schema
from text2sql.core.spider import SpiderDataError, SpiderDataset
from text2sql.core.sql_output import extract_sql


class DistributedGenerationError(RuntimeError):
    """Raised when a generation shard cannot be produced or merged safely."""


def parse_gpu_ids(value: str) -> Tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part or not part.isdigit() for part in parts):
        raise ValueError("--gpus must be a comma-separated list of GPU indices")
    gpu_ids = tuple(int(part) for part in parts)
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpus must not contain duplicate GPU indices")
    return gpu_ids


def round_robin_shards(
    example_indices: Sequence[int], worker_count: int
) -> List[List[int]]:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    shards: List[List[int]] = [[] for _ in range(worker_count)]
    for position, index in enumerate(example_indices):
        shards[position % worker_count].append(index)
    return shards


def generation_contract(config: AppConfig, dataset: SpiderDataset) -> Dict[str, Any]:
    payload = {
        "split": config.spider.split,
        "examples_file": str(dataset.examples_path.resolve()),
        "examples_file_sha256": _sha256_file(dataset.examples_path),
        "tables_file": str(dataset.tables_path.resolve()),
        "tables_file_sha256": _sha256_file(dataset.tables_path),
        "model": {
            "source": config.model.source,
            "id": config.model.model_id,
            "revision": config.model.revision,
            "checkpoint_identity": config.model.checkpoint_identity,
            "dtype": config.model.dtype,
            "device": "cuda:0",
            "attention_implementation": config.model.attention_implementation,
            "trust_remote_code": config.model.trust_remote_code,
            "cache_dir": (
                str(config.model.cache_dir)
                if config.model.cache_dir is not None
                else None
            ),
        },
        "generation": {
            "do_sample": config.generation.do_sample,
            "num_beams": config.generation.num_beams,
            "repetition_penalty": config.generation.repetition_penalty,
            "max_time_seconds": config.generation.max_time_seconds,
            "max_input_tokens": config.generation.max_input_tokens,
            "max_new_tokens": config.generation.max_new_tokens,
            "batch_size": config.generation.batch_size,
        },
    }
    return {
        "payload": payload,
        "sha256": _sha256_bytes(_json_bytes(payload)),
    }


def _write_jsonl_record(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
    handle.write("\n")
    handle.flush()


def _compact_generation_payload(result: GenerationResult) -> Dict[str, Any]:
    return {
        "status": result.status,
        "elapsed_seconds": result.elapsed_seconds,
        "raw_output": result.raw_output,
        "error_type": result.error_type,
        "error_message": result.error_message,
    }


def run_generation_worker(
    *,
    config_path: Path,
    assignment_path: Path,
    shard_path: Path,
    status_path: Path,
    backend_name: str,
    allow_model_download: bool,
) -> Dict[str, Any]:
    """Generate one assigned shard without opening or executing any SQLite DB."""

    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    if not isinstance(assignment, dict):
        raise DistributedGenerationError("worker assignment must be a JSON object")
    indices = assignment.get("example_indices")
    if not isinstance(indices, list) or any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in indices
    ):
        raise DistributedGenerationError("worker assignment has invalid indices")
    if len(set(indices)) != len(indices):
        raise DistributedGenerationError("worker assignment contains duplicate indices")

    config = load_config(config_path)
    if config.model.source == "local":
        identity = assignment.get("model_checkpoint_identity")
        if not isinstance(identity, str) or not identity.startswith(
            LOCAL_IDENTITY_PREFIX
        ):
            raise DistributedGenerationError(
                "worker assignment is missing the local checkpoint identity"
            )
        config = replace(
            config,
            model=replace(config.model, checkpoint_identity=identity),
        )
    config = replace(config, model=replace(config.model, device="cuda:0"))
    dataset = SpiderDataset(config.spider)
    contract = generation_contract(config, dataset)
    if contract["sha256"] != assignment.get("generation_contract_sha256"):
        raise DistributedGenerationError(
            "worker generation contract does not match the parent"
        )
    source = _source_snapshot()
    if source["python_tree_sha256"] != assignment.get("source_tree_sha256"):
        raise DistributedGenerationError("worker source tree does not match the parent")

    worker_id = str(assignment.get("worker_id"))
    physical_gpu = assignment.get("physical_gpu")
    status: Dict[str, Any] = {
        "schema_version": 1,
        "run_id": assignment.get("run_id"),
        "attempt": assignment.get("attempt"),
        "worker_id": worker_id,
        "physical_gpu": physical_gpu,
        "logical_device": "cuda:0",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "backend_selector": backend_name,
        "assigned_examples": len(indices),
        "completed_examples": 0,
        "status": "initializing_backend",
        "started_at_ns": time.time_ns(),
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(status_path, status)

    examples = []
    for index in indices:
        example = dataset.get_example(index)
        if example.split != assignment.get("split"):
            raise DistributedGenerationError(
                "worker split mismatch at index %d" % index
            )
        examples.append(example)
    mock_responses = {
        "%s:%d" % (example.split, example.index): example.gold_sql
        for example in examples
    }
    backend = None
    try:
        backend = _create_backend(
            backend_name,
            config,
            mock_responses=mock_responses,
            allow_model_download=allow_model_download,
        )
        status["backend"] = backend.metadata()
        status["status"] = "generating"
        _atomic_json(status_path, status)
        with shard_path.open("w", encoding="utf-8") as shard_handle:
            for example in examples:
                example_id = "%s:%d" % (example.split, example.index)
                schema_text = serialize_schema(dataset.get_schema(example.db_id))
                messages = build_messages(example.question, schema_text)
                request = GenerationRequest(
                    example_id=example_id, messages=tuple(messages)
                )
                try:
                    generation = backend.generate(request)
                except Exception as exc:
                    generation = GenerationResult(
                        status="error",
                        error_type="backend_unhandled_error",
                        error_message="%s: %s" % (type(exc).__name__, exc),
                        model_id=config.model.model_id,
                        requested_revision=config.model.revision,
                    )
                if generation.status == "success":
                    parsed = extract_sql(generation.raw_output)
                else:
                    parsed = ParseResult(
                        status="error",
                        error_type="generation_failed",
                        error_message=(
                            "SQL parsing skipped because generation failed"
                        ),
                    )
                record = {
                    "schema_version": 4,
                    "run_id": assignment.get("run_id"),
                    "example_id": example_id,
                    "split": example.split,
                    "index": example.index,
                    "db_id": example.db_id,
                    "question": example.question,
                    "prompt_sha256": _sha256_bytes(_json_bytes(messages)),
                    "generation": _compact_generation_payload(generation),
                    "sql_parsing": parsed.to_dict(),
                    "generation_contract_sha256": contract["sha256"],
                }
                _write_jsonl_record(shard_handle, record)
                status["completed_examples"] += 1
                _atomic_json(status_path, status)
        status["backend"] = backend.metadata()
        status["status"] = "completed"
        status["finished_at_ns"] = time.time_ns()
        _atomic_json(status_path, status)
        return status
    except BaseException as exc:
        status["status"] = "failed"
        status["finished_at_ns"] = time.time_ns()
        status["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc)[:2000],
        }
        _atomic_json(status_path, status)
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass


def load_generation_shards(
    shards_root: Path,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    records: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    if not shards_root.is_dir():
        return records, warnings
    for path in sorted(shards_root.glob("attempt-*/*.jsonl")):
        attempt_match = re.fullmatch(r"attempt-([0-9]+)", path.parent.name)
        if attempt_match is None:
            raise DistributedGenerationError(
                "generation shard is outside an attempt-NNN directory: %s" % path
            )
        artifact_attempt = int(attempt_match.group(1))
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if line_number == len(lines):
                    warnings.append(
                        "ignored incomplete final JSONL line in %s" % path
                    )
                    continue
                raise DistributedGenerationError(
                    "invalid JSONL at %s:%d: %s" % (path, line_number, exc)
                ) from exc
            if not isinstance(record, dict) or not isinstance(
                record.get("example_id"), str
            ):
                raise DistributedGenerationError(
                    "invalid generation record at %s:%d" % (path, line_number)
                )
            example_id = record["example_id"]
            record["_artifact_attempt"] = artifact_attempt
            if example_id in records:
                previous = records[example_id]
                previous_attempt = previous.get("_artifact_attempt")
                current_attempt = artifact_attempt
                if (
                    isinstance(previous_attempt, bool)
                    or not isinstance(previous_attempt, int)
                    or isinstance(current_attempt, bool)
                    or not isinstance(current_attempt, int)
                    or current_attempt <= previous_attempt
                ):
                    raise DistributedGenerationError(
                        "duplicate generated example_id without a newer retry: %s"
                        % example_id
                    )
                warnings.append(
                    "selected retry attempt %d over attempt %d for %s"
                    % (current_attempt, previous_attempt, example_id)
                )
            records[example_id] = record
    return records, warnings


def validate_and_order_generation_records(
    records: Mapping[str, Mapping[str, Any]],
    dataset: SpiderDataset,
    expected_indices: Sequence[int],
    run_id: str,
    contract_sha256: str,
) -> List[Dict[str, Any]]:
    expected_ids = [
        "%s:%d" % (dataset.get_example(index).split, index)
        for index in expected_indices
    ]
    actual_ids = set(records)
    missing = [example_id for example_id in expected_ids if example_id not in actual_ids]
    unexpected = sorted(actual_ids - set(expected_ids))
    if missing or unexpected:
        raise DistributedGenerationError(
            "generation shard coverage mismatch: %d missing, %d unexpected"
            % (len(missing), len(unexpected))
        )
    ordered: List[Dict[str, Any]] = []
    for index, example_id in zip(expected_indices, expected_ids):
        record = dict(records[example_id])
        example = dataset.get_example(index)
        expected_fields = {
            "run_id": run_id,
            "split": example.split,
            "index": example.index,
            "db_id": example.db_id,
            "question": example.question,
            "generation_contract_sha256": contract_sha256,
        }
        mismatched = [
            key for key, value in expected_fields.items() if record.get(key) != value
        ]
        if mismatched:
            raise DistributedGenerationError(
                "generation record %s mismatches: %s"
                % (example_id, ", ".join(mismatched))
            )
        record.pop("_artifact_attempt", None)
        ordered.append(record)
    return ordered


def _terminate_processes(processes: Iterable[subprocess.Popen]) -> None:
    live = [process for process in processes if process.poll() is None]
    for process in live:
        process.terminate()
    deadline = time.monotonic() + 10.0
    for process in live:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
    for process in live:
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass


def _wait_for_worker_ready(
    process: subprocess.Popen,
    status_path: Path,
    worker_id: str,
    timeout_seconds: float = 600.0,
    poll_callback: Optional[Callable[[], None]] = None,
) -> None:
    """Wait until one HF worker has loaded its model before loading the next."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if poll_callback is not None:
            poll_callback()
        return_code = process.poll()
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                status = {}
            state = status.get("status")
            if state in {"generating", "completed"}:
                return
            if state == "failed":
                raise DistributedGenerationError(
                    "%s failed while initializing its backend" % worker_id
                )
        if return_code is not None:
            raise DistributedGenerationError(
                "%s exited with code %d before becoming ready"
                % (worker_id, return_code)
            )
        time.sleep(0.2)
    raise DistributedGenerationError(
        "%s did not finish backend initialization within %.0f seconds"
        % (worker_id, timeout_seconds)
    )


def launch_generation_attempt(
    *,
    config_path: Path,
    run_dir: Path,
    run_id: str,
    attempt: int,
    split: str,
    remaining_indices: Sequence[int],
    gpu_ids: Sequence[int],
    backend_name: str,
    allow_model_download: bool,
    generation_contract_sha256: str,
    source_tree_sha256: str,
    model_checkpoint_identity: Optional[str] = None,
    startup_delay_seconds: float = 0.0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    message_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    if not remaining_indices:
        return {"attempt": attempt, "workers": [], "assigned_examples": 0}
    active_gpu_ids = tuple(gpu_ids[: len(remaining_indices)])
    shards = round_robin_shards(remaining_indices, len(active_gpu_ids))
    attempt_dir = run_dir / "shards" / ("attempt-%03d" % attempt)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    processes: List[subprocess.Popen] = []
    log_handles = []
    worker_specs: List[Dict[str, Any]] = []
    last_reported_completed = -1

    def report_progress() -> None:
        nonlocal last_reported_completed
        completed = 0
        for path in attempt_dir.glob("*.status.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            value = payload.get("completed_examples")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                completed += value
        completed = min(completed, len(remaining_indices))
        if completed != last_reported_completed:
            last_reported_completed = completed
            if progress_callback is not None:
                progress_callback(completed, len(remaining_indices))

    report_progress()
    previous_sigterm_handler = None
    signal_handler_installed = False
    try:
        if hasattr(signal, "SIGTERM"):
            try:
                previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

                def handle_sigterm(_signum: int, _frame: Any) -> None:
                    raise DistributedGenerationError(
                        "generation supervisor received SIGTERM"
                    )

                signal.signal(signal.SIGTERM, handle_sigterm)
                signal_handler_installed = True
            except ValueError:
                # Signal handlers may only be installed by the main thread.
                signal_handler_installed = False
        for worker_number, (physical_gpu, indices) in enumerate(
            zip(active_gpu_ids, shards)
        ):
            worker_id = "worker-%02d" % worker_number
            assignment_path = attempt_dir / (worker_id + ".assignment.json")
            shard_path = attempt_dir / (worker_id + ".jsonl")
            status_path = attempt_dir / (worker_id + ".status.json")
            log_path = attempt_dir / (worker_id + ".log")
            assignment = {
                "schema_version": 1,
                "run_id": run_id,
                "attempt": attempt,
                "worker_id": worker_id,
                "physical_gpu": physical_gpu,
                "logical_device": "cuda:0",
                "split": split,
                "example_indices": indices,
                "generation_contract_sha256": generation_contract_sha256,
                "source_tree_sha256": source_tree_sha256,
                "model_checkpoint_identity": model_checkpoint_identity,
            }
            _atomic_json(assignment_path, assignment)
            command = [
                sys.executable,
                "-m",
                "text2sql",
                "_generation-worker",
                "--config",
                str(config_path),
                "--assignment",
                str(assignment_path),
                "--shard",
                str(shard_path),
                "--status",
                str(status_path),
                "--backend",
                backend_name,
            ]
            if allow_model_download:
                command.append("--allow-model-download")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            log_handle = log_path.open("w", encoding="utf-8")
            log_handles.append(log_handle)
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                text=True,
            )
            processes.append(process)
            if message_callback is not None:
                message_callback(
                    "starting %s on physical GPU %d (worker device cuda:0)"
                    % (worker_id, physical_gpu)
                )
            worker_specs.append(
                {
                    "worker_id": worker_id,
                    "physical_gpu": physical_gpu,
                    "logical_device": "cuda:0",
                    "assigned_examples": len(indices),
                    "assignment": str(assignment_path.relative_to(run_dir)),
                    "shard": str(shard_path.relative_to(run_dir)),
                    "status": str(status_path.relative_to(run_dir)),
                    "log": str(log_path.relative_to(run_dir)),
                    "pid": process.pid,
                }
            )
            if backend_name == "hf":
                _wait_for_worker_ready(
                    process,
                    status_path,
                    worker_id,
                    poll_callback=report_progress,
                )
                if message_callback is not None:
                    message_callback(
                        "%s is ready on physical GPU %d"
                        % (worker_id, physical_gpu)
                    )
            if startup_delay_seconds > 0 and worker_number + 1 < len(active_gpu_ids):
                time.sleep(startup_delay_seconds)

        while any(process.poll() is None for process in processes):
            report_progress()
            if any(
                process.poll() not in (None, 0)
                for process in processes
            ):
                _terminate_processes(processes)
                break
            time.sleep(0.2)
        return_codes = [process.wait() for process in processes]
        report_progress()
        for spec, return_code in zip(worker_specs, return_codes):
            spec["return_code"] = return_code
        failures = [
            spec for spec in worker_specs if spec.get("return_code") != 0
        ]
        if failures:
            details = ", ".join(
                "%s(exit=%s, log=%s)"
                % (spec["worker_id"], spec["return_code"], spec["log"])
                for spec in failures
            )
            if message_callback is not None:
                message_callback("generation worker failure: " + details)
            raise DistributedGenerationError(
                "one or more generation workers failed: " + details
            )
    except BaseException as exc:
        if message_callback is not None:
            message_callback(
                "generation attempt %d stopped: %s" % (attempt, exc)
            )
        _terminate_processes(processes)
        raise
    finally:
        if signal_handler_installed:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
        for handle in log_handles:
            handle.close()

    statuses = []
    for spec in worker_specs:
        status_path = run_dir / spec["status"]
        if not status_path.is_file():
            raise DistributedGenerationError(
                "generation worker did not write status: %s" % spec["worker_id"]
            )
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "completed":
            raise DistributedGenerationError(
                "generation worker did not complete: %s" % spec["worker_id"]
            )
        if status.get("completed_examples") != spec["assigned_examples"]:
            raise DistributedGenerationError(
                "generation worker count mismatch: %s" % spec["worker_id"]
            )
        statuses.append(status)
    if progress_callback is not None:
        progress_callback(len(remaining_indices), len(remaining_indices))
    resolved_revisions = {
        status.get("backend", {}).get("resolved_revision")
        for status in statuses
        if backend_name == "hf"
    }
    if backend_name == "hf" and (
        None in resolved_revisions or len(resolved_revisions) != 1
    ):
        raise DistributedGenerationError(
            "HF workers did not resolve one identical model revision"
        )
    return {
        "attempt": attempt,
        "workers": worker_specs,
        "worker_statuses": statuses,
        "assigned_examples": len(remaining_indices),
        "resolved_revision": (
            next(iter(resolved_revisions)) if resolved_revisions else "local"
        ),
    }
