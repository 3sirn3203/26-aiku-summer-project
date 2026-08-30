from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sqlite3
import statistics
import sys
import time
import uuid
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from text2sql import __version__
from text2sql.core.backends.base import GenerationBackend
from text2sql.core.backends.mock import GoldMockBackend
from text2sql.config import AppConfig, SmokeSampleConfig
from text2sql.core.evaluation import compare_results, result_hash
from text2sql.core.executor import execute_sql
from text2sql.core.models import ExecutionResult, GenerationRequest, GenerationResult, ParseResult
from text2sql.core.model_source import prepare_model_config, validate_download_policy
from text2sql.core.official_eval import (
    OfficialEvaluationError,
    OfficialEvaluationItem,
    evaluate_official,
    validate_official_environment,
)
from text2sql.single_turn.prompt import build_messages
from text2sql.core.schema import serialize_schema
from text2sql.core.spider import SpiderDataError, SpiderDataset
from text2sql.core.sql_output import extract_sql


_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(str(temporary), str(path))


def _effective_config_payload(config: AppConfig) -> Dict[str, Any]:
    return {
        "spider": {
            "root": str(config.spider.root),
            "examples_file": config.spider.examples_file,
            "tables_file": config.spider.tables_file,
            "database_dir": config.spider.database_dir,
            "split": config.spider.split,
        },
        "smoke": {
            "samples": [
                {
                    "index": sample.index,
                    "db_id": sample.db_id,
                    "category": sample.category,
                    "question_sha256": sample.question_sha256,
                    "gold_sql_sha256": sample.gold_sql_sha256,
                }
                for sample in config.smoke.samples
            ],
            "minimum_executable": config.smoke.minimum_executable,
        },
        "model": {
            "source": config.model.source,
            "id": config.model.model_id,
            "revision": config.model.revision,
            "checkpoint_identity": config.model.checkpoint_identity,
            "dtype": config.model.dtype,
            "device": config.model.device,
            "attention_implementation": config.model.attention_implementation,
            "trust_remote_code": config.model.trust_remote_code,
            "cache_dir": (
                str(config.model.cache_dir) if config.model.cache_dir is not None else None
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
        "execution": {
            "timeout_seconds": config.execution.timeout_seconds,
            "max_sql_bytes": config.execution.max_sql_bytes,
            "max_result_rows": config.execution.max_result_rows,
            "max_result_bytes": config.execution.max_result_bytes,
            "worker_memory_limit_bytes": config.execution.worker_memory_limit_bytes,
        },
        "official_evaluation": {
            "enabled": config.official_evaluation.enabled,
            "evaluator_root": str(config.official_evaluation.evaluator_root),
            "test_suite_database_root": str(
                config.official_evaluation.test_suite_database_root
            ),
            "upstream_url": config.official_evaluation.upstream_url,
            "upstream_commit": config.official_evaluation.upstream_commit,
            "plug_value": config.official_evaluation.plug_value,
            "keep_distinct": config.official_evaluation.keep_distinct,
            "timeout_seconds": config.official_evaluation.timeout_seconds,
            "nltk_data_dir": (
                str(config.official_evaluation.nltk_data_dir)
                if config.official_evaluation.nltk_data_dir is not None
                else None
            ),
        },
        "output": {"directory": str(config.output.directory)},
    }


def _failure_payload(stage: str, exc: Exception) -> Dict[str, str]:
    return {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc)[:2000],
    }


def _source_snapshot() -> Dict[str, Any]:
    package_root = Path(__file__).resolve().parents[1]
    source_paths = [package_root / "config.py"]
    source_paths.extend((package_root / "core").rglob("*.py"))
    source_paths.extend((package_root / "single_turn").rglob("*.py"))
    file_hashes = {
        str(path.relative_to(package_root)): _sha256_file(path)
        for path in sorted(source_paths)
    }
    return {
        "package_version": __version__,
        "package_root": str(package_root),
        "scope": ["config.py", "core/**/*.py", "single_turn/**/*.py"],
        "python_files_sha256": file_hashes,
        "python_tree_sha256": _sha256_bytes(_json_bytes(file_hashes)),
    }


def _execution_payload(result: ExecutionResult) -> Dict[str, Any]:
    payload = result.to_dict()
    payload["query_elapsed_ms"] = (
        result.query_elapsed_ns / 1_000_000
        if result.query_elapsed_ns is not None
        else None
    )
    payload["worker_elapsed_ms"] = (
        result.worker_elapsed_ns / 1_000_000
        if result.worker_elapsed_ns is not None
        else None
    )
    payload["parent_elapsed_ms"] = (
        result.parent_elapsed_ns / 1_000_000
        if result.parent_elapsed_ns is not None
        else None
    )
    payload["row_count"] = len(result.rows) if result.succeeded else None
    payload["result_hash"] = result_hash(result)
    return payload


def _query_timing_summary(
    records: Sequence[Mapping[str, Any]], execution_key: str
) -> Dict[str, Any]:
    """Summarize successful SQL query intervals for one execution role.

    This intentionally excludes process-launch overhead and every non-success
    result.  It is therefore suitable as an observation for future reward
    work without silently mixing timeouts, failed queries, or evaluator time
    into the database query latency distribution.
    """

    executions = [record[execution_key] for record in records]
    successful = [
        execution for execution in executions if execution["status"] == "success"
    ]
    values_ns = [
        execution["query_elapsed_ns"]
        for execution in successful
        if isinstance(execution.get("query_elapsed_ns"), int)
        and not isinstance(execution["query_elapsed_ns"], bool)
        and execution["query_elapsed_ns"] >= 0
    ]

    def aggregate(values: Sequence[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"min": None, "max": None, "mean": None, "median": None}
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
        }

    values_ms = [value / 1_000_000 for value in values_ns]
    return {
        "total_executions": len(executions),
        "successful_executions": len(successful),
        "count": len(values_ns),
        "excluded_non_success": len(executions) - len(successful),
        "excluded_success_without_query_timing": len(successful) - len(values_ns),
        "query_elapsed_ns": aggregate(values_ns),
        "query_elapsed_ms": aggregate(values_ms),
        "inclusion": "status=success and query_elapsed_ns is available",
    }


def _vm_step_summary(
    records: Sequence[Mapping[str, Any]], execution_key: str
) -> Dict[str, Any]:
    executions = [record[execution_key] for record in records]
    measured = [
        item
        for item in executions
        if isinstance(item.get("vm_steps_lower_bound"), int)
        and not isinstance(item["vm_steps_lower_bound"], bool)
        and item["vm_steps_lower_bound"] >= 0
    ]
    complete = [
        item for item in measured if item.get("vm_step_measurement_complete") is True
    ]
    values = [item["vm_steps_lower_bound"] for item in complete]

    def aggregate(items: Sequence[float]) -> Dict[str, Optional[float]]:
        if not items:
            return {"min": None, "max": None, "mean": None, "median": None}
        return {
            "min": min(items),
            "max": max(items),
            "mean": statistics.fmean(items),
            "median": statistics.median(items),
        }

    intervals = sorted(
        {
            item["vm_step_progress_interval"]
            for item in measured
            if isinstance(item.get("vm_step_progress_interval"), int)
        }
    )
    return {
        "measurement": "progress_handler_bounds_not_exact_stmt_status",
        "total_executions": len(executions),
        "measured_count": len(measured),
        "complete_count": len(complete),
        "censored_count": len(measured) - len(complete),
        "unavailable_count": len(executions) - len(measured),
        "progress_intervals": intervals,
        "complete_vm_steps_lower_bound": aggregate(values),
    }


def _not_run(error_type: str, message: str) -> ExecutionResult:
    return ExecutionResult(
        status="not_run",
        error_type=error_type,
        error_message=message,
    )


def _create_backend(
    backend_name: str,
    config: AppConfig,
    mock_responses: Mapping[str, str],
    allow_model_download: bool,
) -> GenerationBackend:
    if backend_name == "mock":
        return GoldMockBackend(mock_responses)
    if backend_name == "hf":
        # Lazy module import is intentional: local mock runs must not import torch.
        from text2sql.core.backends.hf import HuggingFaceBackend

        return HuggingFaceBackend(
            model_config=config.model,
            generation_config=config.generation,
            allow_model_download=allow_model_download,
        )
    raise ValueError("Unknown backend: %s" % backend_name)


def _selected_examples(
    dataset: SpiderDataset, samples: Sequence[SmokeSampleConfig]
) -> List[Tuple[SmokeSampleConfig, Any]]:
    selected = []
    for sample in samples:
        example = dataset.get_example(sample.index)
        if example.db_id != sample.db_id:
            raise SpiderDataError(
                "Smoke manifest mismatch at dev:%d: expected db_id=%s, found %s"
                % (sample.index, sample.db_id, example.db_id)
            )
        question_digest = _sha256_text(example.question)
        gold_digest = _sha256_text(example.gold_sql)
        if question_digest != sample.question_sha256:
            raise SpiderDataError(
                "Smoke manifest question hash mismatch at dev:%d" % sample.index
            )
        if gold_digest != sample.gold_sql_sha256:
            raise SpiderDataError(
                "Smoke manifest gold SQL hash mismatch at dev:%d" % sample.index
            )
        selected.append((sample, example))
    return selected


def _run_directory(base: Path, backend_name: str, run_name: Optional[str]) -> Tuple[str, Path]:
    if run_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = "%s-%s-%s" % (timestamp, backend_name, uuid.uuid4().hex[:8])
    if not _SAFE_RUN_NAME.fullmatch(run_name):
        raise ValueError("run_name may contain only letters, digits, dot, underscore, and dash")
    run_dir = (base / run_name).resolve()
    base_resolved = base.resolve()
    try:
        run_dir.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError("run directory escapes the configured output root") from exc
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_name, run_dir


def run_smoke(
    config: AppConfig,
    backend_name: str,
    allow_model_download: bool = False,
    run_name: Optional[str] = None,
    invocation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    local_checkpoint = None
    if backend_name == "hf":
        validate_download_policy((config.model,), allow_model_download)
        prepared_model, local_checkpoint = prepare_model_config(config.model)
        config = replace(config, model=prepared_model)
    started_wall = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    dataset = SpiderDataset(config.spider)
    validation = dataset.validate()
    if not validation.ok:
        raise SpiderDataError(
            "Spider validation failed:\n- " + "\n- ".join(validation.errors)
        )
    selected = _selected_examples(dataset, config.smoke.samples)
    official_preflight: Optional[Dict[str, Any]] = None
    if config.official_evaluation.enabled:
        official_preflight = validate_official_environment(
            evaluator_root=config.official_evaluation.evaluator_root,
            database_root=config.official_evaluation.test_suite_database_root,
            tables_path=dataset.tables_path,
            expected_commit=config.official_evaluation.upstream_commit,
            nltk_data_dir=config.official_evaluation.nltk_data_dir,
            db_ids=[example.db_id for _, example in selected],
        )
        if not official_preflight["ok"]:
            raise OfficialEvaluationError(
                "Official evaluator preflight failed:\n- "
                + "\n- ".join(official_preflight["errors"])
            )
    run_id, run_dir = _run_directory(config.output.directory, backend_name, run_name)

    db_paths = {example.db_id: dataset.database_path(example.db_id) for _, example in selected}
    database_hashes_before = {
        db_id: _sha256_file(path) for db_id, path in sorted(db_paths.items())
    }
    mock_responses = {
        "%s:%d" % (example.split, example.index): example.gold_sql
        for _, example in selected
    }
    effective_config = _effective_config_payload(config)
    config_hash = _sha256_bytes(_json_bytes(effective_config))
    manifest: Dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id,
        "status": "initializing_backend",
        "started_at": started_wall.isoformat(),
        "invocation": dict(invocation or {"interface": "python_api"}),
        "backend_selector": backend_name,
        "backend": {
            "backend": backend_name,
            "model_id": config.model.model_id if backend_name == "hf" else None,
            "model_source": config.model.source if backend_name == "hf" else None,
            "checkpoint_identity": (
                config.model.checkpoint_identity if backend_name == "hf" else None
            ),
            "allow_model_download": allow_model_download,
        },
        "contract": {
            "single_turn": True,
            "generation_calls_per_example": 1,
            "execution_feedback": False,
            "self_correction": False,
            "few_shot": False,
            "database_values_in_prompt": False,
            "official_spider_metric": config.official_evaluation.enabled,
            "official_test_suite_accuracy": config.official_evaluation.enabled,
            "official_original_exact_set_match": config.official_evaluation.enabled,
            "prediction_sql_execution_time": {
                "recorded": True,
                "reward_design_in_scope": False,
                "primary_observation": "predicted_execution.query_elapsed_ns",
                "clock": "time.perf_counter_ns (monotonic)",
                "query_interval_start": "immediately_before_sqlite_connection_execute",
                "query_interval_end": "after_fetchmany_completion_or_query_error",
                "execution_order": ["prediction", "gold"],
                "cache_policy": "no_explicit_cache_reset; prediction_runs_before_gold",
                "summary_inclusion": (
                    "status=success and query_elapsed_ns is available"
                ),
                "official_evaluator_runtime_included": False,
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
        },
        "config_source_path": str(config.source_path),
        "source_config": config.raw,
        "source_config_sha256": _sha256_bytes(_json_bytes(config.raw)),
        "config": effective_config,
        "config_sha256": config_hash,
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "sqlite_version": sqlite3.sqlite_version,
        },
        "source": _source_snapshot(),
        "local_checkpoint": local_checkpoint,
        "official_evaluation": {
            "enabled": config.official_evaluation.enabled,
            "preflight": official_preflight,
        },
        "dataset": {
            "split": config.spider.split,
            "example_file": config.spider.examples_file,
            "tables_file": config.spider.tables_file,
            "example_file_sha256": _sha256_file(dataset.examples_path),
            "tables_file_sha256": _sha256_file(dataset.tables_path),
            "selected_database_sha256_before": database_hashes_before,
            "validation": validation.to_dict(),
        },
    }
    _atomic_json(run_dir / "run_manifest.json", manifest)

    records: List[Dict[str, Any]] = []
    records_path = run_dir / "records.jsonl"
    backend: Optional[GenerationBackend] = None
    try:
        backend = _create_backend(
            backend_name,
            config,
            mock_responses=mock_responses,
            allow_model_download=allow_model_download,
        )
        backend_metadata = backend.metadata()
    except Exception as exc:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        records_path.write_text("", encoding="utf-8")
        database_hashes_after = {
            db_id: _sha256_file(path) for db_id, path in sorted(db_paths.items())
        }
        databases_unchanged = database_hashes_before == database_hashes_after
        finished_wall = datetime.now(timezone.utc)
        failure = _failure_payload("backend_initialization", exc)
        summary = {
            "schema_version": 2,
            "run_id": run_id,
            "backend": backend_name,
            "pipeline_pass": False,
            "total_examples": len(selected),
            "processed_examples": 0,
            "selected_databases_unchanged": databases_unchanged,
            "failure": failure,
            "started_at": started_wall.isoformat(),
            "finished_at": finished_wall.isoformat(),
            "elapsed_seconds": time.monotonic() - started_monotonic,
        }
        _atomic_json(run_dir / "summary.json", summary)
        manifest.update(
            {
                "status": "failed",
                "finished_at": finished_wall.isoformat(),
                "elapsed_seconds": summary["elapsed_seconds"],
                "failure": failure,
                "artifacts": {
                    "records": "records.jsonl",
                    "summary": "summary.json",
                },
            }
        )
        manifest["dataset"]["selected_database_sha256_after"] = database_hashes_after
        manifest["dataset"]["selected_databases_unchanged"] = databases_unchanged
        _atomic_json(run_dir / "run_manifest.json", manifest)
        raise RuntimeError(
            "Backend initialization failed; diagnostics written to %s: %s"
            % (run_dir, exc)
        ) from exc

    manifest["status"] = "running"
    manifest["backend"] = backend_metadata
    _atomic_json(run_dir / "run_manifest.json", manifest)

    processing_error: Optional[Exception] = None
    processing_error_stage = "pipeline_execution"
    try:
        with records_path.open("w", encoding="utf-8") as records_handle:
            for sample, example in selected:
                example_id = "%s:%d" % (example.split, example.index)
                schema_text = serialize_schema(dataset.get_schema(example.db_id))
                messages = build_messages(example.question, schema_text)
                request = GenerationRequest(example_id=example_id, messages=tuple(messages))

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
                        error_message="SQL parsing skipped because generation failed",
                    )

                execution_args = {
                    "timeout_seconds": config.execution.timeout_seconds,
                    "max_sql_bytes": config.execution.max_sql_bytes,
                    "max_result_rows": config.execution.max_result_rows,
                    "max_result_bytes": config.execution.max_result_bytes,
                    "worker_memory_limit_bytes": (
                        config.execution.worker_memory_limit_bytes
                    ),
                }
                db_path = dataset.database_path(example.db_id)
                if parsed.status == "success" and parsed.sql is not None:
                    predicted_execution = execute_sql(db_path, parsed.sql, **execution_args)
                else:
                    predicted_execution = _not_run(
                        parsed.error_type or "sql_parse_error",
                        parsed.error_message or "Prediction was not executable",
                    )
                # Run the prediction first so its latency is not improved by a
                # gold-query execution warming SQLite or filesystem caches.
                gold_execution = execute_sql(db_path, example.gold_sql, **execution_args)

                order_sensitive = bool(example.parsed_sql.get("orderBy"))
                comparison = compare_results(
                    predicted_execution,
                    gold_execution,
                    order_sensitive=order_sensitive,
                )
                comparison["metric"] = "local_single_database_result_match"
                comparison["official_spider_metric"] = False
                record = {
                    "schema_version": 2,
                    "run_id": run_id,
                    "example_id": example_id,
                    "split": example.split,
                    "index": example.index,
                    "db_id": example.db_id,
                    "category": sample.category,
                    "question": example.question,
                    "gold_sql": example.gold_sql,
                    "schema_sha256": _sha256_text(schema_text),
                    "prompt_sha256": _sha256_bytes(_json_bytes(messages)),
                    "messages": messages,
                    "generation": generation.to_dict(),
                    "sql_parsing": parsed.to_dict(),
                    "predicted_execution": _execution_payload(predicted_execution),
                    "gold_execution": _execution_payload(gold_execution),
                    "comparison": comparison,
                }
                records.append(record)
                records_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                records_handle.write("\n")
                records_handle.flush()
    except Exception as exc:
        processing_error = exc
    try:
        backend.close()
    except Exception as exc:
        if processing_error is None:
            processing_error = exc

    official_batch: Optional[Dict[str, Any]] = None
    if processing_error is None:
        try:
            if config.official_evaluation.enabled:
                official_items = [
                    OfficialEvaluationItem(
                        example_id=record["example_id"],
                        db_id=record["db_id"],
                        gold_sql=record["gold_sql"],
                        predicted_sql=(
                            record["sql_parsing"]["sql"]
                            if record["sql_parsing"]["status"] == "success"
                            else None
                        ),
                    )
                    for record in records
                ]
                official_batch = evaluate_official(
                    official_items,
                    evaluator_root=config.official_evaluation.evaluator_root,
                    database_root=(
                        config.official_evaluation.test_suite_database_root
                    ),
                    tables_path=dataset.tables_path,
                    expected_commit=config.official_evaluation.upstream_commit,
                    nltk_data_dir=config.official_evaluation.nltk_data_dir,
                    timeout_seconds=config.official_evaluation.timeout_seconds,
                    max_sql_bytes=config.execution.max_sql_bytes,
                    worker_memory_limit_bytes=(
                        config.execution.worker_memory_limit_bytes
                    ),
                    plug_value=config.official_evaluation.plug_value,
                    keep_distinct=config.official_evaluation.keep_distinct,
                    preflight_report=official_preflight,
                )
                official_results = official_batch["results"]
                if len(official_results) != len(records):
                    raise OfficialEvaluationError(
                        "Official evaluator returned the wrong result count"
                    )
                for record, official_result in zip(records, official_results):
                    if official_result["example_id"] != record["example_id"]:
                        raise OfficialEvaluationError(
                            "Official evaluator changed example ordering"
                        )
                    record["official_evaluation"] = official_result
                manifest["official_evaluation"]["result"] = {
                    key: value
                    for key, value in official_batch.items()
                    if key != "results"
                }
            else:
                for record in records:
                    record["official_evaluation"] = {
                        "status": "disabled",
                        "exact_set_match": None,
                        "test_suite": None,
                    }
            _atomic_jsonl(records_path, records)
        except Exception as exc:
            processing_error = exc
            processing_error_stage = "official_evaluation"

    if processing_error is not None:
        database_hashes_after = {
            db_id: _sha256_file(path) for db_id, path in sorted(db_paths.items())
        }
        databases_unchanged = database_hashes_before == database_hashes_after
        finished_wall = datetime.now(timezone.utc)
        failure = _failure_payload(processing_error_stage, processing_error)
        summary = {
            "schema_version": 2,
            "run_id": run_id,
            "backend": backend_name,
            "pipeline_pass": False,
            "total_examples": len(selected),
            "processed_examples": len(records),
            "selected_databases_unchanged": databases_unchanged,
            "failure": failure,
            "started_at": started_wall.isoformat(),
            "finished_at": finished_wall.isoformat(),
            "elapsed_seconds": time.monotonic() - started_monotonic,
        }
        _atomic_json(run_dir / "summary.json", summary)
        manifest.update(
            {
                "status": "failed",
                "finished_at": finished_wall.isoformat(),
                "elapsed_seconds": summary["elapsed_seconds"],
                "failure": failure,
                "artifacts": {
                    "records": "records.jsonl",
                    "summary": "summary.json",
                },
            }
        )
        manifest["dataset"]["selected_database_sha256_after"] = database_hashes_after
        manifest["dataset"]["selected_databases_unchanged"] = databases_unchanged
        _atomic_json(run_dir / "run_manifest.json", manifest)
        raise RuntimeError(
            "Smoke pipeline failed; diagnostics written to %s: %s"
            % (run_dir, processing_error)
        ) from processing_error

    database_hashes_after = {
        db_id: _sha256_file(path) for db_id, path in sorted(db_paths.items())
    }
    databases_unchanged = database_hashes_before == database_hashes_after
    generation_statuses = Counter(record["generation"]["status"] for record in records)
    parsing_statuses = Counter(record["sql_parsing"]["status"] for record in records)
    prediction_statuses = Counter(
        record["predicted_execution"]["status"] for record in records
    )
    gold_statuses = Counter(record["gold_execution"]["status"] for record in records)
    local_result_matches = sum(
        bool(record["comparison"]["result_match"]) for record in records
    )
    total = len(selected)
    executable = prediction_statuses.get("success", 0)
    if config.official_evaluation.enabled:
        assert official_batch is not None
        exact_statuses = Counter(
            record["official_evaluation"]["exact_set_match"]["status"]
            for record in records
        )
        test_suite_statuses = Counter(
            record["official_evaluation"]["test_suite"]["status"]
            for record in records
        )
        observed_exact_set_matches = sum(
            bool(record["official_evaluation"]["exact_set_match"]["match"])
            for record in records
        )
        observed_test_suite_matches = sum(
            bool(record["official_evaluation"]["test_suite"]["match"])
            for record in records
        )
        exact_set_match_valid = bool(
            official_batch["metrics"]["exact_set_match"]["valid"]
        )
        test_suite_accuracy_valid = bool(
            official_batch["metrics"]["test_suite"]["valid"]
        )
        exact_set_match_classified_examples = official_batch["metrics"][
            "exact_set_match"
        ]["classified_examples"]
        test_suite_classified_examples = official_batch["metrics"][
            "test_suite"
        ]["classified_examples"]
        exact_set_matches: Optional[int] = (
            observed_exact_set_matches if exact_set_match_valid else None
        )
        test_suite_matches: Optional[int] = (
            observed_test_suite_matches if test_suite_accuracy_valid else None
        )
        official_evaluation_ok = (
            exact_set_match_valid and test_suite_accuracy_valid
        )
        official_contract_met = (
            official_evaluation_ok
            and official_batch["processed_examples"] == total
            and (
                backend_name != "mock"
                or (
                    exact_set_matches == total
                    and test_suite_matches == total
                )
            )
        )
    else:
        exact_statuses = Counter()
        test_suite_statuses = Counter()
        exact_set_matches = None
        test_suite_matches = None
        exact_set_match_valid = None
        test_suite_accuracy_valid = None
        exact_set_match_classified_examples = None
        test_suite_classified_examples = None
        official_evaluation_ok = None
        official_contract_met = True
    if backend_name == "mock":
        prediction_contract_met = (
            parsing_statuses.get("success", 0) == total
            and executable == total
            and local_result_matches == total
        )
    else:
        prediction_contract_met = executable >= config.smoke.minimum_executable
    pipeline_pass = (
        len(records) == total
        and generation_statuses.get("success", 0) == total
        and gold_statuses.get("success", 0) == total
        and prediction_contract_met
        and official_contract_met
        and databases_unchanged
    )
    finished_wall = datetime.now(timezone.utc)
    summary: Dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id,
        "backend": backend_name,
        "backend_selector": backend_name,
        "backend_implementation": backend_metadata.get("backend"),
        "pipeline_pass": pipeline_pass,
        "accuracy_measurement": backend_name != "mock",
        "official_spider_metric": config.official_evaluation.enabled,
        "official_evaluation_ok": official_evaluation_ok,
        "official_contract_met": official_contract_met,
        "exact_set_match_valid": exact_set_match_valid,
        "test_suite_accuracy_valid": test_suite_accuracy_valid,
        "exact_set_match_classified_examples": (
            exact_set_match_classified_examples
        ),
        "test_suite_classified_examples": test_suite_classified_examples,
        "total_examples": total,
        "processed_examples": len(records),
        "generation_statuses": dict(generation_statuses),
        "sql_parsing_statuses": dict(parsing_statuses),
        "prediction_execution_statuses": dict(prediction_statuses),
        "gold_execution_statuses": dict(gold_statuses),
        "minimum_executable_required": config.smoke.minimum_executable,
        "executable_predictions": executable,
        "prediction_contract_met": prediction_contract_met,
        "exact_set_match_statuses": dict(exact_statuses),
        "exact_set_matches": exact_set_matches,
        "exact_set_match_accuracy": (
            exact_set_matches / total
            if total and exact_set_matches is not None
            else None
        ),
        "test_suite_statuses": dict(test_suite_statuses),
        "test_suite_matches": test_suite_matches,
        "test_suite_accuracy": (
            test_suite_matches / total
            if total and test_suite_matches is not None
            else None
        ),
        "local_result_matches": local_result_matches,
        "local_result_match_rate": (
            local_result_matches / total if total else 0.0
        ),
        "local_result_match_policy": (
            "single original Spider DB; diagnostic only; not an official metric"
        ),
        "prediction_query_timing": _query_timing_summary(
            records, "predicted_execution"
        ),
        "gold_query_timing": _query_timing_summary(records, "gold_execution"),
        "prediction_vm_steps": _vm_step_summary(
            records, "predicted_execution"
        ),
        "gold_vm_steps": _vm_step_summary(records, "gold_execution"),
        "selected_databases_unchanged": databases_unchanged,
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "started_at": started_wall.isoformat(),
        "finished_at": finished_wall.isoformat(),
    }
    if backend_name == "mock":
        summary["notice"] = (
            "Gold-backed mock results validate the pipeline only and are not model accuracy."
        )
    _atomic_json(run_dir / "summary.json", summary)

    manifest["status"] = "completed" if pipeline_pass else "failed"
    manifest["finished_at"] = finished_wall.isoformat()
    manifest["elapsed_seconds"] = summary["elapsed_seconds"]
    manifest["backend"] = backend_metadata
    manifest["dataset"]["selected_database_sha256_after"] = database_hashes_after
    manifest["dataset"]["selected_databases_unchanged"] = databases_unchanged
    manifest["artifacts"] = {
        "records": "records.jsonl",
        "summary": "summary.json",
    }
    _atomic_json(run_dir / "run_manifest.json", manifest)
    return {"run_directory": str(run_dir), "summary": summary}
