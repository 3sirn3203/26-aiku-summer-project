from __future__ import annotations

import json
import os
import platform
import sqlite3
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from text2sql.config import AppConfig
from text2sql.single_turn.distributed import (
    DistributedGenerationError,
    generation_contract,
    launch_generation_attempt,
    load_generation_shards,
    validate_and_order_generation_records,
)
from text2sql.core.evaluation import compare_results
from text2sql.core.executor import execute_sql
from text2sql.core.official_eval import (
    OfficialEvaluationError,
    OfficialEvaluationItem,
    evaluate_official,
    validate_official_environment,
)
from text2sql.core.progress import ProgressReporter
from text2sql.core.model_source import (
    expected_model_identity,
    inspect_peft_adapter,
    prepare_model_config,
    validate_download_policy,
)
from text2sql.single_turn.smoke_runner import (
    _atomic_json,
    _atomic_jsonl,
    _effective_config_payload,
    _execution_payload,
    _failure_payload,
    _json_bytes,
    _not_run,
    _query_timing_summary,
    _vm_step_summary,
    _run_directory,
    _SAFE_RUN_NAME,
    _selected_examples,
    _sha256_bytes,
    _sha256_file,
    _source_snapshot,
)
from text2sql.core.spider import SpiderDataError, SpiderDataset


_INFRASTRUCTURE_OFFICIAL_STATUSES = {
    "evaluator_error",
    "evaluator_timeout",
    "gold_error",
}


def _read_evaluation_records(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    records: List[Dict[str, Any]] = []
    incomplete_final_line = False
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if line_number == len(lines):
                incomplete_final_line = True
                break
            raise RuntimeError(
                "invalid evaluation JSONL at %s:%d" % (path, line_number)
            ) from exc
        if not isinstance(record, dict):
            raise RuntimeError(
                "evaluation JSONL record is not an object at %s:%d"
                % (path, line_number)
            )
        records.append(record)
    if incomplete_final_line:
        # This is a derived, append-only artifact.  Repair only the incomplete
        # tail left by an interrupted write before appending resumed records.
        _atomic_jsonl(path, records)
    return records


def _full_contract(run_type: str, selection: str) -> Dict[str, Any]:
    return {
        "run_type": run_type,
        "selection": selection,
        "single_turn": True,
        "generation_calls_per_example": 1,
        "execution_feedback": False,
        "self_correction": False,
        "few_shot": False,
        "database_values_in_prompt": False,
        "distributed_inference": {
            "strategy": "one_independent_model_process_per_physical_gpu",
            "model_parallelism": False,
            "ddp": False,
            "sharding": "stable_round_robin_by_original_example_index",
            "startup": "workers_load_models_one_at_a_time_until_ready",
            "worker_responsibility": "prompt_generation_and_sql_parsing_only",
            "parent_imports_model_libraries": False,
        },
        "execution_and_evaluation": {
            "owner": "single_parent_cpu_process_after_all_generation_workers_exit",
            "parallel_sql_execution": False,
            "execution_order_per_example": ["prediction", "gold"],
            "official_evaluation_after_local_execution": True,
        },
        "prediction_sql_execution_time": {
            "recorded": True,
            "reward_design_in_scope": False,
            "primary_observation": "predicted_execution.query_elapsed_ns",
            "clock": "time.perf_counter_ns (monotonic)",
            "query_interval_start": "immediately_before_sqlite_connection_execute",
            "query_interval_end": "after_fetchmany_completion_or_query_error",
            "execution_order": ["prediction", "gold"],
            "cache_policy": "no_explicit_cache_reset; prediction_runs_before_gold",
            "summary_inclusion": "status=success and query_elapsed_ns is available",
            "generation_worker_runtime_included": False,
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
        },
    }


def _write_failure(
    *,
    run_dir: Path,
    manifest: Dict[str, Any],
    stage: str,
    exc: Exception,
    started_wall: datetime,
    started_monotonic: float,
    processed_examples: int,
    total_examples: int,
) -> None:
    finished = datetime.now(timezone.utc)
    failure = _failure_payload(stage, exc)
    summary = {
        "schema_version": 4,
        "run_id": manifest["run_id"],
        "run_type": manifest["run_type"],
        "pipeline_pass": False,
        "total_examples": total_examples,
        "processed_examples": processed_examples,
        "failure": failure,
        "started_at": started_wall.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "resume_supported": True,
    }
    _atomic_json(run_dir / "summary.json", summary)
    manifest.update(
        {
            "status": "failed",
            "failure": failure,
            "finished_at": finished.isoformat(),
            "elapsed_seconds": summary["elapsed_seconds"],
            "artifacts": {
                "generation_records": "generation_records.jsonl",
                "records": "records.jsonl",
                "summary": "summary.json",
                "shards": "shards/",
            },
        }
    )
    _atomic_json(run_dir / "run_manifest.json", manifest)


def _validate_resume_manifest(
    manifest: Mapping[str, Any],
    *,
    backend_name: str,
    run_type: str,
    selection: str,
    config_sha256: str,
    source_config_sha256: str,
    source_tree_sha256: str,
    generation_contract_sha256: str,
    official_preflight_sha256: Optional[str],
    split: str,
    total_examples: int,
) -> None:
    expected = {
        "run_type": run_type,
        "backend_selector": backend_name,
        "config_sha256": config_sha256,
        "source_config_sha256": source_config_sha256,
    }
    mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
    dataset = manifest.get("dataset", {})
    if (
        dataset.get("split") != split
        or dataset.get("selection") != selection
        or dataset.get("total_examples") != total_examples
    ):
        mismatched.append("dataset")
    source = manifest.get("source", {})
    if source.get("python_tree_sha256") != source_tree_sha256:
        mismatched.append("source.python_tree_sha256")
    generation = manifest.get("generation", {})
    if generation.get("contract_sha256") != generation_contract_sha256:
        mismatched.append("generation.contract_sha256")
    official = manifest.get("official_evaluation", {})
    if official.get("preflight_sha256") != official_preflight_sha256:
        mismatched.append("official_evaluation.preflight_sha256")
    if mismatched:
        raise RuntimeError(
            "refusing to resume because the run contract changed: %s"
            % ", ".join(mismatched)
        )


def _validate_existing_evaluation_records(
    records: Sequence[Mapping[str, Any]],
    generation_records: Sequence[Mapping[str, Any]],
) -> None:
    if len(records) > len(generation_records):
        raise RuntimeError("records.jsonl contains more rows than generation output")
    seen = set()
    for position, record in enumerate(records):
        expected = generation_records[position]
        example_id = record.get("example_id")
        if example_id in seen:
            raise RuntimeError("records.jsonl contains duplicate example IDs")
        seen.add(example_id)
        if example_id != expected.get("example_id"):
            raise RuntimeError(
                "records.jsonl is not a prefix in original Spider order"
            )
        for key in ("prompt_sha256", "sql_parsing"):
            if record.get(key) != expected.get(key):
                raise RuntimeError(
                    "records.jsonl does not match generation output for %s"
                    % example_id
                )


def _official_metric_summary(
    records: Sequence[Mapping[str, Any]], metric: str
) -> Dict[str, Any]:
    results = [record["official_evaluation"][metric] for record in records]
    statuses = Counter(result["status"] for result in results)
    infrastructure_failures = sum(
        status in _INFRASTRUCTURE_OFFICIAL_STATUSES
        for status in (result["status"] for result in results)
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
            if len(records) and matches is not None
            else None
        ),
    }


def _worker_backend_metadata(run_dir: Path) -> List[Dict[str, Any]]:
    statuses = []
    for path in sorted((run_dir / "shards").glob("attempt-*/*.status.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("status") in {
            "completed",
            "failed",
        }:
            statuses.append(payload)
    return statuses


def run_full_evaluation(
    config: AppConfig,
    *,
    backend_name: str,
    gpu_ids: Sequence[int],
    allow_model_download: bool = False,
    run_name: Optional[str] = None,
    resume_run: Optional[str] = None,
    selection: str = "all",
    progress: Optional[ProgressReporter] = None,
    invocation: Optional[Mapping[str, Any]] = None,
    adapter_dir: Optional[Path] = None,
    workflow_contract: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate all or fixed-smoke examples with data-parallel inference."""

    if not gpu_ids:
        raise ValueError("distributed evaluation requires at least one GPU worker")
    if run_name is not None and resume_run is not None:
        raise ValueError("run_name and resume_run are mutually exclusive")
    if selection not in {"all", "smoke"}:
        raise ValueError("selection must be 'all' or 'smoke'")
    local_checkpoint = None
    adapter_inspection = None
    if backend_name == "hf":
        validate_download_policy((config.model,), allow_model_download)
        prepared_model, local_checkpoint = prepare_model_config(config.model)
        config = replace(config, model=prepared_model)
    elif backend_name == "peft":
        if config.model.source != "hub":
            raise ValueError("PEFT evaluation requires model.source=hub")
        if adapter_dir is None:
            raise ValueError("PEFT evaluation requires --adapter-dir")
        validate_download_policy((config.model,), allow_model_download)
        adapter_inspection = inspect_peft_adapter(
            adapter_dir,
            expected_base_model_id=config.model.model_id,
            expected_base_model_revision=config.model.revision,
        )
        if not adapter_inspection.ok or adapter_inspection.identity is None:
            raise ValueError(
                "invalid PEFT adapter %s: %s"
                % (adapter_inspection.path, "; ".join(adapter_inspection.errors))
            )
    elif backend_name == "two_turn":
        if config.model.source != "hub":
            raise ValueError("two-turn evaluation requires model.source=hub")
        if workflow_contract is None:
            raise ValueError("two-turn evaluation requires a workflow contract")
        validate_download_policy((config.model,), allow_model_download)
        if adapter_dir is not None:
            adapter_inspection = inspect_peft_adapter(
                adapter_dir,
                expected_base_model_id=config.model.model_id,
                expected_base_model_revision=config.model.revision,
            )
            if not adapter_inspection.ok or adapter_inspection.identity is None:
                raise ValueError(
                    "invalid PEFT adapter %s: %s"
                    % (
                        adapter_inspection.path,
                        "; ".join(adapter_inspection.errors),
                    )
                )
    elif adapter_dir is not None:
        raise ValueError("adapter_dir is valid only with backend_name='peft'")
    elif workflow_contract is not None:
        raise ValueError("workflow_contract is valid only for two-turn evaluation")
    started_wall = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    reporter = progress if progress is not None else ProgressReporter(enabled=False)
    dataset = SpiderDataset(config.spider)
    validation = dataset.validate()
    if not validation.ok:
        raise SpiderDataError(
            "Spider validation failed:\n- " + "\n- ".join(validation.errors)
        )
    if selection == "all":
        selected_examples = list(dataset.examples)
        run_type = "full_evaluation"
    else:
        selected_examples = [
            example
            for _, example in _selected_examples(dataset, config.smoke.samples)
        ]
        run_type = "distributed_smoke"
    expected_indices = [example.index for example in selected_examples]
    total = len(expected_indices)
    if total == 0:
        raise SpiderDataError("configured Spider split is empty")

    official_preflight: Optional[Dict[str, Any]] = None
    if config.official_evaluation.enabled:
        official_preflight = validate_official_environment(
            evaluator_root=config.official_evaluation.evaluator_root,
            database_root=config.official_evaluation.test_suite_database_root,
            tables_path=dataset.tables_path,
            expected_commit=config.official_evaluation.upstream_commit,
            nltk_data_dir=config.official_evaluation.nltk_data_dir,
            db_ids=[example.db_id for example in selected_examples],
        )
        if not official_preflight["ok"]:
            raise OfficialEvaluationError(
                "Official evaluator preflight failed:\n- "
                + "\n- ".join(official_preflight["errors"])
            )

    effective_config = _effective_config_payload(config)
    config_sha256 = _sha256_bytes(_json_bytes(effective_config))
    source_config_sha256 = _sha256_bytes(_json_bytes(config.raw))
    official_preflight_sha256 = (
        _sha256_bytes(_json_bytes(official_preflight))
        if official_preflight is not None
        else None
    )
    source = _source_snapshot()
    adapter_contract = (
        adapter_inspection.to_dict() if adapter_inspection is not None else None
    )
    generation = generation_contract(
        config,
        dataset,
        adapter_contract=adapter_contract,
        workflow_contract=workflow_contract,
    )
    db_paths = {
        db_id: dataset.database_path(db_id)
        for db_id in sorted({example.db_id for example in selected_examples})
    }
    worker_context = (
        {
            "database_paths": {
                "%s:%d" % (example.split, example.index): str(
                    dataset.database_path(example.db_id)
                )
                for example in selected_examples
            }
        }
        if backend_name == "two_turn"
        else None
    )

    if resume_run is not None:
        if _SAFE_RUN_NAME.fullmatch(resume_run) is None:
            raise ValueError(
                "resume_run may contain only letters, digits, dot, underscore, and dash"
            )
        run_dir = (config.output.directory / resume_run).resolve()
        try:
            run_dir.relative_to(config.output.directory.resolve())
        except ValueError as exc:
            raise ValueError("resume run directory escapes the output root") from exc
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("resume manifest does not exist: %s" % manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = str(manifest.get("run_id"))
        _validate_resume_manifest(
            manifest,
            backend_name=backend_name,
            run_type=run_type,
            selection=selection,
            config_sha256=config_sha256,
            source_config_sha256=source_config_sha256,
            source_tree_sha256=source["python_tree_sha256"],
            generation_contract_sha256=generation["sha256"],
            official_preflight_sha256=official_preflight_sha256,
            split=config.spider.split,
            total_examples=total,
        )
        manifest.setdefault("resume_invocations", []).append(dict(invocation or {}))
        database_hashes_before = manifest["dataset"][
            "selected_database_sha256_before"
        ]
    else:
        run_id, run_dir = _run_directory(
            config.output.directory, backend_name, run_name
        )
        database_hashes_before = {
            db_id: _sha256_file(path) for db_id, path in db_paths.items()
        }
        manifest = {
            "schema_version": 4,
            "run_id": run_id,
            "run_type": run_type,
            "status": "initializing",
            "started_at": started_wall.isoformat(),
            "invocation": dict(invocation or {"interface": "python_api"}),
            "backend_selector": backend_name,
            "gpu_ids": list(gpu_ids),
            "contract": (
                {
                    **_full_contract(run_type, selection),
                    "single_turn": False,
                    "generation_calls_per_example": 2,
                    "execution_feedback": True,
                    "self_correction": True,
                    "workflow": dict(workflow_contract or {}),
                }
                if backend_name == "two_turn"
                else _full_contract(run_type, selection)
            ),
            "config_source_path": str(config.source_path),
            "source_config": config.raw,
            "source_config_sha256": source_config_sha256,
            "config": effective_config,
            "config_sha256": config_sha256,
            "runtime": {
                "python_version": os.sys.version.split()[0],
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "sqlite_version": sqlite3.sqlite_version,
            },
            "source": source,
            "local_checkpoint": local_checkpoint,
            "adapter": adapter_contract,
            "generation": {
                "contract": generation["payload"],
                "contract_sha256": generation["sha256"],
                "attempts": [],
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
                "example_file": config.spider.examples_file,
                "tables_file": config.spider.tables_file,
                "example_file_sha256": _sha256_file(dataset.examples_path),
                "tables_file_sha256": _sha256_file(dataset.tables_path),
                "selected_database_sha256_before": database_hashes_before,
                "validation": validation.to_dict(),
            },
        }
    manifest["status"] = "generating"
    manifest["active_gpu_ids"] = list(gpu_ids)
    _atomic_json(run_dir / "run_manifest.json", manifest)

    shards_root = run_dir / "shards"
    try:
        generated_by_id, shard_warnings = load_generation_shards(shards_root)
        expected_ids = [
            "%s:%d" % (dataset.get_example(index).split, index)
            for index in expected_indices
        ]
        unexpected = sorted(set(generated_by_id) - set(expected_ids))
        if unexpected:
            raise DistributedGenerationError(
                "generation shards contain unexpected IDs: %s"
                % ", ".join(unexpected[:10])
            )
        if resume_run is not None:
            retryable_ids = [
                example_id
                for example_id, record in generated_by_id.items()
                if record.get("generation", {}).get("status") != "success"
                and record.get("generation", {}).get("error_type")
                != "input_too_long"
            ]
            for example_id in retryable_ids:
                generated_by_id.pop(example_id)
            if retryable_ids:
                shard_warnings.append(
                    "retrying %d generation infrastructure error(s)"
                    % len(retryable_ids)
                )
        remaining_indices = [
            index
            for index, example_id in zip(expected_indices, expected_ids)
            if example_id not in generated_by_id
        ]
        generation_initial = len(generated_by_id)
        reporter.start(
            "generation",
            "LLM generation",
            total,
            initial=generation_initial,
        )
        if remaining_indices:
            existing_attempts = list(shards_root.glob("attempt-*")) if shards_root.exists() else []
            attempt_number = len(existing_attempts)
            attempt_result = launch_generation_attempt(
                config_path=config.source_path,
                run_dir=run_dir,
                run_id=run_id,
                attempt=attempt_number,
                split=config.spider.split,
                remaining_indices=remaining_indices,
                gpu_ids=gpu_ids,
                backend_name=backend_name,
                allow_model_download=allow_model_download,
                generation_contract_sha256=generation["sha256"],
                source_tree_sha256=source["python_tree_sha256"],
                model_checkpoint_identity=config.model.checkpoint_identity,
                adapter_contract=adapter_contract,
                workflow_contract=workflow_contract,
                worker_context=worker_context,
                startup_delay_seconds=(
                    1.0
                    if backend_name in {"hf", "peft", "two_turn"}
                    else 0.0
                ),
                progress_callback=(
                    lambda completed, _remaining_total: reporter.update(
                        generation_initial + completed
                    )
                ),
                message_callback=reporter.message,
            )
            manifest["generation"].setdefault("attempts", []).append(attempt_result)
            _atomic_json(run_dir / "run_manifest.json", manifest)
        generated_by_id, final_shard_warnings = load_generation_shards(shards_root)
        shard_warnings.extend(final_shard_warnings)
        generation_records = validate_and_order_generation_records(
            generated_by_id,
            dataset,
            expected_indices,
            run_id,
            generation["sha256"],
        )
        _atomic_jsonl(run_dir / "generation_records.jsonl", generation_records)
        if backend_name == "two_turn":
            _atomic_jsonl(
                run_dir / "trajectories.jsonl",
                [
                    {
                        "schema_version": record["schema_version"],
                        "run_id": record["run_id"],
                        "example_id": record["example_id"],
                        "split": record["split"],
                        "index": record["index"],
                        "db_id": record["db_id"],
                        "question": record["question"],
                        "prompt_sha256": record["prompt_sha256"],
                        "workflow": record["workflow"],
                        "generation": record["generation"],
                        "sql_parsing": record["sql_parsing"],
                    }
                    for record in generation_records
                ],
            )
        manifest["generation"]["completed_examples"] = len(generation_records)
        manifest["generation"]["shard_warnings"] = sorted(set(shard_warnings))
        worker_statuses = _worker_backend_metadata(run_dir)
        manifest["generation"]["worker_statuses"] = worker_statuses
        if backend_name in {"hf", "peft", "two_turn"}:
            revisions = {
                status.get("backend", {}).get("resolved_revision")
                for status in worker_statuses
                if isinstance(status.get("backend"), Mapping)
            }
            revisions.discard(None)
            if len(revisions) != 1:
                raise DistributedGenerationError(
                    "generation worker statuses do not share one resolved model revision"
                )
            resolved_revision = next(iter(revisions))
            expected_identity = expected_model_identity(config.model)
            if resolved_revision != expected_identity:
                raise DistributedGenerationError(
                    "resolved model identity does not match the configured checkpoint"
                )
            manifest["generation"]["resolved_revision"] = resolved_revision
            if adapter_inspection is not None:
                adapter_identities = {
                    status.get("backend", {}).get("adapter", {}).get("identity")
                    for status in worker_statuses
                    if isinstance(status.get("backend"), Mapping)
                }
                if adapter_identities != {adapter_inspection.identity}:
                    raise DistributedGenerationError(
                        "generation workers did not load the contracted adapter"
                    )
        reporter.update(total)
        reporter.finish()
    except Exception as exc:
        reporter.message("distributed generation failed: %s" % exc)
        reporter.finish()
        _write_failure(
            run_dir=run_dir,
            manifest=manifest,
            stage="distributed_generation",
            exc=exc,
            started_wall=started_wall,
            started_monotonic=started_monotonic,
            processed_examples=len(locals().get("generated_by_id", {})),
            total_examples=total,
        )
        raise RuntimeError(
            "Distributed generation failed; resume artifacts are in %s: %s"
            % (run_dir, exc)
        ) from exc

    manifest["status"] = "executing_sql"
    _atomic_json(run_dir / "run_manifest.json", manifest)
    records_path = run_dir / "records.jsonl"
    try:
        records = _read_evaluation_records(records_path)
        _validate_existing_evaluation_records(records, generation_records)
        reporter.start(
            "sql_execution",
            "Sequential SQL execution",
            total,
            initial=len(records),
        )
        execution_args = {
            "timeout_seconds": config.execution.timeout_seconds,
            "max_sql_bytes": config.execution.max_sql_bytes,
            "max_result_rows": config.execution.max_result_rows,
            "max_result_bytes": config.execution.max_result_bytes,
            "worker_memory_limit_bytes": config.execution.worker_memory_limit_bytes,
        }
        with records_path.open("a", encoding="utf-8") as handle:
            for generation_record in generation_records[len(records) :]:
                example = dataset.get_example(generation_record["index"])
                if (
                    generation_record["example_id"]
                    != "%s:%d" % (example.split, example.index)
                ):
                    raise RuntimeError("generation record changed example identity")
                parsed = generation_record["sql_parsing"]
                db_path = dataset.database_path(example.db_id)
                if parsed["status"] == "success" and parsed.get("sql") is not None:
                    predicted_execution = execute_sql(
                        db_path, parsed["sql"], **execution_args
                    )
                else:
                    predicted_execution = _not_run(
                        parsed.get("error_type") or "sql_parse_error",
                        parsed.get("error_message") or "Prediction was not executable",
                    )
                gold_execution = execute_sql(
                    db_path, example.gold_sql, **execution_args
                )
                comparison = compare_results(
                    predicted_execution,
                    gold_execution,
                    order_sensitive=bool(example.parsed_sql.get("orderBy")),
                )
                comparison["metric"] = "local_single_database_result_match"
                comparison["official_spider_metric"] = False
                record = dict(generation_record)
                record.update(
                    {
                        "category": "full_split",
                        "gold_sql": example.gold_sql,
                        "predicted_execution": _execution_payload(
                            predicted_execution
                        ),
                        "gold_execution": _execution_payload(gold_execution),
                        "comparison": comparison,
                    }
                )
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                handle.flush()
                records.append(record)
                reporter.update(len(records))
        if len(records) != total:
            raise RuntimeError("SQL execution did not produce every evaluation record")
        reporter.update(total)
        reporter.finish()
    except Exception as exc:
        reporter.message("sequential SQL execution failed: %s" % exc)
        reporter.finish()
        _write_failure(
            run_dir=run_dir,
            manifest=manifest,
            stage="sequential_sql_execution",
            exc=exc,
            started_wall=started_wall,
            started_monotonic=started_monotonic,
            processed_examples=len(locals().get("records", [])),
            total_examples=total,
        )
        raise RuntimeError(
            "Sequential SQL execution failed; resume artifacts are in %s: %s"
            % (run_dir, exc)
        ) from exc

    manifest["status"] = "official_evaluation"
    _atomic_json(run_dir / "run_manifest.json", manifest)
    try:
        previous_official_result = manifest.get("official_evaluation", {}).get(
            "result", {}
        )
        official_policy: Optional[Mapping[str, Any]] = previous_official_result.get(
            "policy"
        )
        official_evaluator: Optional[Mapping[str, Any]] = previous_official_result.get(
            "evaluator"
        )
        if config.official_evaluation.enabled:
            if resume_run is not None:
                for record in records:
                    current = record.get("official_evaluation")
                    if not isinstance(current, dict):
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
            official_completed = total - len(pending_positions)
            reporter.start(
                "official_evaluation",
                "Official Spider evaluation",
                total,
                initial=official_completed,
            )
            official_batch_size = 8
            completed_batches = 0
            for start in range(0, len(pending_positions), official_batch_size):
                positions = pending_positions[start : start + official_batch_size]
                items = [
                    OfficialEvaluationItem(
                        example_id=records[position]["example_id"],
                        db_id=records[position]["db_id"],
                        gold_sql=records[position]["gold_sql"],
                        predicted_sql=(
                            records[position]["sql_parsing"].get("sql")
                            if records[position]["sql_parsing"]["status"] == "success"
                            else None
                        ),
                    )
                    for position in positions
                ]
                batch = evaluate_official(
                    items,
                    evaluator_root=config.official_evaluation.evaluator_root,
                    database_root=config.official_evaluation.test_suite_database_root,
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
                results = batch["results"]
                if len(results) != len(positions):
                    raise OfficialEvaluationError(
                        "official evaluator returned the wrong batch size"
                    )
                for position, official_result in zip(positions, results):
                    if official_result["example_id"] != records[position]["example_id"]:
                        raise OfficialEvaluationError(
                            "official evaluator changed example ordering"
                        )
                    records[position]["official_evaluation"] = official_result
                official_policy = batch["policy"]
                official_evaluator = batch["evaluator"]
                completed_batches += 1
                official_completed += len(positions)
                reporter.update(official_completed)
                _atomic_jsonl(records_path, records)
            manifest["official_evaluation"]["result"] = {
                "processed_examples": len(records),
                "batch_size": official_batch_size,
                "completed_batches_this_invocation": completed_batches,
                "policy": official_policy,
                "evaluator": official_evaluator,
            }
            reporter.update(total)
            reporter.finish()
        else:
            for record in records:
                record["official_evaluation"] = {
                    "status": "disabled",
                    "exact_set_match": None,
                    "test_suite": None,
                }
            _atomic_jsonl(records_path, records)
    except Exception as exc:
        reporter.message("official evaluation failed: %s" % exc)
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
        db_id: _sha256_file(path) for db_id, path in db_paths.items()
    }
    databases_unchanged = database_hashes_before == database_hashes_after
    generation_statuses = Counter(record["generation"]["status"] for record in records)
    parsing_statuses = Counter(record["sql_parsing"]["status"] for record in records)
    prediction_statuses = Counter(
        record["predicted_execution"]["status"] for record in records
    )
    gold_statuses = Counter(record["gold_execution"]["status"] for record in records)
    local_result_matches = sum(
        bool(record["comparison"].get("result_match")) for record in records
    )
    generation_infrastructure_errors = sum(
        record["generation"].get("status") != "success"
        and record["generation"].get("error_type") != "input_too_long"
        for record in records
    )
    if config.official_evaluation.enabled:
        exact = _official_metric_summary(records, "exact_set_match")
        test_suite = _official_metric_summary(records, "test_suite")
        official_evaluation_ok: Optional[bool] = exact["valid"] and test_suite["valid"]
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
        official_evaluation_ok = None
    base_contract_met = (
        len(records) == total
        and gold_statuses.get("success", 0) == total
        and generation_infrastructure_errors == 0
        and databases_unchanged
        and (official_evaluation_ok is not False)
    )
    if backend_name == "mock":
        mock_contract_met = (
            generation_statuses.get("success", 0) == total
            and parsing_statuses.get("success", 0) == total
            and prediction_statuses.get("success", 0) == total
            and local_result_matches == total
            and (
                not config.official_evaluation.enabled
                or (exact["matches"] == total and test_suite["matches"] == total)
            )
        )
    else:
        mock_contract_met = True
    pipeline_pass = base_contract_met and mock_contract_met
    finished_wall = datetime.now(timezone.utc)
    summary = {
        "schema_version": 4,
        "run_id": run_id,
        "run_type": run_type,
        "split": config.spider.split,
        "selection": selection,
        "backend": backend_name,
        "pipeline_pass": pipeline_pass,
        "accuracy_measurement": backend_name != "mock",
        "total_examples": total,
        "processed_examples": len(records),
        "generation_worker_count": len(worker_statuses),
        "generation_attempt_count": len(
            {status.get("attempt") for status in worker_statuses}
        ),
        "requested_gpu_count_this_invocation": len(gpu_ids),
        "generation_statuses": dict(generation_statuses),
        "generation_infrastructure_errors": generation_infrastructure_errors,
        "sql_parsing_statuses": dict(parsing_statuses),
        "prediction_execution_statuses": dict(prediction_statuses),
        "gold_execution_statuses": dict(gold_statuses),
        "executable_predictions": prediction_statuses.get("success", 0),
        "official_spider_metric": config.official_evaluation.enabled,
        "official_evaluation_ok": official_evaluation_ok,
        "exact_set_match_valid": exact["valid"],
        "exact_set_match_classified_examples": exact["classified_examples"],
        "exact_set_match_infrastructure_failures": exact["infrastructure_failures"],
        "exact_set_match_statuses": exact["statuses"],
        "exact_set_matches": exact["matches"],
        "exact_set_match_accuracy": exact["accuracy"],
        "test_suite_accuracy_valid": test_suite["valid"],
        "test_suite_classified_examples": test_suite["classified_examples"],
        "test_suite_infrastructure_failures": test_suite[
            "infrastructure_failures"
        ],
        "test_suite_statuses": test_suite["statuses"],
        "test_suite_matches": test_suite["matches"],
        "test_suite_accuracy": test_suite["accuracy"],
        "local_result_matches": local_result_matches,
        "local_result_match_rate": local_result_matches / total,
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
        "started_at": started_wall.isoformat(),
        "finished_at": finished_wall.isoformat(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
    }
    if backend_name == "mock":
        summary["notice"] = (
            "Gold-backed mock results validate orchestration only and are not model accuracy."
        )
    _atomic_json(run_dir / "summary.json", summary)
    manifest["status"] = "completed" if pipeline_pass else "failed"
    manifest["finished_at"] = finished_wall.isoformat()
    manifest["elapsed_seconds"] = summary["elapsed_seconds"]
    manifest["dataset"]["selected_database_sha256_after"] = database_hashes_after
    manifest["dataset"]["selected_databases_unchanged"] = databases_unchanged
    manifest["artifacts"] = {
        "generation_records": "generation_records.jsonl",
        "records": "records.jsonl",
        "summary": "summary.json",
        "shards": "shards/",
    }
    if backend_name == "two_turn":
        manifest["artifacts"]["trajectories"] = "trajectories.jsonl"
    _atomic_json(run_dir / "run_manifest.json", manifest)
    return {"run_directory": str(run_dir), "summary": summary}
