from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from text2sql import __version__
from text2sql.core.evaluation import result_hash
from text2sql.core.models import ExecutionResult
from text2sql.multi_turn_agent.config import AgentAppConfig


SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def atomic_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(str(temporary), str(path))


def run_directory(
    output_root: Path,
    run_name: Optional[str],
) -> Tuple[str, Path]:
    run_id = run_name or ("agent-" + uuid.uuid4().hex[:12])
    if SAFE_RUN_NAME.fullmatch(run_id) is None:
        raise ValueError(
            "run name may contain only letters, digits, dot, underscore, and dash"
        )
    root = output_root.resolve()
    path = (root / run_id).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("run directory escapes configured output root") from exc
    path.mkdir(parents=True, exist_ok=False)
    return run_id, path


def source_snapshot() -> Dict[str, Any]:
    package_root = Path(__file__).resolve().parents[1]
    source_paths = [package_root / "config.py"]
    source_paths.extend((package_root / "core").rglob("*.py"))
    source_paths.extend((package_root / "multi_turn_agent").rglob("*.py"))
    hashes = {
        str(path.relative_to(package_root)): sha256_file(path)
        for path in sorted(source_paths)
    }
    return {
        "package_version": __version__,
        "package_root": str(package_root),
        "scope": ["config.py", "core/**/*.py", "multi_turn_agent/**/*.py"],
        "python_files_sha256": hashes,
        "python_tree_sha256": sha256_bytes(json_bytes(hashes)),
    }


def effective_config_payload(config: AgentAppConfig) -> Dict[str, Any]:
    def model_payload(role: str) -> Dict[str, Any]:
        role_config = config.roles[role]
        model = role_config.model
        generation = role_config.generation
        return {
            "model": {
                "source": model.source,
                "id": model.model_id,
                "revision": model.revision,
                "checkpoint_identity": model.checkpoint_identity,
                "dtype": model.dtype,
                "device": model.device,
                "attention_implementation": model.attention_implementation,
                "trust_remote_code": model.trust_remote_code,
                "cache_dir": str(model.cache_dir) if model.cache_dir else None,
            },
            "generation": {
                "do_sample": generation.do_sample,
                "num_beams": generation.num_beams,
                "repetition_penalty": generation.repetition_penalty,
                "max_time_seconds": generation.max_time_seconds,
                "max_input_tokens": generation.max_input_tokens,
                "max_new_tokens": generation.max_new_tokens,
                "batch_size": generation.batch_size,
            },
            "trainable": role_config.trainable,
            "minimum_free_vram_bytes": role_config.minimum_free_vram_bytes,
        }

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
        "roles": {
            role: model_payload(role)
            for role in ("planner", "coder", "verifier")
        },
        "workflow": {
            "max_iterations": config.workflow.max_iterations,
            "observation_max_rows": config.workflow.observation_max_rows,
            "observation_max_bytes": config.workflow.observation_max_bytes,
            "infrastructure_retry_limit": config.workflow.infrastructure_retry_limit,
            "execution_concurrency": config.workflow.execution_concurrency,
        },
        "gpu_pools": {
            "planner": list(config.gpu_pools.planner),
            "coder": list(config.gpu_pools.coder),
            "verifier": list(config.gpu_pools.verifier),
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
                if config.official_evaluation.nltk_data_dir
                else None
            ),
        },
        "output": {"directory": str(config.output.directory)},
    }


def execution_payload(result: ExecutionResult) -> Dict[str, Any]:
    # Final records intentionally omit result rows.  The agent trajectory already
    # contains a separately bounded five-row observation, while the stable hash
    # and row count below are sufficient to audit local result comparison without
    # turning a full-dev artifact into a second copy of every query result.
    payload = {
        "status": result.status,
        "columns": list(result.columns),
        "row_fingerprint_version": result.row_fingerprint_version,
        "ordered_rows_fingerprint": result.ordered_rows_fingerprint,
        "unordered_rows_fingerprint": result.unordered_rows_fingerprint,
        "elapsed_seconds": result.elapsed_seconds,
        "query_elapsed_ns": result.query_elapsed_ns,
        "worker_elapsed_ns": result.worker_elapsed_ns,
        "parent_elapsed_ns": result.parent_elapsed_ns,
        "vm_steps_lower_bound": result.vm_steps_lower_bound,
        "vm_steps_upper_bound_exclusive": result.vm_steps_upper_bound_exclusive,
        "vm_step_progress_interval": result.vm_step_progress_interval,
        "vm_step_measurement_complete": result.vm_step_measurement_complete,
        "error_type": result.error_type,
        "error_message": result.error_message,
        "truncated": result.truncated,
        "denied_action": result.denied_action,
    }
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
    payload["row_count"] = (
        result.row_count
        if result.succeeded and result.row_count is not None
        else (len(result.rows) if result.succeeded else None)
    )
    payload["result_hash"] = result_hash(result)
    return payload


def query_timing_summary(
    records: Sequence[Mapping[str, Any]], execution_key: str
) -> Dict[str, Any]:
    executions = [record[execution_key] for record in records]
    successful = [item for item in executions if item["status"] == "success"]
    values = [
        item["query_elapsed_ns"]
        for item in successful
        if isinstance(item.get("query_elapsed_ns"), int)
        and not isinstance(item["query_elapsed_ns"], bool)
        and item["query_elapsed_ns"] >= 0
    ]

    def aggregate(items: Sequence[float]) -> Dict[str, Optional[float]]:
        if not items:
            return {"min": None, "max": None, "mean": None, "median": None}
        return {
            "min": min(items),
            "max": max(items),
            "mean": statistics.fmean(items),
            "median": statistics.median(items),
        }

    return {
        "total_executions": len(executions),
        "successful_executions": len(successful),
        "count": len(values),
        "excluded_non_success": len(executions) - len(successful),
        "excluded_success_without_query_timing": len(successful) - len(values),
        "inclusion": "status=success and query_elapsed_ns is available",
        "query_elapsed_ns": aggregate(values),
        "query_elapsed_ms": aggregate([value / 1_000_000 for value in values]),
    }


def vm_step_summary(
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

    return {
        "measurement": "progress_handler_bounds_not_exact_stmt_status",
        "total_executions": len(executions),
        "measured_count": len(measured),
        "complete_count": len(complete),
        "censored_count": len(measured) - len(complete),
        "unavailable_count": len(executions) - len(measured),
        "progress_intervals": sorted(
            {
                item["vm_step_progress_interval"]
                for item in measured
                if isinstance(item.get("vm_step_progress_interval"), int)
            }
        ),
        "complete_vm_steps_lower_bound": aggregate(values),
    }
