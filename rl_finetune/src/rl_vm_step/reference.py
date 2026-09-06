from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping

from text2sql.config import ExecutionConfig
from text2sql.core.executor import execute_sql
from text2sql.core.models import ExecutionResult
from rl_vm_step.measurement import VMStepMeasurementError, measurement_from_execution


REFERENCE_SCHEMA_VERSION = 1


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _execution_config_sha256(config: ExecutionConfig) -> str:
    encoded = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _database_identity(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _execution_kwargs(config: ExecutionConfig) -> Dict[str, Any]:
    return {
        "timeout_seconds": config.timeout_seconds,
        "max_sql_bytes": config.max_sql_bytes,
        "max_result_rows": config.max_result_rows,
        "max_result_bytes": config.max_result_bytes,
        "worker_memory_limit_bytes": config.worker_memory_limit_bytes,
    }


def _compact_execution(result: ExecutionResult) -> Dict[str, Any]:
    return {
        "status": result.status,
        "columns": list(result.columns),
        "row_count": result.row_count,
        "row_fingerprint_version": result.row_fingerprint_version,
        "ordered_rows_fingerprint": result.ordered_rows_fingerprint,
        "unordered_rows_fingerprint": result.unordered_rows_fingerprint,
        "vm_steps_lower_bound": result.vm_steps_lower_bound,
        "vm_steps_upper_bound_exclusive": result.vm_steps_upper_bound_exclusive,
        "vm_step_progress_interval": result.vm_step_progress_interval,
        "vm_step_measurement_complete": result.vm_step_measurement_complete,
        "error_type": result.error_type,
        "error_message": result.error_message,
    }


def build_gold_reference(
    record: Mapping[str, Any], execution_config: ExecutionConfig
) -> Dict[str, Any]:
    db_path = Path(str(record["db_path"]))
    gold_sql = str(record["gold_sql"])
    result = execute_sql(db_path, gold_sql, **_execution_kwargs(execution_config))
    payload: Dict[str, Any] = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "status": "invalid",
        "example_id": str(record.get("example_id", "")),
        "database": _database_identity(db_path),
        "gold_sql_sha256": _sha256_text(gold_sql),
        "order_sensitive": bool(record["order_sensitive"]),
        "execution_config_sha256": _execution_config_sha256(execution_config),
        "sqlite_version": sqlite3.sqlite_version,
        "execution": _compact_execution(result),
        "vm_step": None,
    }
    if not result.succeeded:
        payload["reason"] = "gold_execution_error"
        return payload
    try:
        measurement = measurement_from_execution(result)
    except VMStepMeasurementError as exc:
        payload["reason"] = "gold_vm_measurement_error: %s" % exc
        return payload
    fingerprint_fields = (
        result.row_count,
        result.row_fingerprint_version,
        result.ordered_rows_fingerprint,
        result.unordered_rows_fingerprint,
    )
    if any(value is None for value in fingerprint_fields):
        payload["reason"] = "gold_execution_has_incomplete_fingerprint"
        return payload
    payload["status"] = "ready"
    payload["vm_step"] = measurement.to_dict()
    return payload


def validate_gold_reference(
    reference: Mapping[str, Any],
    record: Mapping[str, Any],
    execution_config: ExecutionConfig,
) -> None:
    if reference.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise ValueError("unsupported gold reference schema")
    if reference.get("status") != "ready":
        raise ValueError("gold reference is not ready")
    if reference.get("gold_sql_sha256") != _sha256_text(str(record["gold_sql"])):
        raise ValueError("gold SQL does not match reference")
    if reference.get("order_sensitive") != bool(record["order_sensitive"]):
        raise ValueError("result ordering policy does not match reference")
    if reference.get("execution_config_sha256") != _execution_config_sha256(
        execution_config
    ):
        raise ValueError("execution configuration does not match reference")
    if reference.get("sqlite_version") != sqlite3.sqlite_version:
        raise ValueError("SQLite version does not match reference")
    database = reference.get("database")
    if not isinstance(database, Mapping):
        raise ValueError("gold reference has no database identity")
    if dict(database) != _database_identity(Path(str(record["db_path"]))):
        raise ValueError("database identity does not match reference")
    execution = reference.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("gold reference has no execution metadata")
    measurement_from_execution(execution)


def gold_execution_from_reference(reference: Mapping[str, Any]) -> ExecutionResult:
    execution = reference.get("execution")
    if not isinstance(execution, Mapping) or reference.get("status") != "ready":
        raise ValueError("gold reference is not ready")
    allowed = {
        "status",
        "columns",
        "row_count",
        "row_fingerprint_version",
        "ordered_rows_fingerprint",
        "unordered_rows_fingerprint",
        "vm_steps_lower_bound",
        "vm_steps_upper_bound_exclusive",
        "vm_step_progress_interval",
        "vm_step_measurement_complete",
        "error_type",
        "error_message",
    }
    kwargs = {key: value for key, value in execution.items() if key in allowed}
    kwargs["rows"] = []
    return ExecutionResult(**kwargs)
