from __future__ import annotations

import base64
import json
import math
from typing import Any, Dict, List

from text2sql.core.models import ExecutionResult


MAX_OBSERVATION_ROWS = 5
MAX_OBSERVATION_BYTES = 4096


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, bytes):
        return {"__bytes_base64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return {"__repr__": repr(value)}


def _encoded_size(payload: Dict[str, Any]) -> int:
    return len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _truncate_utf8(value: str, byte_limit: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= byte_limit:
        return value
    suffix = "..."
    suffix_bytes = len(suffix.encode("utf-8"))
    if byte_limit <= suffix_bytes:
        return suffix[:byte_limit]
    prefix = encoded[: byte_limit - suffix_bytes]
    while prefix:
        try:
            return prefix.decode("utf-8") + suffix
        except UnicodeDecodeError:
            prefix = prefix[:-1]
    return suffix


def bound_execution_observation(
    execution: ExecutionResult,
    max_rows: int = MAX_OBSERVATION_ROWS,
    max_bytes: int = MAX_OBSERVATION_BYTES,
) -> Dict[str, Any]:
    """Return a JSON-safe prefix observation with a hard encoded-byte bound."""

    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if max_bytes < 512:
        raise ValueError("max_bytes must be at least 512")

    source_rows = list(execution.rows)
    selected_rows: List[Any] = [_json_safe(row) for row in source_rows[:max_rows]]
    payload: Dict[str, Any] = {
        "status": _truncate_utf8(str(execution.status), 256),
        "error_type": (
            _truncate_utf8(str(execution.error_type), 256)
            if execution.error_type is not None
            else None
        ),
        "error_message": (
            _truncate_utf8(str(execution.error_message), 1024)
            if execution.error_message is not None
            else None
        ),
        "columns": [_truncate_utf8(str(column), 256) for column in execution.columns],
        "row_count": len(source_rows) if execution.status == "success" else None,
        "rows": selected_rows,
        "query_elapsed_ns": execution.query_elapsed_ns,
        "vm_steps_lower_bound": execution.vm_steps_lower_bound,
        "vm_steps_upper_bound_exclusive": (
            execution.vm_steps_upper_bound_exclusive
        ),
        "vm_step_progress_interval": execution.vm_step_progress_interval,
        "vm_step_measurement_complete": (
            execution.vm_step_measurement_complete
        ),
        "truncated": bool(execution.truncated or len(source_rows) > max_rows),
    }
    if _encoded_size(payload) <= max_bytes:
        return payload

    payload["truncated"] = True
    while payload["rows"] and _encoded_size(payload) > max_bytes:
        payload["rows"].pop()
    while payload["columns"] and _encoded_size(payload) > max_bytes:
        payload["columns"].pop()
    if _encoded_size(payload) > max_bytes and payload["error_message"]:
        original = str(payload["error_message"])
        low, high = 0, len(original.encode("utf-8"))
        while low < high:
            middle = (low + high + 1) // 2
            payload["error_message"] = _truncate_utf8(original, middle)
            if _encoded_size(payload) <= max_bytes:
                low = middle
            else:
                high = middle - 1
        payload["error_message"] = _truncate_utf8(original, low)
    if _encoded_size(payload) > max_bytes:
        payload["error_message"] = None
    if _encoded_size(payload) > max_bytes:
        payload["status"] = _truncate_utf8(str(payload["status"]), 64)
        if payload["error_type"] is not None:
            payload["error_type"] = _truncate_utf8(str(payload["error_type"]), 64)
    if _encoded_size(payload) > max_bytes:
        raise RuntimeError("Unable to fit the execution observation within its byte limit")
    return payload
