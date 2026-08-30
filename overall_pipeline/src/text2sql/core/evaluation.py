from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Dict, List

from text2sql.core.models import ExecutionResult


def _canonical_row(row: List[Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def result_hash(result: ExecutionResult) -> str:
    payload = {
        "status": result.status,
        "columns": result.columns,
        "rows": result.rows,
        "truncated": result.truncated,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare_results(
    predicted: ExecutionResult,
    gold: ExecutionResult,
    order_sensitive: bool,
) -> Dict[str, Any]:
    if not predicted.succeeded or not gold.succeeded:
        return {
            "policy": "ordered" if order_sensitive else "unordered_multiset",
            "comparable": False,
            "ordered_match": False,
            "unordered_match": False,
            "result_match": False,
        }
    predicted_rows = [_canonical_row(row) for row in predicted.rows]
    gold_rows = [_canonical_row(row) for row in gold.rows]
    ordered_match = predicted_rows == gold_rows
    unordered_match = Counter(predicted_rows) == Counter(gold_rows)
    return {
        "policy": "ordered" if order_sensitive else "unordered_multiset",
        "comparable": True,
        "ordered_match": ordered_match,
        "unordered_match": unordered_match,
        "result_match": ordered_match if order_sensitive else unordered_match,
    }
