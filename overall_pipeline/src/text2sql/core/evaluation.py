from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from text2sql.core.models import ExecutionResult


ROW_FINGERPRINT_VERSION = 1
_FINGERPRINT_MODULUS = 1 << 256
_ORDERED_DOMAIN = b"text2sql:ordered-rows:v1\x00"
_UNORDERED_DOMAINS = (
    b"text2sql:unordered-row:v1:a\x00",
    b"text2sql:unordered-row:v1:b\x00",
)


def _canonical_row(row: List[Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_row_bytes(row: List[Any]) -> bytes:
    return _canonical_row(row).encode("utf-8")


class RowFingerprintAccumulator:
    """Compute bounded-memory fingerprints for a streamed SQL result.

    The ordered digest is a conventional SHA-256 over length-framed rows.  The
    unordered digest uses two independently domain-separated modular sums of
    per-row SHA-256 values, so row order is ignored while duplicate
    multiplicity is retained.  Row count is compared separately.
    """

    def __init__(self) -> None:
        self.row_count = 0
        self._ordered = hashlib.sha256(_ORDERED_DOMAIN)
        self._unordered = [0, 0]

    def add(self, row: List[Any]) -> bytes:
        encoded = canonical_row_bytes(row)
        self.add_encoded(encoded)
        return encoded

    def add_encoded(self, encoded: bytes) -> None:
        length_prefix = len(encoded).to_bytes(8, "big")
        self._ordered.update(length_prefix)
        self._ordered.update(encoded)
        for index, domain in enumerate(_UNORDERED_DOMAINS):
            row_hasher = hashlib.sha256(domain)
            row_hasher.update(length_prefix)
            row_hasher.update(encoded)
            digest = row_hasher.digest()
            self._unordered[index] = (
                self._unordered[index] + int.from_bytes(digest, "big")
            ) % _FINGERPRINT_MODULUS
        self.row_count += 1

    def fingerprints(self) -> Tuple[str, str]:
        unordered = ":".join("%064x" % value for value in self._unordered)
        return self._ordered.hexdigest(), unordered


@dataclass(frozen=True)
class _FingerprintView:
    row_count: int
    version: int
    ordered: str
    unordered: str


def _fingerprint_view(result: ExecutionResult) -> _FingerprintView:
    metadata = (
        result.row_count,
        result.row_fingerprint_version,
        result.ordered_rows_fingerprint,
        result.unordered_rows_fingerprint,
    )
    if all(value is not None for value in metadata):
        assert result.row_count is not None
        assert result.row_fingerprint_version is not None
        assert result.ordered_rows_fingerprint is not None
        assert result.unordered_rows_fingerprint is not None
        return _FingerprintView(
            row_count=result.row_count,
            version=result.row_fingerprint_version,
            ordered=result.ordered_rows_fingerprint,
            unordered=result.unordered_rows_fingerprint,
        )
    if any(value is not None for value in metadata) or result.truncated:
        raise ValueError("result has incomplete row fingerprint metadata")
    accumulator = RowFingerprintAccumulator()
    for row in result.rows:
        accumulator.add(row)
    ordered, unordered = accumulator.fingerprints()
    return _FingerprintView(
        row_count=accumulator.row_count,
        version=ROW_FINGERPRINT_VERSION,
        ordered=ordered,
        unordered=unordered,
    )


def result_hash(result: ExecutionResult) -> str:
    payload: Dict[str, Any] = {
        "status": result.status,
        "columns": result.columns,
    }
    if result.succeeded:
        try:
            fingerprint = _fingerprint_view(result)
        except ValueError:
            payload.update({"rows": result.rows, "truncated": result.truncated})
        else:
            payload.update(
                {
                    "row_count": fingerprint.row_count,
                    "row_fingerprint_version": fingerprint.version,
                    "ordered_rows_fingerprint": fingerprint.ordered,
                }
            )
    else:
        payload.update({"rows": result.rows, "truncated": result.truncated})
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
    try:
        predicted_fingerprint = _fingerprint_view(predicted)
        gold_fingerprint = _fingerprint_view(gold)
    except ValueError:
        return {
            "policy": "ordered" if order_sensitive else "unordered_multiset",
            "comparable": False,
            "ordered_match": False,
            "unordered_match": False,
            "result_match": False,
        }
    compatible = predicted_fingerprint.version == gold_fingerprint.version
    same_count = predicted_fingerprint.row_count == gold_fingerprint.row_count
    ordered_match = bool(
        compatible
        and same_count
        and predicted_fingerprint.ordered == gold_fingerprint.ordered
    )
    unordered_match = bool(
        compatible
        and same_count
        and predicted_fingerprint.unordered == gold_fingerprint.unordered
    )
    return {
        "policy": "ordered" if order_sensitive else "unordered_multiset",
        "comparable": compatible,
        "ordered_match": ordered_match,
        "unordered_match": unordered_match,
        "result_match": ordered_match if order_sensitive else unordered_match,
        "row_fingerprint_version": (
            predicted_fingerprint.version if compatible else None
        ),
    }
