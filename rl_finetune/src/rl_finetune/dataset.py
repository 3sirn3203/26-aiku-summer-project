from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from text2sql.config import AppConfig
from text2sql.core.schema import serialize_schema
from text2sql.core.spider import SpiderDataset
from text2sql.single_turn.prompt import build_messages


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _selected_indices(
    total: int,
    *,
    limit: Optional[int] = None,
    offset: int = 0,
    indices: Optional[Sequence[int]] = None,
) -> List[int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")
    if indices is not None:
        selected = list(indices)
        if any(index < 0 or index >= total for index in selected):
            raise ValueError("selected index is outside the dataset")
        if len(set(selected)) != len(selected):
            raise ValueError("selected indices must be unique")
        return selected[:limit] if limit is not None else selected
    selected = list(range(offset, total))
    return selected[:limit] if limit is not None else selected


def read_indices_json(path: Path) -> List[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in payload
    ):
        raise ValueError("indices file must contain a JSON list of integers")
    return payload


def build_prompt_records(
    config: AppConfig,
    *,
    limit: Optional[int] = None,
    offset: int = 0,
    indices: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Build TRL-compatible prompt rows without executing SQL.

    The returned records intentionally include gold SQL and DB paths only as
    reward metadata.  They are not inserted into the prompt.
    """

    dataset = SpiderDataset(config.spider)
    selected = _selected_indices(
        len(dataset.examples), limit=limit, offset=offset, indices=indices
    )
    records: List[Dict[str, Any]] = []
    for index in selected:
        example = dataset.get_example(index)
        schema_text = serialize_schema(dataset.get_schema(example.db_id))
        messages = build_messages(example.question, schema_text)
        records.append(
            {
                "prompt": messages,
                "example_id": "%s:%d" % (example.split, example.index),
                "split": example.split,
                "index": example.index,
                "db_id": example.db_id,
                "question": example.question,
                "db_path": str(dataset.database_path(example.db_id)),
                "gold_sql": example.gold_sql,
                "order_sensitive": bool(example.parsed_sql.get("orderBy")),
                "schema_sha256": _sha256_text(schema_text),
                "prompt_sha256": _sha256_bytes(_json_bytes(messages)),
            }
        )
    return records


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


