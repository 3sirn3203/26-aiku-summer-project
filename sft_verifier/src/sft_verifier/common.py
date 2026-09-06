from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from text2sql.core.models import SchemaMetadata
from text2sql.core.schema import serialize_schema

from . import DATA_SCHEMA_VERSION


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
DEFAULT_SPIDER_ROOT = REPOSITORY_ROOT / "data" / "spider_data"
DEFAULT_TEST_SUITE_ROOT = REPOSITORY_ROOT / "data" / "spider_test_suite" / "database"
DEFAULT_EVALUATOR_ROOT = (
    REPOSITORY_ROOT / "overall_pipeline" / "vendor" / "spider_test_suite_eval"
)
DEFAULT_NLTK_DATA = REPOSITORY_ROOT / "data" / "nltk_data"
UPSTREAM_COMMIT = "e97acc546ecbee8fa27fa8dbf025ef61493a876c"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("%s:%d is not a JSON object" % (path, line_number))
            records.append(payload)
    return records


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_bucket(value: str, seed: int) -> int:
    digest = hashlib.sha256((str(seed) + "\0" + value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def load_schemas(tables_path: Path) -> Dict[str, str]:
    payload = read_json(tables_path)
    if not isinstance(payload, list):
        raise ValueError("tables.json must contain a list")
    result: Dict[str, str] = {}
    for item in payload:
        metadata = SchemaMetadata(
            db_id=str(item["db_id"]),
            table_names=tuple(item["table_names_original"]),
            column_names=tuple(tuple(value) for value in item["column_names_original"]),
            column_types=tuple(item["column_types"]),
            primary_keys=tuple(item["primary_keys"]),
            foreign_keys=tuple(tuple(value) for value in item["foreign_keys"]),
        )
        result[metadata.db_id] = serialize_schema(metadata)
    return result


def load_spider_train(spider_root: Path) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    index = 0
    for filename in ("train_spider.json", "train_others.json"):
        payload = read_json(spider_root / filename)
        if not isinstance(payload, list):
            raise ValueError("%s must contain a list" % filename)
        for source_index, item in enumerate(payload):
            examples.append(
                {
                    "example_id": "train:%d" % index,
                    "index": index,
                    "source_file": filename,
                    "source_index": source_index,
                    "db_id": item["db_id"],
                    "question": item["question"],
                    "gold_sql": item["query"],
                    "parsed_gold_sql": item.get("sql", {}),
                }
            )
            index += 1
    return examples


def select_examples(
    manifest: Mapping[str, Any], split: str, limit: Optional[int] = None
) -> Iterator[Dict[str, Any]]:
    count = 0
    for item in manifest.get("examples", []):
        if item.get("split") != split:
            continue
        yield dict(item)
        count += 1
        if limit is not None and count >= limit:
            break


def dataset_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_split: Dict[str, int] = {}
    by_decision: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    by_category: Dict[str, int] = {}
    for record in records:
        split = str(record.get("split", "unknown"))
        by_split[split] = by_split.get(split, 0) + 1
        label = record.get("label", {})
        if isinstance(label, Mapping):
            decision = str(label.get("decision", "unknown"))
            by_decision[decision] = by_decision.get(decision, 0) + 1
        source = str(record.get("source", "unknown"))
        by_source[source] = by_source.get(source, 0) + 1
        category = str(record.get("error_category", "none"))
        by_category[category] = by_category.get(category, 0) + 1
    return {
        "schema_version": DATA_SCHEMA_VERSION,
        "records": len(records),
        "by_split": by_split,
        "by_decision": by_decision,
        "by_source": by_source,
        "by_error_category": by_category,
    }
