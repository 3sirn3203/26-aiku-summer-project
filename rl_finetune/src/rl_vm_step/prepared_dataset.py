from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from text2sql.config import AppConfig
from rl_vm_step.reference import validate_gold_reference


PREPARED_DATASET_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid JSONL at %s:%d" % (path, line_number)
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError("prepared dataset rows must be JSON objects")
            records.append(payload)
    return records


def build_prepared_manifest(
    *,
    config: AppConfig,
    records: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    train_dataset_path: Path,
    gold_reference_path: Path,
    rejected_reference_path: Path,
    selection: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": PREPARED_DATASET_SCHEMA_VERSION,
        "status": "ready",
        "model": {
            "id": config.model.model_id,
            "revision": config.model.revision,
        },
        "execution_config": asdict(config.execution),
        "selection": {
            **dict(selection),
            "selected_count": len(records) + len(rejected),
            "eligible_count": len(records),
            "rejected_count": len(rejected),
        },
        "measurement": "sqlite_progress_handler_interval_midpoint",
        "files": {
            "train_dataset": {
                "path": train_dataset_path.name,
                "sha256": sha256_file(train_dataset_path),
            },
            "gold_reference": {
                "path": gold_reference_path.name,
                "sha256": sha256_file(gold_reference_path),
            },
            "rejected_gold_reference": {
                "path": rejected_reference_path.name,
                "sha256": sha256_file(rejected_reference_path),
            },
        },
    }


def load_prepared_dataset(directory: Path, config: AppConfig) -> tuple[
    List[Dict[str, Any]], Dict[str, Any]
]:
    directory = directory.expanduser().resolve()
    manifest_path = directory / "dataset_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("prepared dataset manifest does not exist: %s" % manifest_path) from exc
    except json.JSONDecodeError as exc:
        raise ValueError("prepared dataset manifest is invalid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("prepared dataset manifest must be an object")
    if manifest.get("schema_version") != PREPARED_DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported prepared dataset schema")
    if manifest.get("status") != "ready":
        raise ValueError("prepared dataset is not ready")
    expected_model = {"id": config.model.model_id, "revision": config.model.revision}
    if manifest.get("model") != expected_model:
        raise ValueError("prepared dataset model provenance does not match")
    if manifest.get("execution_config") != asdict(config.execution):
        raise ValueError("prepared dataset execution configuration does not match")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("prepared dataset manifest has no file provenance")
    resolved_files: Dict[str, Path] = {}
    for key in ("train_dataset", "gold_reference", "rejected_gold_reference"):
        item = files.get(key)
        if not isinstance(item, Mapping):
            raise ValueError("prepared dataset file entry %r is missing" % key)
        path = directory / str(item.get("path", ""))
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValueError("prepared dataset file %r failed integrity validation" % key)
        resolved_files[key] = path
    records = read_jsonl(resolved_files["train_dataset"])
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("prepared dataset selection metadata is missing")
    if len(records) != selection.get("eligible_count") or not records:
        raise ValueError("prepared dataset record count does not match manifest")
    for record in records:
        reference = record.get("gold_reference")
        if not isinstance(reference, Mapping):
            raise ValueError("prepared dataset row has no gold reference")
        validate_gold_reference(reference, record, config.execution)
    return records, manifest
