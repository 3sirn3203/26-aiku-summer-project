from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from text2sql.core.executor import preflight_read_only_sql


_PROTOCOL_PREFIX = "__SPIDER_OFFICIAL_JSON__="
_DB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_REQUIRED_EVALUATOR_FILES = (
    "evaluation.py",
    "process_sql.py",
    "exec_eval.py",
    "parse.py",
    "LICENSE",
    "UPSTREAM_COMMIT",
)
_PINNED_DEPENDENCIES = {
    "nltk": "3.9.1",
    "sqlparse": "0.5.3",
    "tqdm": "4.67.1",
}
_OFFICIAL_WORKER_THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
_PINNED_EVALUATOR_SHA256 = {
    "evaluation.py": "7401e4014a8955376a7919c06903a7f0ab403c99e89f94204cd8f4c8e32ae779",
    "process_sql.py": "927fc564f7a8e34f09f009a2f5564a83fdf95226440dde84c87871fd65fe55a1",
    "exec_eval.py": "29d034db28904490c28037537a14fbb0150b6e86cef0049076c0511d6b6b77f7",
    "parse.py": "ef04211a6e1c1e142571157f5c1999613e3451084c044083b2de1977f1f622c5",
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}


class OfficialEvaluationError(RuntimeError):
    """Raised when the pinned Spider evaluator cannot produce valid metrics."""


@dataclass(frozen=True)
class OfficialEvaluationItem:
    example_id: str
    db_id: str
    gold_sql: str
    predicted_sql: Optional[str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_db_directory(database_root: Path, db_id: str) -> Path:
    if _DB_ID_PATTERN.fullmatch(db_id) is None:
        raise OfficialEvaluationError("Invalid db_id for official evaluation: %r" % db_id)
    root = database_root.expanduser().resolve()
    candidate = (root / db_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise OfficialEvaluationError(
            "Official-evaluation DB path escapes its configured root"
        ) from exc
    return candidate


def _database_snapshot(database_root: Path, db_id: str) -> Dict[str, Any]:
    directory = _safe_db_directory(database_root, db_id)
    base_path = directory / (db_id + ".sqlite")
    variants = sorted(
        path for path in directory.iterdir() if path.is_file() and ".sqlite" in path.name
    ) if directory.is_dir() else []
    file_entries = [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in variants
    ]
    errors: List[str] = []
    if not directory.is_dir():
        errors.append("test-suite DB directory is missing: %s" % directory)
    if not base_path.is_file():
        errors.append("base test-suite DB is missing: %s" % base_path)
    if len(variants) < 2:
        errors.append(
            "db_id=%s has %d SQLite file(s); generated test-suite variants are missing"
            % (db_id, len(variants))
        )
    return {
        "db_id": db_id,
        "directory": str(directory),
        "base_database": str(base_path),
        "base_database_sha256": (
            _sha256_file(base_path) if base_path.is_file() else None
        ),
        "variant_count": len(variants),
        "total_size_bytes": sum(entry["size_bytes"] for entry in file_entries),
        "directory_sha256": _sha256_json(file_entries),
        "errors": errors,
    }


def validate_official_environment(
    *,
    evaluator_root: Path,
    database_root: Path,
    tables_path: Path,
    expected_commit: str,
    nltk_data_dir: Optional[Path],
    db_ids: Sequence[str],
) -> Dict[str, Any]:
    """Validate every pinned dependency and selected generated DB suite.

    This function never loads a model.  It deliberately checks only the DB IDs
    that the requested evaluation will use so its source hashes can be stored in
    the run manifest without hashing the multi-gigabyte bundle on every smoke run.
    """

    errors: List[str] = []
    warnings: List[str] = []
    evaluator_root = evaluator_root.expanduser().resolve()
    database_root = database_root.expanduser().resolve()
    tables_path = tables_path.expanduser().resolve()
    resolved_nltk_dir = (
        nltk_data_dir.expanduser().resolve() if nltk_data_dir is not None else None
    )

    evaluator_files: Dict[str, Dict[str, Any]] = {}
    for filename in _REQUIRED_EVALUATOR_FILES:
        path = evaluator_root / filename
        if not path.is_file():
            errors.append("official evaluator file is missing: %s" % path)
            continue
        file_sha256 = _sha256_file(path)
        evaluator_files[filename] = {
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256,
        }
        expected_sha256 = _PINNED_EVALUATOR_SHA256.get(filename)
        if expected_sha256 is not None and file_sha256 != expected_sha256:
            errors.append(
                "vendored evaluator source hash mismatch for %s: expected %s, found %s"
                % (filename, expected_sha256, file_sha256)
            )

    actual_commit: Optional[str] = None
    commit_path = evaluator_root / "UPSTREAM_COMMIT"
    if commit_path.is_file():
        actual_commit = commit_path.read_text(encoding="utf-8").strip()
        if actual_commit != expected_commit:
            errors.append(
                "vendored evaluator commit mismatch: expected %s, found %s"
                % (expected_commit, actual_commit)
            )

    dependency_versions: Dict[str, Optional[str]] = {}
    for distribution, expected_version in _PINNED_DEPENDENCIES.items():
        try:
            actual_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            actual_version = None
            errors.append(
                "official evaluator dependency is not installed: %s==%s"
                % (distribution, expected_version)
            )
        dependency_versions[distribution] = actual_version
        if actual_version is not None and actual_version != expected_version:
            errors.append(
                "official evaluator dependency mismatch: %s==%s is required, found %s"
                % (distribution, expected_version, actual_version)
            )

    nltk_resources: Dict[str, bool] = {"punkt": False, "punkt_tab": False}
    if resolved_nltk_dir is None:
        errors.append("official_evaluation.nltk_data_dir must be configured")
    elif not resolved_nltk_dir.is_dir():
        errors.append("NLTK data directory is missing: %s" % resolved_nltk_dir)
    elif dependency_versions.get("nltk") is not None:
        try:
            import nltk

            search_paths = [str(resolved_nltk_dir)] + list(nltk.data.path)
            for resource in nltk_resources:
                try:
                    nltk.data.find("tokenizers/%s" % resource, paths=search_paths)
                    nltk_resources[resource] = True
                except LookupError:
                    errors.append(
                        "NLTK tokenizer resource is missing: %s in %s"
                        % (resource, resolved_nltk_dir)
                    )
        except Exception as exc:
            errors.append("failed to inspect NLTK resources: %s: %s" % (type(exc).__name__, exc))

    tables_payload: Dict[str, Any] = {"path": str(tables_path)}
    if tables_path.is_file():
        tables_payload.update(
            {
                "size_bytes": tables_path.stat().st_size,
                "sha256": _sha256_file(tables_path),
            }
        )
    else:
        errors.append("Spider tables file is missing: %s" % tables_path)

    unique_db_ids: List[str] = []
    seen = set()
    for db_id in db_ids:
        if db_id in seen:
            continue
        seen.add(db_id)
        unique_db_ids.append(db_id)
    database_snapshots: Dict[str, Dict[str, Any]] = {}
    for db_id in unique_db_ids:
        try:
            snapshot = _database_snapshot(database_root, db_id)
        except Exception as exc:
            errors.append("db_id=%s preflight failed: %s" % (db_id, exc))
            continue
        database_snapshots[db_id] = snapshot
        errors.extend(snapshot["errors"])

    if not db_ids:
        warnings.append("No DB IDs were supplied for official-evaluation preflight")

    report = {
        "ok": not errors,
        "evaluator_root": str(evaluator_root),
        "database_root": str(database_root),
        "expected_commit": expected_commit,
        "actual_commit": actual_commit,
        "evaluator_files": evaluator_files,
        "evaluator_tree_sha256": _sha256_json(evaluator_files),
        "dependencies": dependency_versions,
        "nltk_data_dir": str(resolved_nltk_dir) if resolved_nltk_dir else None,
        "nltk_resources": nltk_resources,
        "tables": tables_payload,
        "databases": database_snapshots,
        "errors": errors,
        "warnings": warnings,
    }
    return report


def _worker_environment(nltk_data_dir: Optional[Path]) -> Dict[str, str]:
    environment = dict(os.environ)
    # The Linux evaluator worker runs under a 1 GiB RLIMIT_AS.  Numerical
    # libraries may otherwise create one BLAS thread per host CPU and exhaust
    # that address-space allowance during import, before evaluation begins.
    environment.update(_OFFICIAL_WORKER_THREAD_ENVIRONMENT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    source_root = str(Path(__file__).resolve().parents[2])
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root + os.pathsep + existing_python_path
        if existing_python_path
        else source_root
    )
    if nltk_data_dir is not None:
        configured = str(nltk_data_dir.resolve())
        existing = environment.get("NLTK_DATA")
        environment["NLTK_DATA"] = (
            configured + os.pathsep + existing if existing else configured
        )
    return environment


def _run_worker(
    operation: str,
    request: Mapping[str, Any],
    *,
    timeout_seconds: float,
    nltk_data_dir: Optional[Path],
) -> Dict[str, Any]:
    started_ns = time.perf_counter_ns()
    command = [sys.executable, "-m", "text2sql.core.official_eval_worker"]
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(request, ensure_ascii=False),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
            env=_worker_environment(nltk_data_dir),
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "evaluator_timeout",
            "match": False,
            "error": {
                "type": "evaluator_timeout",
                "message": "%s evaluation exceeded %.3f seconds"
                % (operation, timeout_seconds),
            },
            "parent_elapsed_ns": time.perf_counter_ns() - started_ns,
        }
    elapsed_ns = time.perf_counter_ns() - started_ns
    protocol_lines = [
        line[len(_PROTOCOL_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(_PROTOCOL_PREFIX)
    ]
    if completed.returncode != 0 or len(protocol_lines) != 1:
        return {
            "status": "evaluator_error",
            "match": False,
            "error": {
                "type": "worker_protocol_error",
                "message": (
                    "official evaluator worker exited with code %d; stderr=%s"
                    % (completed.returncode, completed.stderr[-2000:])
                )[:2000],
            },
            "parent_elapsed_ns": elapsed_ns,
        }
    try:
        payload = json.loads(protocol_lines[0])
    except (TypeError, json.JSONDecodeError) as exc:
        return {
            "status": "evaluator_error",
            "match": False,
            "error": {
                "type": "worker_protocol_error",
                "message": "invalid worker JSON: %s" % exc,
            },
            "parent_elapsed_ns": elapsed_ns,
        }
    if not isinstance(payload, dict) or payload.get("operation") != operation:
        return {
            "status": "evaluator_error",
            "match": False,
            "error": {
                "type": "worker_protocol_error",
                "message": "worker response operation mismatch",
            },
            "parent_elapsed_ns": elapsed_ns,
        }
    payload["parent_elapsed_ns"] = elapsed_ns
    if completed.stderr:
        payload["worker_stderr"] = completed.stderr[-2000:]
    return payload


def _validate_items(items: Sequence[OfficialEvaluationItem]) -> None:
    if not items:
        raise OfficialEvaluationError("Official evaluation requires at least one item")
    seen = set()
    for position, item in enumerate(items):
        if not isinstance(item, OfficialEvaluationItem):
            raise OfficialEvaluationError(
                "Official evaluation item %d has the wrong type" % position
            )
        if not item.example_id or item.example_id in seen:
            raise OfficialEvaluationError(
                "Official evaluation example IDs must be non-empty and unique"
            )
        seen.add(item.example_id)
        _safe_db_directory(Path("."), item.db_id)
        if not item.gold_sql.strip():
            raise OfficialEvaluationError(
                "Gold SQL is empty for example %s" % item.example_id
            )


def evaluate_official(
    items: Sequence[OfficialEvaluationItem],
    *,
    evaluator_root: Path,
    database_root: Path,
    tables_path: Path,
    expected_commit: str,
    nltk_data_dir: Optional[Path],
    timeout_seconds: float,
    max_sql_bytes: int = 100000,
    worker_memory_limit_bytes: int = 1024 * 1024 * 1024,
    plug_value: bool = False,
    keep_distinct: bool = False,
    preflight_report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Score a fixed ordered batch with the pinned official Spider primitives.

    Exact Set Match and Test Suite Accuracy run in separate subprocesses so a
    timeout in generated-DB execution cannot erase a completed structural score.
    Every test-suite run receives a disposable copy of its DB directory.
    """

    _validate_items(items)
    if (
        isinstance(worker_memory_limit_bytes, bool)
        or not isinstance(worker_memory_limit_bytes, int)
        or worker_memory_limit_bytes < 1
    ):
        raise OfficialEvaluationError(
            "Official evaluator worker memory limit must be a positive integer"
        )
    if plug_value or keep_distinct:
        raise OfficialEvaluationError(
            "This project requires plug_value=false and keep_distinct=false"
        )
    setup = dict(preflight_report) if preflight_report is not None else validate_official_environment(
        evaluator_root=evaluator_root,
        database_root=database_root,
        tables_path=tables_path,
        expected_commit=expected_commit,
        nltk_data_dir=nltk_data_dir,
        db_ids=[item.db_id for item in items],
    )
    if not setup.get("ok"):
        raise OfficialEvaluationError(
            "Official evaluator preflight failed: %s"
            % "; ".join(str(error) for error in setup.get("errors", []))
        )

    expected_setup_values = {
        "evaluator_root": str(evaluator_root.resolve()),
        "database_root": str(database_root.resolve()),
        "actual_commit": expected_commit,
    }
    for key, expected_value in expected_setup_values.items():
        if setup.get(key) != expected_value:
            raise OfficialEvaluationError(
                "Official evaluator preflight is stale or mismatched for %s" % key
            )
    setup_tables = setup.get("tables")
    if not isinstance(setup_tables, Mapping) or setup_tables.get("path") != str(
        tables_path.resolve()
    ):
        raise OfficialEvaluationError(
            "Official evaluator preflight is stale or mismatched for tables.json"
        )
    setup_databases = setup.get("databases")
    if not isinstance(setup_databases, Mapping) or any(
        item.db_id not in setup_databases for item in items
    ):
        raise OfficialEvaluationError(
            "Official evaluator preflight does not cover every requested DB"
        )

    evaluator_root = evaluator_root.resolve()
    database_root = database_root.resolve()
    tables_path = tables_path.resolve()
    results: List[Dict[str, Any]] = []
    for item in items:
        source_db_directory = _safe_db_directory(database_root, item.db_id)
        normalized_prediction = (
            item.predicted_sql.replace("value", "1")
            if item.predicted_sql is not None
            else None
        )
        if normalized_prediction is None or not normalized_prediction.strip():
            test_suite_eligibility = {
                "eligible": False,
                "status": "prediction_unavailable",
                "reason": "no parsed prediction SQL is available",
            }
        else:
            safety_error = preflight_read_only_sql(
                normalized_prediction, max_sql_bytes=max_sql_bytes
            )
            test_suite_eligibility = {
                "eligible": safety_error is None,
                "status": "eligible" if safety_error is None else "ineligible",
                "reason": (
                    "one bounded read-only SELECT after official normalization"
                    if safety_error is None
                    else safety_error.error_message
                ),
                "safety_error": (
                    safety_error.to_dict() if safety_error is not None else None
                ),
            }
        normalization = {
            "rule": "case-sensitive global str.replace('value', '1')",
            "applied": (
                item.predicted_sql is not None
                and normalized_prediction != item.predicted_sql
            ),
            "normalized_sql_sha256": (
                hashlib.sha256(normalized_prediction.encode("utf-8")).hexdigest()
                if normalized_prediction is not None
                else None
            ),
        }
        with tempfile.TemporaryDirectory(prefix="spider-official-") as temp_directory:
            copied_root = Path(temp_directory) / "database"
            copied_db_directory = copied_root / item.db_id
            copied_root.mkdir()
            shutil.copytree(source_db_directory, copied_db_directory)
            common_request = {
                "evaluator_root": str(evaluator_root),
                "database_root": str(copied_root),
                "tables_path": str(tables_path),
                "nltk_data_dir": str(nltk_data_dir.resolve()) if nltk_data_dir else None,
                "example_id": item.example_id,
                "db_id": item.db_id,
                "gold_sql": item.gold_sql,
                "predicted_sql": item.predicted_sql,
                "plug_value": False,
                "keep_distinct": False,
                "worker_memory_limit_bytes": worker_memory_limit_bytes,
            }
            if item.predicted_sql is None or not item.predicted_sql.strip():
                exact_result: Dict[str, Any] = {
                    "status": "prediction_unavailable",
                    "match": False,
                    "hardness": None,
                    "error": None,
                    "parent_elapsed_ns": None,
                }
            else:
                exact_request = dict(common_request, operation="exact")
                exact_result = _run_worker(
                    "exact",
                    exact_request,
                    timeout_seconds=timeout_seconds,
                    nltk_data_dir=nltk_data_dir,
                )

            if not test_suite_eligibility["eligible"]:
                test_suite_result: Dict[str, Any] = {
                    "status": (
                        "prediction_unavailable"
                        if test_suite_eligibility["status"]
                        == "prediction_unavailable"
                        else "prediction_ineligible"
                    ),
                    "match": False,
                    "error": None,
                    "parent_elapsed_ns": None,
                }
            elif item.predicted_sql is None or not item.predicted_sql.strip():
                test_suite_result = {
                    "status": "prediction_unavailable",
                    "match": False,
                    "error": None,
                    "parent_elapsed_ns": None,
                }
            else:
                test_suite_request = dict(common_request, operation="test_suite")
                test_suite_result = _run_worker(
                    "test_suite",
                    test_suite_request,
                    timeout_seconds=timeout_seconds,
                    nltk_data_dir=nltk_data_dir,
                )

        result = {
            "example_id": item.example_id,
            "db_id": item.db_id,
            "prediction_normalization": normalization,
            "test_suite_eligibility": test_suite_eligibility,
            "exact_set_match": exact_result,
            "test_suite": test_suite_result,
        }
        results.append(result)

    returned_ids = [result["example_id"] for result in results]
    expected_ids = [item.example_id for item in items]
    if returned_ids != expected_ids or len(results) != len(items):
        raise OfficialEvaluationError(
            "Official evaluator changed the requested example count or order"
        )
    infrastructure_statuses = {
        "evaluator_error",
        "evaluator_timeout",
        "gold_error",
    }
    metric_validity: Dict[str, Dict[str, Any]] = {}
    for metric in ("exact_set_match", "test_suite"):
        statuses = [result[metric]["status"] for result in results]
        infrastructure_failures = sum(
            status in infrastructure_statuses for status in statuses
        )
        metric_validity[metric] = {
            "valid": infrastructure_failures == 0 and len(statuses) == len(items),
            "classified_examples": len(statuses) - infrastructure_failures,
            "infrastructure_failures": infrastructure_failures,
        }
    ok = all(metric["valid"] for metric in metric_validity.values())
    return {
        "schema_version": 1,
        "ok": ok,
        "total_examples": len(items),
        "processed_examples": len(results),
        "metrics": metric_validity,
        "policy": {
            "test_suite_metric": "Spider Test Suite Accuracy",
            "exact_metric": "Spider Original Exact Set Match",
            "plug_value": False,
            "keep_distinct": False,
            "test_suite_requires_adapter_read_only_preflight": True,
            "test_suite_safety_gate": (
                "adapter-verified bounded single read-only SELECT after official normalization"
            ),
            "test_suite_uses_disposable_database_copy": True,
            "official_worker_memory_limit_bytes": worker_memory_limit_bytes,
            "official_worker_memory_limit_platform": "Linux RLIMIT_AS; fail closed",
            "official_worker_thread_environment": dict(
                _OFFICIAL_WORKER_THREAD_ENVIRONMENT
            ),
            "prediction_normalization": (
                "case-sensitive global str.replace('value', '1')"
            ),
            "official_evaluator_runtime_included_in_sql_execution_time": False,
        },
        "evaluator": {
            "root": str(evaluator_root),
            "expected_commit": expected_commit,
            "actual_commit": setup.get("actual_commit"),
            "source_tree_sha256": setup.get("evaluator_tree_sha256"),
            "tables_sha256": setup.get("tables", {}).get("sha256"),
        },
        "results": results,
    }


def item_to_dict(item: OfficialEvaluationItem) -> Dict[str, Any]:
    """Return a JSON-safe item representation for diagnostics and tests."""

    return asdict(item)
