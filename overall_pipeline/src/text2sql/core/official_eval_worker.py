from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping


_PROTOCOL_PREFIX = "__SPIDER_OFFICIAL_JSON__="
_EMPTY_SQL = {
    "except": None,
    "from": {"conds": [], "table_units": []},
    "groupBy": [],
    "having": [],
    "intersect": None,
    "limit": None,
    "orderBy": [],
    "select": [False, []],
    "union": None,
    "where": [],
}


def _error(operation: str, status: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "operation": operation,
        "status": status,
        "match": False,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc)[:2000],
        },
    }


def _load_official(request: Mapping[str, Any]) -> Any:
    evaluator_root = Path(str(request["evaluator_root"])).resolve()
    if not (evaluator_root / "evaluation.py").is_file():
        raise FileNotFoundError("official evaluation.py is missing")
    nltk_data_dir = request.get("nltk_data_dir")
    if nltk_data_dir:
        import nltk

        resolved = str(Path(str(nltk_data_dir)).resolve())
        if resolved not in nltk.data.path:
            nltk.data.path.insert(0, resolved)
    sys.path.insert(0, str(evaluator_root))
    import evaluation as official_evaluation

    return official_evaluation


def _database_path(request: Mapping[str, Any]) -> Path:
    database_root = Path(str(request["database_root"])).resolve()
    db_id = str(request["db_id"])
    candidate = (database_root / db_id / (db_id + ".sqlite")).resolve()
    try:
        candidate.relative_to(database_root)
    except ValueError as exc:
        raise ValueError("database path escapes the disposable root") from exc
    if not candidate.is_file():
        raise FileNotFoundError("base database is missing: %s" % candidate)
    return candidate


def _apply_worker_memory_limit(request: Mapping[str, Any]) -> None:
    """Apply the configured address-space cap before importing the evaluator.

    macOS is the local mock-only development environment, so RLIMIT_AS is
    intentionally enforced only in the Linux server worker.  A Linux worker
    fails closed if the limit cannot be installed.
    """

    raw_limit = request.get("worker_memory_limit_bytes")
    if isinstance(raw_limit, bool) or not isinstance(raw_limit, int) or raw_limit < 1:
        raise ValueError("worker_memory_limit_bytes must be a positive integer")
    if not sys.platform.startswith("linux"):
        return
    try:
        import resource
    except ImportError as exc:
        raise RuntimeError("Linux evaluator worker requires resource.RLIMIT_AS") from exc
    if not hasattr(resource, "RLIMIT_AS"):
        raise RuntimeError("Linux evaluator worker has no resource.RLIMIT_AS")
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
    target_limit = raw_limit
    if hard_limit != resource.RLIM_INFINITY:
        target_limit = min(target_limit, int(hard_limit))
    if soft_limit != resource.RLIM_INFINITY:
        target_limit = min(target_limit, int(soft_limit))
    if target_limit < 1:
        raise RuntimeError("Linux evaluator worker address-space limit is unusable")
    resource.setrlimit(resource.RLIMIT_AS, (target_limit, hard_limit))
    effective_soft_limit, _ = resource.getrlimit(resource.RLIMIT_AS)
    if (
        effective_soft_limit == resource.RLIM_INFINITY
        or int(effective_soft_limit) > raw_limit
    ):
        raise RuntimeError("Linux evaluator worker address-space limit was not applied")


def _score_exact(request: Mapping[str, Any], official: Any) -> Dict[str, Any]:
    operation = "exact"
    db_id = str(request["db_id"])
    db_path = _database_path(request)
    tables_path = Path(str(request["tables_path"])).resolve()
    gold_sql_text = str(request["gold_sql"])
    predicted_sql_text = str(request["predicted_sql"]).replace("value", "1")
    evaluator = official.Evaluator()
    try:
        schema = official.Schema(official.get_schema(str(db_path)))
        gold_sql = official.get_sql(schema, gold_sql_text)
        hardness = evaluator.eval_hardness(gold_sql)
        key_maps = official.build_foreign_key_map_from_json(str(tables_path))
        key_map = key_maps[db_id]
    except Exception as exc:
        payload = _error(operation, "gold_error", exc)
        payload["hardness"] = None
        return payload

    prediction_parse_error = None
    try:
        predicted_sql = official.get_sql(schema, predicted_sql_text)
    except Exception as exc:
        predicted_sql = dict(_EMPTY_SQL)
        predicted_sql["from"] = {"conds": [], "table_units": []}
        predicted_sql["select"] = [False, []]
        prediction_parse_error = {
            "type": type(exc).__name__,
            "message": str(exc)[:2000],
        }
    try:
        gold_valid_columns = official.build_valid_col_units(
            gold_sql["from"]["table_units"], schema
        )
        gold_sql = official.rebuild_sql_val(gold_sql)
        gold_sql = official.rebuild_sql_col(gold_valid_columns, gold_sql, key_map)
        predicted_valid_columns = official.build_valid_col_units(
            predicted_sql["from"]["table_units"], schema
        )
        predicted_sql = official.rebuild_sql_val(predicted_sql)
        predicted_sql = official.rebuild_sql_col(
            predicted_valid_columns, predicted_sql, key_map
        )
        match = bool(evaluator.eval_exact_match(predicted_sql, gold_sql))
    except Exception as exc:
        payload = _error(operation, "evaluator_error", exc)
        payload["hardness"] = hardness
        payload["prediction_parse_error"] = prediction_parse_error
        return payload
    return {
        "operation": operation,
        "status": "scored",
        "match": match,
        "hardness": hardness,
        "prediction_parse_error": prediction_parse_error,
        "error": None,
    }


def _score_test_suite(request: Mapping[str, Any], official: Any) -> Dict[str, Any]:
    operation = "test_suite"
    db_path = _database_path(request)
    try:
        score = official.eval_exec_match(
            db=str(db_path),
            p_str=str(request["predicted_sql"]).replace("value", "1"),
            g_str=str(request["gold_sql"]),
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
        )
    except AssertionError as exc:
        return _error(operation, "gold_error", exc)
    except Exception as exc:
        return _error(operation, "evaluator_error", exc)
    return {
        "operation": operation,
        "status": "scored",
        "match": bool(score),
        "error": None,
    }


def _dispatch(request: Mapping[str, Any]) -> Dict[str, Any]:
    operation = str(request.get("operation", "unknown"))
    if request.get("plug_value") is not False or request.get("keep_distinct") is not False:
        raise ValueError("worker requires plug_value=false and keep_distinct=false")
    official = _load_official(request)
    if operation == "exact":
        return _score_exact(request, official)
    if operation == "test_suite":
        return _score_test_suite(request, official)
    raise ValueError("unsupported operation: %s" % operation)


def main() -> int:
    started_ns = time.perf_counter_ns()
    operation = "unknown"
    try:
        request = json.loads(sys.stdin.read())
        if not isinstance(request, dict):
            raise TypeError("worker request must be a JSON object")
        operation = str(request.get("operation", "unknown"))
        _apply_worker_memory_limit(request)
        result = _dispatch(request)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        result = _error(operation, "evaluator_error", exc)
    result["worker_elapsed_ns"] = time.perf_counter_ns() - started_ns
    print(_PROTOCOL_PREFIX + json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
