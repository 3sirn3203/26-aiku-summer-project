"""No torch import here: hard process timeout also covers blocking SQLite functions."""
from collections import Counter
import json
from pathlib import Path
import random
import resource
import sqlite3
import sys
import time

from .sql import validate_sql


def execute(path, sql, cfg, preview):
    validate_sql(sql, cfg["max_sql_chars"])
    started = time.monotonic()
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True,
                                 timeout=min(1.0, cfg["timeout_seconds"]))
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, min(cfg["max_result_bytes"], 1000000))
        connection.enable_load_extension(False)
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                   sqlite3.SQLITE_RECURSIVE}

        def authorize(action, arg1, arg2, db, trigger):
            if action not in allowed or (action == sqlite3.SQLITE_FUNCTION and
                                        str(arg2).lower() in {"load_extension", "writefile", "readfile"}):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        connection.set_progress_handler(
            lambda: int(time.monotonic() - started > cfg["timeout_seconds"]), 1000)
        cursor = connection.execute(sql)
        columns = [col[0] for col in cursor.description]
        rows, count, size = [], 0, 0
        for row in cursor:
            if time.monotonic() - started > cfg["timeout_seconds"]:
                return {"ok": False, "kind": "timeout", "error": "SQL execution timeout"}
            count += 1
            # Bound comparisons; do not silently compare only previews.
            size += len(repr(row).encode("utf-8"))
            if not preview and (count > cfg["max_result_rows"] or size > cfg["max_result_bytes"]):
                return {"ok": False, "kind": "result_limit",
                        "error": "Full result exceeds evaluation resource limit",
                        "row_count": count, "result_bytes": size,
                        "elapsed": time.monotonic() - started}
            if not preview or (count <= cfg["max_rows"] and size <= cfg["max_response_chars"]):
                rows.append(list(row))
        return {"ok": True, "columns": columns, "rows": rows, "row_count": count,
                "truncated": len(rows) < count, "elapsed": time.monotonic() - started}
    except sqlite3.Error as exc:
        timeout = "interrupt" in str(exc).lower()
        return {"ok": False, "kind": "timeout" if timeout else "sql_error",
                "error": "SQL execution timeout" if timeout else str(exc)[:500],
                "elapsed": time.monotonic() - started}
    finally:
        connection.close()


def run(payload):
    cfg = payload["config"]
    memory = cfg["worker_memory_mb"] * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    path, sql = payload["path"], payload["sql"]
    if payload["op"] == "execute":
        result = execute(path, sql, cfg, payload["preview"])
        # Blob display is JSON-safe; comparison keeps original bytes below.
        return result
    gold = execute(path, payload["gold"], cfg, False)
    if not gold.get("ok"):
        return {"infrastructure_error": f"Gold execution failed: {gold}"}
    pred = execute(path, sql, cfg, False)
    if pred.get("infrastructure_error"):
        return pred
    if pred.get("kind") == "result_limit":
        return {"ok": True, "match": False, "prediction_failure": pred}
    if not pred.get("ok"):
        return {"ok": True, "match": False}
    g, p = [tuple(row) for row in gold["rows"]], [tuple(row) for row in pred["rows"]]
    # Same order heuristic as Spider test-suite evaluator.
    ordered = "order by" in payload["gold"].lower()
    if cfg["evaluator_path"]:
        sys.path.insert(0, str(Path(cfg["evaluator_path"]).resolve()))
        from exec_eval import result_eq
        random.seed(0)
        match = result_eq(g, p, ordered)
    else:
        match = len(gold["columns"]) == len(pred["columns"]) and (
            g == p if ordered else Counter(g) == Counter(p))
    return {"ok": True, "match": bool(match)}


if __name__ == "__main__":
    try:
        output = run(json.load(sys.stdin))
    except Exception as exc:
        output = {"infrastructure_error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(output, ensure_ascii=False, default=lambda x: {"blob_hex": x.hex()} if isinstance(x, bytes) else str(x)))
