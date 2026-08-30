from __future__ import annotations

import base64
import math
import multiprocessing
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from text2sql.core.evaluation import (
    ROW_FINGERPRINT_VERSION,
    RowFingerprintAccumulator,
    canonical_row_bytes,
)
from text2sql.core.models import ExecutionResult
from text2sql.core.sql_text import (
    function_calls,
    split_sql_statements,
    top_level_statement_keyword,
)


# Reuse the existing coarse timeout callback so instrumentation does not
# materially perturb the primary query latency.  This yields a bounded count,
# not sqlite3_stmt_status()'s exact SQLITE_STMTSTATUS_VM_STEP value.
_VM_STEP_PROGRESS_INTERVAL = 1000


def _action_codes(names: List[str]) -> Dict[int, str]:
    values: Dict[int, str] = {}
    for name in names:
        value = getattr(sqlite3, name, None)
        if isinstance(value, int):
            values[value] = name
    return values


_KNOWN_ACTIONS = _action_codes(
    [
        "SQLITE_CREATE_INDEX",
        "SQLITE_CREATE_TABLE",
        "SQLITE_CREATE_TEMP_INDEX",
        "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_TEMP_TRIGGER",
        "SQLITE_CREATE_TEMP_VIEW",
        "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_VIEW",
        "SQLITE_DELETE",
        "SQLITE_DROP_INDEX",
        "SQLITE_DROP_TABLE",
        "SQLITE_DROP_TEMP_INDEX",
        "SQLITE_DROP_TEMP_TABLE",
        "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_DROP_TEMP_VIEW",
        "SQLITE_DROP_TRIGGER",
        "SQLITE_DROP_VIEW",
        "SQLITE_INSERT",
        "SQLITE_PRAGMA",
        "SQLITE_TRANSACTION",
        "SQLITE_UPDATE",
        "SQLITE_ATTACH",
        "SQLITE_DETACH",
        "SQLITE_ALTER_TABLE",
        "SQLITE_REINDEX",
        "SQLITE_ANALYZE",
        "SQLITE_CREATE_VTABLE",
        "SQLITE_DROP_VTABLE",
        "SQLITE_SAVEPOINT",
        "SQLITE_SELECT",
        "SQLITE_READ",
        "SQLITE_FUNCTION",
        "SQLITE_RECURSIVE",
    ]
)
_ALLOWED_ACTION_CODES = {
    value
    for value in (
        getattr(sqlite3, "SQLITE_SELECT", None),
        getattr(sqlite3, "SQLITE_READ", None),
        getattr(sqlite3, "SQLITE_FUNCTION", None),
        getattr(sqlite3, "SQLITE_RECURSIVE", None),
    )
    if isinstance(value, int)
}
_UNSAFE_FUNCTIONS = {
    "edit",
    "eval",
    "format",
    "fts3_tokenizer",
    "load_extension",
    "printf",
    "randomblob",
    "readfile",
    "shell",
    "writefile",
    "zeroblob",
}
_PARENT_STARTUP_GRACE_SECONDS = 1.0
_WRITE_STATEMENT_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "REPLACE",
    "CREATE",
    "DROP",
    "ALTER",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "VACUUM",
    "REINDEX",
    "ANALYZE",
}


class _InvalidUtf8Text:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw


class _ResultValueTooLarge(Exception):
    pass


def _decode_sqlite_text(raw: bytes) -> Any:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return _InvalidUtf8Text(raw)


def _read_only_uri(path: Path) -> str:
    return "%s?mode=ro&cache=private" % path.resolve().as_uri()


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
    if isinstance(value, _InvalidUtf8Text):
        return {
            "__sqlite_text_base64__": base64.b64encode(value.raw).decode("ascii")
        }
    return {"__repr__": repr(value)}


def _error_result(
    status: str,
    error_type: str,
    message: str,
    elapsed: float = 0.0,
    denied_action: Optional[str] = None,
    query_elapsed_ns: Optional[int] = None,
    worker_elapsed_ns: Optional[int] = None,
    parent_elapsed_ns: Optional[int] = None,
    truncated: bool = False,
) -> ExecutionResult:
    return ExecutionResult(
        status=status,
        error_type=error_type,
        error_message=message[:2000],
        elapsed_seconds=elapsed,
        query_elapsed_ns=query_elapsed_ns,
        worker_elapsed_ns=worker_elapsed_ns,
        parent_elapsed_ns=parent_elapsed_ns,
        denied_action=denied_action,
        truncated=truncated,
    )


def preflight_read_only_sql(sql: str, max_sql_bytes: int) -> Optional[ExecutionResult]:
    """Apply DB-independent guards and return an error, or ``None`` if safe.

    Passing this preflight means only that the input is one bounded, read-only
    ``SELECT`` (possibly introduced by CTEs) without known side-effect or
    memory-amplifying functions. SQLite parsing, schema validity, runtime, and
    execution and bounded-payload rules are intentionally evaluated later.
    """

    if max_sql_bytes < 1:
        raise ValueError("max_sql_bytes must be positive")
    if not isinstance(sql, str) or not sql.strip():
        return _error_result("syntax_error", "syntax_error", "SQL is empty")
    if "\x00" in sql:
        return _error_result("syntax_error", "invalid_sql", "SQL contains a NUL byte")
    if len(sql.encode("utf-8")) > max_sql_bytes:
        return _error_result(
            "invalid_sql",
            "sql_length_limit",
            "SQL exceeds the configured byte limit",
        )
    blocked_functions = sorted(function_calls(sql) & _UNSAFE_FUNCTIONS)
    if blocked_functions:
        return _error_result(
            "unsafe_sql",
            "unsafe_sql",
            "Blocked SQLite function: %s" % blocked_functions[0],
            denied_action="SQLITE_FUNCTION:%s" % blocked_functions[0],
        )
    statements = split_sql_statements(sql)
    if len(statements) != 1:
        return _error_result(
            "unsafe_sql",
            "multiple_statements",
            "Exactly one SQL statement is allowed",
        )
    statement_keyword = top_level_statement_keyword(statements[0])
    if statement_keyword in _WRITE_STATEMENT_KEYWORDS:
        return _error_result(
            "unsafe_sql",
            "unsafe_sql",
            "Only a read-only SELECT statement is allowed",
            denied_action="SQLITE_STATEMENT:%s" % statement_keyword,
        )
    if statement_keyword != "SELECT":
        return _error_result(
            "syntax_error",
            "syntax_error",
            "SQL does not contain a recognizable read-only SELECT statement",
        )
    return None


def _classify_sqlite_error(
    exc: sqlite3.Error,
    elapsed: float,
    denied_actions: List[str],
    deadline_reached: bool,
    query_elapsed_ns: Optional[int],
) -> ExecutionResult:
    message = str(exc)
    lowered = message.casefold()
    if denied_actions:
        action = denied_actions[0]
        return _error_result(
            "unsafe_sql",
            "unsafe_sql",
            "SQLite authorizer rejected the statement",
            elapsed,
            denied_action=action,
            query_elapsed_ns=query_elapsed_ns,
        )
    if "not authorized" in lowered or "authorization denied" in lowered:
        return _error_result(
            "unsafe_sql",
            "unsafe_sql",
            "SQLite security policy rejected the statement",
            elapsed,
            denied_action="SQLITE_SECURITY_POLICY",
            query_elapsed_ns=query_elapsed_ns,
        )
    if deadline_reached or "interrupted" in lowered:
        return _error_result(
            "execution_timeout",
            "execution_timeout",
            "SQL execution exceeded its time limit",
            elapsed,
            query_elapsed_ns=query_elapsed_ns,
        )
    if "string or blob too big" in lowered or "too big" in lowered:
        return _error_result(
            "result_limit",
            "result_limit",
            "SQLite result exceeded an engine-level size limit",
            elapsed,
            query_elapsed_ns=query_elapsed_ns,
            truncated=True,
        )
    syntax_markers = (
        "syntax error",
        "incomplete input",
        "unrecognized token",
        "unterminated",
    )
    if any(marker in lowered for marker in syntax_markers):
        return _error_result(
            "syntax_error",
            "syntax_error",
            message,
            elapsed,
            query_elapsed_ns=query_elapsed_ns,
        )
    return _error_result(
        "execution_error",
        "execution_error",
        message,
        elapsed,
        query_elapsed_ns=query_elapsed_ns,
    )


def _execute_worker(
    send_connection: Any,
    db_path: str,
    sql: str,
    timeout_seconds: float,
    max_result_rows: int,
    max_result_bytes: int,
    worker_memory_limit_bytes: int,
) -> None:
    worker_started_ns = time.perf_counter_ns()
    deadline_ns: Optional[int] = None
    query_started_ns: Optional[int] = None
    query_elapsed_ns: Optional[int] = None
    vm_progress_callbacks = 0
    connection: Optional[sqlite3.Connection] = None
    denied_actions: List[str] = []
    try:
        try:
            import resource
        except ImportError as exc:
            if sys.platform.startswith("linux"):
                raise RuntimeError(
                    "Linux SQL worker requires resource.RLIMIT_AS"
                ) from exc
        else:
            if sys.platform.startswith("linux"):
                if not hasattr(resource, "RLIMIT_AS"):
                    raise RuntimeError("Linux SQL worker has no resource.RLIMIT_AS")
                soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
                target_limit = worker_memory_limit_bytes
                if hard_limit != resource.RLIM_INFINITY:
                    target_limit = min(target_limit, int(hard_limit))
                if soft_limit != resource.RLIM_INFINITY:
                    target_limit = min(target_limit, int(soft_limit))
                resource.setrlimit(resource.RLIMIT_AS, (target_limit, hard_limit))
                effective_soft_limit, _ = resource.getrlimit(resource.RLIMIT_AS)
                if (
                    effective_soft_limit == resource.RLIM_INFINITY
                    or int(effective_soft_limit) > worker_memory_limit_bytes
                ):
                    raise RuntimeError(
                        "Linux SQL worker address-space limit was not applied"
                    )

        path = Path(db_path)
        connection = sqlite3.connect(
            _read_only_uri(path),
            uri=True,
            timeout=min(timeout_seconds, 1.0),
            isolation_level=None,
            cached_statements=0,
        )
        connection.text_factory = _decode_sqlite_text
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        if hasattr(connection, "setlimit") and hasattr(sqlite3, "SQLITE_LIMIT_LENGTH"):
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, max_result_bytes)
        if hasattr(connection, "enable_load_extension"):
            connection.enable_load_extension(False)

        function_code = getattr(sqlite3, "SQLITE_FUNCTION", None)

        def authorizer(
            action_code: int,
            arg1: Optional[str],
            arg2: Optional[str],
            _database_name: Optional[str],
            _trigger_name: Optional[str],
        ) -> int:
            if function_code is not None and action_code == function_code:
                function_name = (arg2 or arg1 or "").casefold()
                if function_name in _UNSAFE_FUNCTIONS:
                    denied_actions.append("SQLITE_FUNCTION:%s" % function_name)
                    return sqlite3.SQLITE_DENY
            if action_code in _ALLOWED_ACTION_CODES:
                if action_code == getattr(sqlite3, "SQLITE_READ", None):
                    database_name = (_database_name or "").casefold()
                    if database_name not in {"main", ""}:
                        denied_actions.append("SQLITE_READ:%s" % database_name)
                        return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            denied_actions.append(_KNOWN_ACTIONS.get(action_code, "SQLITE_ACTION_%d" % action_code))
            return sqlite3.SQLITE_DENY

        connection.set_authorizer(authorizer)

        def progress_handler() -> int:
            nonlocal vm_progress_callbacks
            vm_progress_callbacks += 1
            return (
                1
                if deadline_ns is not None and time.perf_counter_ns() >= deadline_ns
                else 0
            )

        connection.set_progress_handler(
            progress_handler, _VM_STEP_PROGRESS_INTERVAL
        )
        query_started_ns = time.perf_counter_ns()
        deadline_ns = query_started_ns + int(timeout_seconds * 1_000_000_000)
        cursor = connection.execute(sql)
        columns = [str(item[0]) for item in cursor.description] if cursor.description else []
        rows: List[List[Any]] = []
        retained_payload_bytes = 0
        retain_prefix = True
        fingerprints = RowFingerprintAccumulator()
        while True:
            fetched = cursor.fetchmany(256)
            if not fetched:
                break
            for raw_row in fetched:
                row = [_json_safe(value) for value in raw_row]
                encoded = canonical_row_bytes(row)
                if len(encoded) > max_result_bytes:
                    raise _ResultValueTooLarge(
                        "One SQL result row exceeded %d encoded bytes"
                        % max_result_bytes
                    )
                fingerprints.add_encoded(encoded)
                if retain_prefix:
                    if (
                        len(rows) >= max_result_rows
                        or retained_payload_bytes + len(encoded) > max_result_bytes
                    ):
                        retain_prefix = False
                    else:
                        rows.append(row)
                        retained_payload_bytes += len(encoded)
        query_elapsed_ns = time.perf_counter_ns() - query_started_ns
        elapsed = (time.perf_counter_ns() - worker_started_ns) / 1_000_000_000
        ordered_fingerprint, unordered_fingerprint = fingerprints.fingerprints()
        result = ExecutionResult(
            status="success",
            rows=rows,
            columns=columns,
            row_count=fingerprints.row_count,
            row_fingerprint_version=ROW_FINGERPRINT_VERSION,
            ordered_rows_fingerprint=ordered_fingerprint,
            unordered_rows_fingerprint=unordered_fingerprint,
            elapsed_seconds=elapsed,
            query_elapsed_ns=query_elapsed_ns,
            truncated=fingerprints.row_count > len(rows),
        )
    except sqlite3.Error as exc:
        if query_started_ns is not None and query_elapsed_ns is None:
            query_elapsed_ns = time.perf_counter_ns() - query_started_ns
        elapsed = (time.perf_counter_ns() - worker_started_ns) / 1_000_000_000
        result = _classify_sqlite_error(
            exc,
            elapsed,
            denied_actions,
            deadline_reached=(
                deadline_ns is not None and time.perf_counter_ns() >= deadline_ns
            ),
            query_elapsed_ns=query_elapsed_ns,
        )
    except _ResultValueTooLarge as exc:
        if query_started_ns is not None and query_elapsed_ns is None:
            query_elapsed_ns = time.perf_counter_ns() - query_started_ns
        elapsed = (time.perf_counter_ns() - worker_started_ns) / 1_000_000_000
        result = _error_result(
            "result_limit",
            "result_limit",
            str(exc),
            elapsed,
            query_elapsed_ns=query_elapsed_ns,
            truncated=True,
        )
    except MemoryError:
        if query_started_ns is not None and query_elapsed_ns is None:
            query_elapsed_ns = time.perf_counter_ns() - query_started_ns
        elapsed = (time.perf_counter_ns() - worker_started_ns) / 1_000_000_000
        result = _error_result(
            "result_limit",
            "result_limit",
            "SQL worker exceeded its memory allowance",
            elapsed,
            query_elapsed_ns=query_elapsed_ns,
            truncated=True,
        )
    except Exception as exc:  # The parent still receives a structured worker failure.
        if query_started_ns is not None and query_elapsed_ns is None:
            query_elapsed_ns = time.perf_counter_ns() - query_started_ns
        elapsed = (time.perf_counter_ns() - worker_started_ns) / 1_000_000_000
        result = _error_result(
            "internal_error",
            "executor_internal_error",
            "%s: %s" % (type(exc).__name__, exc),
            elapsed,
            query_elapsed_ns=query_elapsed_ns,
        )
    finally:
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error:
                pass
    if query_started_ns is not None:
        result.vm_steps_lower_bound = (
            vm_progress_callbacks * _VM_STEP_PROGRESS_INTERVAL
        )
        result.vm_step_progress_interval = _VM_STEP_PROGRESS_INTERVAL
        result.vm_step_measurement_complete = result.status == "success"
        if result.vm_step_measurement_complete:
            result.vm_steps_upper_bound_exclusive = (
                result.vm_steps_lower_bound + _VM_STEP_PROGRESS_INTERVAL
            )
    result.worker_elapsed_ns = time.perf_counter_ns() - worker_started_ns
    # The parent replaces this compatibility value with its complete wall time.
    result.elapsed_seconds = result.worker_elapsed_ns / 1_000_000_000
    try:
        send_connection.send(result.to_dict())
    finally:
        send_connection.close()


def _stop_process(process: multiprocessing.Process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(0.5)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(0.5)


def execute_sql(
    db_path: Path,
    sql: str,
    timeout_seconds: float,
    max_sql_bytes: int,
    max_result_rows: int,
    max_result_bytes: int,
    worker_memory_limit_bytes: int = 1024 * 1024 * 1024,
) -> ExecutionResult:
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        return _error_result(
            "execution_error",
            "database_not_found",
            "SQLite database does not exist: %s" % path,
        )
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if (
        max_sql_bytes < 1
        or max_result_rows < 1
        or max_result_bytes < 1
        or worker_memory_limit_bytes < 1
    ):
        raise ValueError("execution and retained-result limits must be positive")
    preflight_error = preflight_read_only_sql(sql, max_sql_bytes)
    if preflight_error is not None:
        return preflight_error
    statements = split_sql_statements(sql)

    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_execute_worker,
        args=(
            send_connection,
            str(path),
            statements[0],
            timeout_seconds,
            max_result_rows,
            max_result_bytes,
            worker_memory_limit_bytes,
        ),
    )
    parent_started_ns = time.perf_counter_ns()
    process.start()
    send_connection.close()
    try:
        parent_hard_timeout = timeout_seconds + _PARENT_STARTUP_GRACE_SECONDS
        remaining = max(
            0.0,
            parent_hard_timeout
            - ((time.perf_counter_ns() - parent_started_ns) / 1_000_000_000),
        )
        if not receive_connection.poll(remaining):
            _stop_process(process)
            parent_elapsed_ns = time.perf_counter_ns() - parent_started_ns
            return _error_result(
                "execution_timeout",
                "execution_timeout",
                (
                    "SQL worker did not return within %.3f seconds of query time "
                    "plus %.3f seconds of startup grace"
                    % (timeout_seconds, _PARENT_STARTUP_GRACE_SECONDS)
                ),
                parent_elapsed_ns / 1_000_000_000,
                parent_elapsed_ns=parent_elapsed_ns,
            )
        try:
            payload = receive_connection.recv()
        except EOFError:
            process.join(0.5)
            parent_elapsed_ns = time.perf_counter_ns() - parent_started_ns
            return _error_result(
                "internal_error",
                "executor_worker_crash",
                "SQL worker exited without returning a result (exit code %s)"
                % process.exitcode,
                parent_elapsed_ns / 1_000_000_000,
                parent_elapsed_ns=parent_elapsed_ns,
            )
        process.join(0.5)
        if process.is_alive():
            _stop_process(process)
        parent_elapsed_ns = time.perf_counter_ns() - parent_started_ns
        result = ExecutionResult(**payload)
        result.parent_elapsed_ns = parent_elapsed_ns
        result.elapsed_seconds = parent_elapsed_ns / 1_000_000_000
        return result
    finally:
        receive_connection.close()
        if process.is_alive():
            _stop_process(process)
        process.close()
