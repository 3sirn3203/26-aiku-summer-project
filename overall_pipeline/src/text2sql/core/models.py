from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class SpiderExample:
    index: int
    split: str
    db_id: str
    question: str
    gold_sql: str
    parsed_sql: Mapping[str, Any]


@dataclass(frozen=True)
class GenerationRequest:
    example_id: str
    messages: Sequence[Mapping[str, str]]


@dataclass(frozen=True)
class SchemaMetadata:
    db_id: str
    table_names: Sequence[str]
    column_names: Sequence[Sequence[Any]]
    column_types: Sequence[str]
    primary_keys: Sequence[Any]
    foreign_keys: Sequence[Sequence[int]]


@dataclass
class GenerationResult:
    status: str
    raw_output: str = ""
    elapsed_seconds: float = 0.0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    model_id: Optional[str] = None
    requested_revision: Optional[str] = None
    resolved_revision: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParseResult:
    status: str
    sql: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionResult:
    """Result of one isolated SQLite execution.

    ``query_elapsed_ns`` measures only ``Connection.execute`` plus result
    fetching. ``worker_elapsed_ns`` measures the worker from process entry
    until its result is ready, and ``parent_elapsed_ns`` measures the complete
    subprocess lifecycle observed by the caller.  A timing is ``None`` when
    that boundary was never reached (for example, SQL rejected before a
    worker starts or a worker killed before it can report).

    ``elapsed_seconds`` is retained for artifact compatibility.  Results
    returned by ``execute_sql`` define it as ``parent_elapsed_ns / 1e9``.

    Successful executions consume the complete cursor. ``rows`` contains only
    a configured prefix, while ``row_count`` and the fingerprints describe the
    complete result. ``truncated`` therefore means that the retained payload is
    incomplete, not that SQL execution itself was interrupted.
    """

    status: str
    rows: List[List[Any]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    # ``rows`` is a bounded prefix retained for observations and artifacts.
    # These fields describe the complete streamed result, including rows that
    # were not retained in memory.  Fingerprints are absent for executions
    # that did not finish successfully and for legacy/in-memory test results.
    row_count: Optional[int] = None
    row_fingerprint_version: Optional[int] = None
    ordered_rows_fingerprint: Optional[str] = None
    unordered_rows_fingerprint: Optional[str] = None
    elapsed_seconds: float = 0.0
    query_elapsed_ns: Optional[int] = None
    worker_elapsed_ns: Optional[int] = None
    parent_elapsed_ns: Optional[int] = None
    # The stdlib sqlite3 wrapper does not expose sqlite3_stmt_status().  The
    # executor therefore reports a progress-handler-derived VM-step interval.
    # For completed statements, actual VM steps are in
    # [lower_bound, upper_bound_exclusive).  Interrupted/capped statements
    # only have a lower bound.
    vm_steps_lower_bound: Optional[int] = None
    vm_steps_upper_bound_exclusive: Optional[int] = None
    vm_step_progress_interval: Optional[int] = None
    vm_step_measurement_complete: Optional[bool] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    truncated: bool = False
    denied_action: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    split: str
    example_count: int
    database_count: int
    schema_count: int
    validated_database_count: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload
