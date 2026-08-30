from __future__ import annotations

from pathlib import Path

from text2sql.config import ExecutionConfig
from text2sql.core.executor import execute_sql
from text2sql.core.models import ExecutionResult, ParseResult
from text2sql.core.sql_output import extract_sql
from rl_finetune.agentic_runtime.models import SQLObservation


class WorkflowInfrastructureError(RuntimeError):
    pass


class SQLWorkflowEnvironment:
    def __init__(
        self,
        execution_config: ExecutionConfig,
        *,
        max_observation_rows: int = 20,
        max_error_chars: int = 500,
    ) -> None:
        if max_observation_rows < 1:
            raise ValueError("max_observation_rows must be positive")
        if max_error_chars < 1:
            raise ValueError("max_error_chars must be positive")
        self.execution_config = execution_config
        self.max_observation_rows = max_observation_rows
        self.max_error_chars = max_error_chars

    def execute_draft(
        self, db_path: Path, raw_output: str
    ) -> tuple[ParseResult, ExecutionResult, SQLObservation]:
        parse_result = extract_sql(raw_output)
        if parse_result.status != "success" or parse_result.sql is None:
            execution = ExecutionResult(
                status="not_run",
                error_type=parse_result.error_type,
                error_message=parse_result.error_message,
            )
            return parse_result, execution, self._observation(execution)

        config = self.execution_config
        execution = execute_sql(
            Path(db_path),
            parse_result.sql,
            timeout_seconds=config.timeout_seconds,
            max_sql_bytes=config.max_sql_bytes,
            max_result_rows=config.max_result_rows,
            max_result_bytes=config.max_result_bytes,
            worker_memory_limit_bytes=config.worker_memory_limit_bytes,
        )
        if execution.error_type in {
            "database_not_found",
            "executor_internal_error",
            "executor_worker_crash",
        }:
            raise WorkflowInfrastructureError(
                "draft executor infrastructure failure: %s: %s"
                % (execution.error_type, execution.error_message or "")
            )
        return parse_result, execution, self._observation(execution)

    def _observation(self, execution: ExecutionResult) -> SQLObservation:
        rows = execution.rows[: self.max_observation_rows]
        return SQLObservation(
            status=execution.status,
            columns=tuple(execution.columns),
            rows=tuple(tuple(value for value in row) for row in rows),
            row_count=(
                execution.row_count
                if execution.status == "success" and execution.row_count is not None
                else (len(execution.rows) if execution.status == "success" else None)
            ),
            truncated=(
                execution.truncated
                or len(execution.rows) > self.max_observation_rows
            ),
            error_type=execution.error_type,
            error_message=(execution.error_message or "")[: self.max_error_chars]
            or None,
        )

