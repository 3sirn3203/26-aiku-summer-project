from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from text2sql.config import ExecutionConfig
from text2sql.core.evaluation import compare_results
from text2sql.core.executor import execute_sql
from text2sql.core.models import ExecutionResult, ParseResult
from text2sql.core.sql_output import extract_sql


@dataclass(frozen=True)
class RewardConfig:
    """Correctness-first reward configuration for Text-to-SQL RL.

    The latency term is deliberately bounded so any correct SQL remains above
    every failure tier.  This keeps optimization pressure aligned with task
    success before query speed.
    """

    parse_failure_reward: float = -1.0
    unsafe_failure_reward: float = -1.0
    execution_failure_reward: float = -0.7
    result_mismatch_reward: float = -0.2
    correct_base_reward: float = 1.0
    latency_weight: float = 0.1
    latency_clip: float = 1.0
    timing_epsilon_ns: int = 1_000

    def __post_init__(self) -> None:
        values = (
            self.parse_failure_reward,
            self.unsafe_failure_reward,
            self.execution_failure_reward,
            self.result_mismatch_reward,
            self.correct_base_reward,
            self.latency_weight,
            self.latency_clip,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("reward values must be finite")
        if self.latency_weight < 0:
            raise ValueError("latency_weight must be non-negative")
        if self.latency_clip < 0:
            raise ValueError("latency_clip must be non-negative")
        if self.timing_epsilon_ns < 1:
            raise ValueError("timing_epsilon_ns must be positive")
        worst_correct = self.correct_base_reward - (
            self.latency_weight * self.latency_clip
        )
        best_failure = max(
            self.parse_failure_reward,
            self.unsafe_failure_reward,
            self.execution_failure_reward,
            self.result_mismatch_reward,
        )
        if worst_correct <= best_failure:
            raise ValueError(
                "correct SQL must keep a higher reward than every failure tier"
            )


@dataclass
class RewardResult:
    reward: Optional[float]
    status: str
    correct: bool
    latency_bonus: float = 0.0
    parsed_sql: Optional[str] = None
    parse_result: Mapping[str, Any] = field(default_factory=dict)
    predicted_execution: Mapping[str, Any] = field(default_factory=dict)
    gold_execution: Mapping[str, Any] = field(default_factory=dict)
    comparison: Mapping[str, Any] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _not_run(parse_result: ParseResult) -> ExecutionResult:
    return ExecutionResult(
        status="not_run",
        error_type=parse_result.error_type or "sql_parse_error",
        error_message=parse_result.error_message or "Prediction was not executable",
    )


def _execution_failure_reward(
    execution: ExecutionResult, config: RewardConfig
) -> float:
    if execution.status == "unsafe_sql":
        return config.unsafe_failure_reward
    return config.execution_failure_reward


def _latency_bonus(
    predicted: ExecutionResult,
    gold: ExecutionResult,
    config: RewardConfig,
) -> float:
    if predicted.query_elapsed_ns is None or gold.query_elapsed_ns is None:
        return 0.0
    if predicted.query_elapsed_ns < 0 or gold.query_elapsed_ns < 0:
        return 0.0
    ratio = (gold.query_elapsed_ns + config.timing_epsilon_ns) / (
        predicted.query_elapsed_ns + config.timing_epsilon_ns
    )
    normalized = math.log(ratio)
    normalized = max(-config.latency_clip, min(config.latency_clip, normalized))
    return config.latency_weight * normalized


def score_execution_pair(
    *,
    parse_result: ParseResult,
    predicted_execution: ExecutionResult,
    gold_execution: ExecutionResult,
    order_sensitive: bool,
    config: RewardConfig = RewardConfig(),
) -> RewardResult:
    """Score one parsed prediction against a gold execution result.

    Gold execution failures return ``reward=None`` because they indicate an
    infrastructure or dataset issue rather than model behavior.  Callers using
    TRL can translate that into a skipped sample.
    """

    comparison: Dict[str, Any] = {
        "comparable": False,
        "result_match": False,
    }
    if not gold_execution.succeeded:
        return RewardResult(
            reward=None,
            status="gold_execution_error",
            correct=False,
            parsed_sql=parse_result.sql,
            parse_result=parse_result.to_dict(),
            predicted_execution=predicted_execution.to_dict(),
            gold_execution=gold_execution.to_dict(),
            comparison=comparison,
            details={
                "reason": "gold SQL did not execute successfully",
                "gold_status": gold_execution.status,
            },
        )

    if parse_result.status != "success":
        return RewardResult(
            reward=config.parse_failure_reward,
            status="parse_error",
            correct=False,
            parsed_sql=None,
            parse_result=parse_result.to_dict(),
            predicted_execution=predicted_execution.to_dict(),
            gold_execution=gold_execution.to_dict(),
            comparison=comparison,
        )

    if not predicted_execution.succeeded:
        return RewardResult(
            reward=_execution_failure_reward(predicted_execution, config),
            status="prediction_execution_error",
            correct=False,
            parsed_sql=parse_result.sql,
            parse_result=parse_result.to_dict(),
            predicted_execution=predicted_execution.to_dict(),
            gold_execution=gold_execution.to_dict(),
            comparison=comparison,
            details={"prediction_status": predicted_execution.status},
        )

    comparison = compare_results(
        predicted_execution,
        gold_execution,
        order_sensitive=order_sensitive,
    )
    if not comparison.get("result_match"):
        return RewardResult(
            reward=config.result_mismatch_reward,
            status="result_mismatch",
            correct=False,
            parsed_sql=parse_result.sql,
            parse_result=parse_result.to_dict(),
            predicted_execution=predicted_execution.to_dict(),
            gold_execution=gold_execution.to_dict(),
            comparison=comparison,
        )

    bonus = _latency_bonus(predicted_execution, gold_execution, config)
    return RewardResult(
        reward=config.correct_base_reward + bonus,
        status="correct",
        correct=True,
        latency_bonus=bonus,
        parsed_sql=parse_result.sql,
        parse_result=parse_result.to_dict(),
        predicted_execution=predicted_execution.to_dict(),
        gold_execution=gold_execution.to_dict(),
        comparison=comparison,
        details={
            "latency_bonus_policy": (
                "bounded log(gold_query_elapsed_ns / predicted_query_elapsed_ns)"
            )
        },
    )


def score_completion(
    *,
    raw_output: str,
    db_path: Path,
    gold_sql: str,
    order_sensitive: bool,
    execution_config: ExecutionConfig,
    reward_config: RewardConfig = RewardConfig(),
    gold_execution: Optional[ExecutionResult] = None,
) -> RewardResult:
    """Parse, execute, compare, and reward one model completion."""

    parse_result = extract_sql(raw_output)
    execution_args = {
        "timeout_seconds": execution_config.timeout_seconds,
        "max_sql_bytes": execution_config.max_sql_bytes,
        "max_result_rows": execution_config.max_result_rows,
        "max_result_bytes": execution_config.max_result_bytes,
        "worker_memory_limit_bytes": execution_config.worker_memory_limit_bytes,
    }
    if parse_result.status == "success" and parse_result.sql is not None:
        predicted_execution = execute_sql(db_path, parse_result.sql, **execution_args)
    else:
        predicted_execution = _not_run(parse_result)
    if gold_execution is None:
        gold_execution = execute_sql(db_path, gold_sql, **execution_args)
    return score_execution_pair(
        parse_result=parse_result,
        predicted_execution=predicted_execution,
        gold_execution=gold_execution,
        order_sensitive=order_sensitive,
        config=reward_config,
    )


