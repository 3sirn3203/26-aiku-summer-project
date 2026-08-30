from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from text2sql.config import ExecutionConfig
from text2sql.core.executor import execute_sql
from text2sql.core.models import ExecutionResult
from rl_finetune.rewards import (
    RewardConfig,
    RewardResult,
    score_completion,
)


@dataclass(frozen=True)
class TRLRewardRuntime:
    execution: ExecutionConfig
    reward: RewardConfig = RewardConfig()
    gold_error_reward: float = 0.0
    trace_path: Optional[Path] = None


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, Mapping):
            content = last.get("content")
            return content if isinstance(content, str) else str(content)
    return str(completion)


def _column(kwargs: Mapping[str, Any], key: str, count: int) -> List[Any]:
    if key not in kwargs:
        raise KeyError("TRL reward call is missing dataset column %r" % key)
    value = kwargs[key]
    if isinstance(value, list):
        if len(value) != count:
            raise ValueError(
                "dataset column %r has %d values for %d completions"
                % (key, len(value), count)
            )
        return value
    if isinstance(value, tuple):
        if len(value) != count:
            raise ValueError(
                "dataset column %r has %d values for %d completions"
                % (key, len(value), count)
            )
        return list(value)
    return [value for _ in range(count)]


def _append_trace(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def make_text2sql_reward_func(
    runtime: TRLRewardRuntime,
) -> Callable[..., List[float]]:
    """Return a GRPO-compatible reward function.

    TRL expects a list of floats.  If gold SQL execution fails, the detailed
    reward result reports ``reward=None`` and this adapter emits
    ``gold_error_reward`` while logging the infrastructure status.
    """

    gold_cache: Dict[str, ExecutionResult] = {}

    def gold_execution_for(db_path: Path, gold_sql: str) -> ExecutionResult:
        key = "%s\n%s" % (db_path.resolve(), gold_sql)
        cached = gold_cache.get(key)
        if cached is not None:
            return cached
        execution_args = {
            "timeout_seconds": runtime.execution.timeout_seconds,
            "max_sql_bytes": runtime.execution.max_sql_bytes,
            "max_result_rows": runtime.execution.max_result_rows,
            "max_result_bytes": runtime.execution.max_result_bytes,
            "worker_memory_limit_bytes": runtime.execution.worker_memory_limit_bytes,
        }
        result = execute_sql(db_path, gold_sql, **execution_args)
        gold_cache[key] = result
        return result

    def reward_func(completions: Sequence[Any], **kwargs: Any) -> List[float]:
        count = len(completions)
        db_paths = _column(kwargs, "db_path", count)
        gold_sqls = _column(kwargs, "gold_sql", count)
        order_sensitive_values = _column(kwargs, "order_sensitive", count)
        example_ids = (
            _column(kwargs, "example_id", count)
            if "example_id" in kwargs
            else [None for _ in range(count)]
        )

        rewards: List[float] = []
        statuses: List[str] = []
        correct_values: List[bool] = []
        latency_bonuses: List[float] = []
        trace_rows: List[Dict[str, Any]] = []

        for position, completion in enumerate(completions):
            db_path = Path(str(db_paths[position]))
            gold_sql = str(gold_sqls[position])
            result: RewardResult = score_completion(
                raw_output=_completion_text(completion),
                db_path=db_path,
                gold_sql=gold_sql,
                order_sensitive=bool(order_sensitive_values[position]),
                execution_config=runtime.execution,
                reward_config=runtime.reward,
                gold_execution=gold_execution_for(db_path, gold_sql),
            )
            reward = (
                runtime.gold_error_reward
                if result.reward is None
                else float(result.reward)
            )
            rewards.append(reward)
            statuses.append(result.status)
            correct_values.append(result.correct)
            latency_bonuses.append(float(result.latency_bonus))
            if runtime.trace_path is not None:
                trace_rows.append(
                    {
                        "example_id": example_ids[position],
                        "position": position,
                        "emitted_reward": reward,
                        "reward_result": result.to_dict(),
                    }
                )

        log_extra = kwargs.get("log_extra")
        if callable(log_extra):
            log_extra("text2sql_reward_status", statuses)
            log_extra("text2sql_correct", correct_values)
            log_extra("text2sql_latency_bonus", latency_bonuses)

        if runtime.trace_path is not None and trace_rows:
            _append_trace(runtime.trace_path, trace_rows)
        return rewards

    return reward_func

