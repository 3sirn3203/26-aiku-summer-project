from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from text2sql.config import ExecutionConfig
from rl_vm_step.config import VMStepRewardConfig
from rl_vm_step.reference import gold_execution_from_reference, validate_gold_reference
from rl_vm_step.reward import score_vm_step_completion


@dataclass(frozen=True)
class TRLVMStepRewardRuntime:
    execution: ExecutionConfig
    reward: VMStepRewardConfig = VMStepRewardConfig()
    trace_path: Optional[Path] = None
    strict_measurement: bool = True


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
    if isinstance(value, (list, tuple)):
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


def make_vm_step_reward_func(
    runtime: TRLVMStepRewardRuntime,
) -> Callable[..., List[float]]:
    def reward_func(completions: Sequence[Any], **kwargs: Any) -> List[float]:
        count = len(completions)
        db_paths = _column(kwargs, "db_path", count)
        gold_sqls = _column(kwargs, "gold_sql", count)
        references = _column(kwargs, "gold_reference", count)
        order_values = _column(kwargs, "order_sensitive", count)
        example_ids = (
            _column(kwargs, "example_id", count)
            if "example_id" in kwargs
            else [None] * count
        )
        rewards: List[float] = []
        trace_rows: List[Dict[str, Any]] = []
        statuses: List[str] = []
        vm_bonuses: List[float] = []
        correct_values: List[bool] = []
        for position, completion in enumerate(completions):
            reference = references[position]
            if not isinstance(reference, Mapping):
                raise ValueError("gold_reference must be a mapping")
            validation_record = {
                "db_path": str(db_paths[position]),
                "gold_sql": str(gold_sqls[position]),
                "order_sensitive": bool(order_values[position]),
            }
            validate_gold_reference(reference, validation_record, runtime.execution)
            result = score_vm_step_completion(
                raw_output=_completion_text(completion),
                db_path=Path(str(db_paths[position])),
                gold_execution=gold_execution_from_reference(reference),
                order_sensitive=bool(order_values[position]),
                execution_config=runtime.execution,
                reward_config=runtime.reward,
            )
            if result.reward is None:
                message = "unscorable VM-step completion: %s" % result.status
                if runtime.strict_measurement:
                    raise RuntimeError(message)
                reward = runtime.reward.correct_base_reward if result.correct else 0.0
            else:
                reward = float(result.reward)
            rewards.append(reward)
            statuses.append(result.status)
            vm_bonuses.append(result.vm_bonus)
            correct_values.append(result.correct)
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
            log_extra("text2sql_vm_step_reward_status", statuses)
            log_extra("text2sql_correct", correct_values)
            log_extra("text2sql_vm_step_bonus", vm_bonuses)
        if runtime.trace_path is not None and trace_rows:
            _append_trace(runtime.trace_path, trace_rows)
        return rewards

    return reward_func
