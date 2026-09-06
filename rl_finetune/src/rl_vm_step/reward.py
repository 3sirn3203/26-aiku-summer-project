from __future__ import annotations

import math
from pathlib import Path
from text2sql.config import ExecutionConfig
from text2sql.core.executor import execute_sql
from text2sql.core.models import ExecutionResult, ParseResult
from text2sql.core.sql_output import extract_sql
from rl_finetune.rewards import RewardConfig, score_execution_pair
from rl_vm_step.config import VMStepRewardConfig
from rl_vm_step.measurement import VMStepMeasurementError, measurement_from_execution
from rl_vm_step.models import VMStepRewardResult


def _base_config(config: VMStepRewardConfig) -> RewardConfig:
    return RewardConfig(
        parse_failure_reward=config.parse_failure_reward,
        unsafe_failure_reward=config.unsafe_failure_reward,
        execution_failure_reward=config.execution_failure_reward,
        result_mismatch_reward=config.result_mismatch_reward,
        correct_base_reward=config.correct_base_reward,
        latency_weight=0.0,
    )


def score_vm_step_execution_pair(
    *,
    parse_result: ParseResult,
    predicted_execution: ExecutionResult,
    gold_execution: ExecutionResult,
    order_sensitive: bool,
    config: VMStepRewardConfig = VMStepRewardConfig(),
) -> VMStepRewardResult:
    base = score_execution_pair(
        parse_result=parse_result,
        predicted_execution=predicted_execution,
        gold_execution=gold_execution,
        order_sensitive=order_sensitive,
        config=_base_config(config),
    )
    common = {
        "parsed_sql": base.parsed_sql,
        "parse_result": base.parse_result,
        "predicted_execution": base.predicted_execution,
        "gold_execution": base.gold_execution,
        "comparison": base.comparison,
    }
    if base.reward is None:
        return VMStepRewardResult(
            reward=None,
            status=base.status,
            correct=False,
            correctness_reward=None,
            details=base.details,
            **common,
        )
    if not base.correct:
        return VMStepRewardResult(
            reward=float(base.reward),
            status=base.status,
            correct=False,
            correctness_reward=float(base.reward),
            details={"vm_bonus_applied": False, **dict(base.details)},
            **common,
        )
    try:
        reference_vm = measurement_from_execution(gold_execution)
    except VMStepMeasurementError as exc:
        return VMStepRewardResult(
            reward=None,
            status="gold_vm_measurement_error",
            correct=True,
            correctness_reward=float(base.reward),
            details={"reason": str(exc)},
            **common,
        )
    try:
        predicted_vm = measurement_from_execution(predicted_execution)
    except VMStepMeasurementError as exc:
        return VMStepRewardResult(
            reward=None,
            status="prediction_vm_measurement_error",
            correct=True,
            correctness_reward=float(base.reward),
            reference_vm=reference_vm,
            details={"reason": str(exc)},
            **common,
        )
    raw_score = math.log(
        (reference_vm.estimate + config.vm_epsilon_steps)
        / (predicted_vm.estimate + config.vm_epsilon_steps)
    )
    clipped_score = max(-config.vm_clip, min(config.vm_clip, raw_score))
    bonus = config.vm_weight * clipped_score
    return VMStepRewardResult(
        reward=float(base.reward) + bonus,
        status="correct",
        correct=True,
        correctness_reward=float(base.reward),
        vm_bonus=bonus,
        predicted_vm=predicted_vm,
        reference_vm=reference_vm,
        details={
            "vm_bonus_applied": True,
            "raw_log_ratio": raw_score,
            "clipped_log_ratio": clipped_score,
            "reference_mode": config.reference_mode,
            "cost_scope": config.cost_scope,
        },
        **common,
    )


def _not_run(parse_result: ParseResult) -> ExecutionResult:
    return ExecutionResult(
        status="not_run",
        error_type=parse_result.error_type or "sql_parse_error",
        error_message=parse_result.error_message or "Prediction was not executable",
    )


def score_vm_step_completion(
    *,
    raw_output: str,
    db_path: Path,
    gold_execution: ExecutionResult,
    order_sensitive: bool,
    execution_config: ExecutionConfig,
    reward_config: VMStepRewardConfig = VMStepRewardConfig(),
) -> VMStepRewardResult:
    parse_result = extract_sql(raw_output)
    if parse_result.status == "success" and parse_result.sql is not None:
        predicted = execute_sql(
            db_path,
            parse_result.sql,
            timeout_seconds=execution_config.timeout_seconds,
            max_sql_bytes=execution_config.max_sql_bytes,
            max_result_rows=execution_config.max_result_rows,
            max_result_bytes=execution_config.max_result_bytes,
            worker_memory_limit_bytes=execution_config.worker_memory_limit_bytes,
        )
    else:
        predicted = _not_run(parse_result)
    return score_vm_step_execution_pair(
        parse_result=parse_result,
        predicted_execution=predicted,
        gold_execution=gold_execution,
        order_sensitive=order_sensitive,
        config=reward_config,
    )
