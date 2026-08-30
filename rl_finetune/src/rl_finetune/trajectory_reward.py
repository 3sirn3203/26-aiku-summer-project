from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from text2sql.config import ExecutionConfig
from text2sql.core.executor import execute_sql
from text2sql.core.models import ExecutionResult
from rl_finetune.agentic_runtime.environment import WorkflowInfrastructureError
from rl_finetune.agentic_runtime.models import Trajectory
from rl_finetune.rewards import RewardConfig, score_completion


def _execution_kwargs(config: ExecutionConfig) -> Dict[str, Any]:
    return {
        "timeout_seconds": config.timeout_seconds,
        "max_sql_bytes": config.max_sql_bytes,
        "max_result_rows": config.max_result_rows,
        "max_result_bytes": config.max_result_bytes,
        "worker_memory_limit_bytes": config.worker_memory_limit_bytes,
    }


class GoldExecutionCache:
    def __init__(self, execution_config: ExecutionConfig) -> None:
        self.execution_config = execution_config
        self._cache: Dict[Tuple[str, str], ExecutionResult] = {}

    def get(self, db_path: Path, gold_sql: str) -> ExecutionResult:
        key = (str(Path(db_path).expanduser().resolve()), gold_sql)
        if key not in self._cache:
            self._cache[key] = execute_sql(
                Path(db_path), gold_sql, **_execution_kwargs(self.execution_config)
            )
        return self._cache[key]


class TrajectoryRewardRuntime:
    def __init__(
        self,
        execution_config: ExecutionConfig,
        *,
        reward_config: Optional[RewardConfig] = None,
        gold_cache: Optional[GoldExecutionCache] = None,
    ) -> None:
        self.execution_config = execution_config
        self.reward_config = reward_config or RewardConfig(latency_weight=0.0)
        if self.reward_config.latency_weight != 0.0:
            raise ValueError("agentic v1 requires latency_weight=0.0")
        self.gold_cache = gold_cache or GoldExecutionCache(execution_config)

    def score_group(
        self,
        trajectories: Sequence[Trajectory],
        record: Mapping[str, Any],
    ) -> bool:
        db_path = Path(str(record["db_path"]))
        gold_sql = str(record["gold_sql"])
        order_sensitive = bool(record["order_sensitive"])
        gold_execution = self.gold_cache.get(db_path, gold_sql)
        if not gold_execution.succeeded:
            for trajectory in trajectories:
                trajectory.reward = None
                trajectory.terminal_status = "gold_execution_error"
                trajectory.reward_result = {
                    "status": "gold_execution_error",
                    "gold_execution": gold_execution.to_dict(),
                }
            return False

        for trajectory in trajectories:
            result = score_completion(
                raw_output=trajectory.final_raw_output,
                db_path=db_path,
                gold_sql=gold_sql,
                order_sensitive=order_sensitive,
                execution_config=self.execution_config,
                reward_config=self.reward_config,
                gold_execution=gold_execution,
            )
            predicted_error = result.predicted_execution.get("error_type")
            if predicted_error in {
                "database_not_found",
                "executor_internal_error",
                "executor_worker_crash",
            }:
                raise WorkflowInfrastructureError(
                    "final executor infrastructure failure: %s" % predicted_error
                )
            trajectory.reward = result.reward
            trajectory.terminal_status = result.status
            trajectory.reward_result = result.to_dict()
        return True

