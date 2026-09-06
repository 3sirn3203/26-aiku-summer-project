from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VMStepRewardConfig:
    """Correctness-first reward settings for VM-step optimization."""

    parse_failure_reward: float = -1.0
    unsafe_failure_reward: float = -1.0
    execution_failure_reward: float = -0.7
    result_mismatch_reward: float = -0.2
    correct_base_reward: float = 1.0
    vm_weight: float = 0.1
    vm_clip: float = 1.0
    vm_epsilon_steps: float = 1_000.0
    reference_mode: str = "gold"
    cost_scope: str = "final_only"

    def __post_init__(self) -> None:
        numeric = (
            self.parse_failure_reward,
            self.unsafe_failure_reward,
            self.execution_failure_reward,
            self.result_mismatch_reward,
            self.correct_base_reward,
            self.vm_weight,
            self.vm_clip,
            self.vm_epsilon_steps,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("VM-step reward values must be finite")
        if self.vm_weight < 0 or self.vm_clip < 0:
            raise ValueError("VM-step weight and clip must be non-negative")
        if self.vm_epsilon_steps <= 0:
            raise ValueError("VM-step epsilon must be positive")
        if self.reference_mode != "gold":
            raise ValueError("v1 supports only reference_mode='gold'")
        if self.cost_scope not in {"final_only", "tool_only", "cumulative"}:
            raise ValueError("unsupported VM-step cost scope")
        if self.cost_scope != "final_only":
            raise ValueError("single-turn v1 requires cost_scope='final_only'")
        worst_correct = self.correct_base_reward - self.vm_weight * self.vm_clip
        best_failure = max(
            self.parse_failure_reward,
            self.unsafe_failure_reward,
            self.execution_failure_reward,
            self.result_mismatch_reward,
        )
        if worst_correct <= best_failure:
            raise ValueError(
                "every correct SQL must reward higher than every failure tier"
            )
