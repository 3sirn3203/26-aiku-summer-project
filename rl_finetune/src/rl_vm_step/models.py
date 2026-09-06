from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence


@dataclass(frozen=True)
class VMStepMeasurement:
    lower_bound: int
    upper_bound_exclusive: int
    interval: int
    estimate: float
    complete: bool = True
    method: str = "sqlite_progress_handler_interval_midpoint"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionCost:
    role: str
    vm_steps: VMStepMeasurement

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "vm_steps": self.vm_steps.to_dict()}


@dataclass(frozen=True)
class TrajectoryCost:
    executions: Sequence[ExecutionCost]
    scope: str
    total_estimate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "executions": [item.to_dict() for item in self.executions],
            "scope": self.scope,
            "total_estimate": self.total_estimate,
        }


@dataclass
class VMStepRewardResult:
    reward: Optional[float]
    status: str
    correct: bool
    correctness_reward: Optional[float]
    vm_bonus: float = 0.0
    predicted_vm: Optional[VMStepMeasurement] = None
    reference_vm: Optional[VMStepMeasurement] = None
    parsed_sql: Optional[str] = None
    parse_result: Mapping[str, Any] = field(default_factory=dict)
    predicted_execution: Mapping[str, Any] = field(default_factory=dict)
    gold_execution: Mapping[str, Any] = field(default_factory=dict)
    comparison: Mapping[str, Any] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload
