from __future__ import annotations

from typing import Iterable

from rl_vm_step.models import ExecutionCost, TrajectoryCost


def aggregate_trajectory_cost(
    *,
    final: ExecutionCost,
    tools: Iterable[ExecutionCost] = (),
    scope: str = "final_only",
) -> TrajectoryCost:
    tool_costs = tuple(tools)
    if scope == "final_only":
        selected = (final,)
    elif scope == "tool_only":
        selected = tool_costs
    elif scope == "cumulative":
        selected = (*tool_costs, final)
    else:
        raise ValueError("unsupported VM-step cost scope: %s" % scope)
    if not selected:
        raise ValueError("selected trajectory cost has no executions")
    return TrajectoryCost(
        executions=selected,
        scope=scope,
        total_estimate=sum(item.vm_steps.estimate for item in selected),
    )
