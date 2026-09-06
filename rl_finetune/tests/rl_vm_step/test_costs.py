from __future__ import annotations

import unittest

from rl_vm_step.costs import aggregate_trajectory_cost
from rl_vm_step.models import ExecutionCost, VMStepMeasurement


def _cost(role: str, estimate: float) -> ExecutionCost:
    return ExecutionCost(
        role=role,
        vm_steps=VMStepMeasurement(
            lower_bound=int(estimate - 500),
            upper_bound_exclusive=int(estimate + 500),
            interval=1_000,
            estimate=estimate,
        ),
    )


class TrajectoryCostTests(unittest.TestCase):
    def test_single_turn_uses_final_only(self) -> None:
        result = aggregate_trajectory_cost(
            final=_cost("final", 2_500),
            tools=[_cost("tool", 10_500)],
            scope="final_only",
        )
        self.assertEqual(result.total_estimate, 2_500)

    def test_cumulative_scope_is_ready_for_multi_turn(self) -> None:
        result = aggregate_trajectory_cost(
            final=_cost("final", 2_500),
            tools=[_cost("tool", 1_500), _cost("tool", 3_500)],
            scope="cumulative",
        )
        self.assertEqual(result.total_estimate, 7_500)


if __name__ == "__main__":
    unittest.main()
