from __future__ import annotations

import unittest

from text2sql.core.models import ExecutionResult, ParseResult
from rl_vm_step.config import VMStepRewardConfig
from rl_vm_step.reward import score_vm_step_execution_pair


def _success(rows, lower: int) -> ExecutionResult:
    return ExecutionResult(
        status="success",
        rows=rows,
        columns=["value"],
        vm_steps_lower_bound=lower,
        vm_steps_upper_bound_exclusive=lower + 1_000,
        vm_step_progress_interval=1_000,
        vm_step_measurement_complete=True,
    )


class VMStepRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parse = ParseResult(status="success", sql="SELECT 1")
        self.gold = _success([[1]], 9_000)

    def test_faster_correct_prediction_gets_higher_reward(self) -> None:
        fast = score_vm_step_execution_pair(
            parse_result=self.parse,
            predicted_execution=_success([[1]], 1_000),
            gold_execution=self.gold,
            order_sensitive=False,
        )
        slow = score_vm_step_execution_pair(
            parse_result=self.parse,
            predicted_execution=_success([[1]], 99_000),
            gold_execution=self.gold,
            order_sensitive=False,
        )
        self.assertGreater(fast.reward, slow.reward)
        self.assertGreater(fast.vm_bonus, 0)
        self.assertLess(slow.vm_bonus, 0)

    def test_fast_mismatch_never_receives_vm_bonus(self) -> None:
        wrong = score_vm_step_execution_pair(
            parse_result=self.parse,
            predicted_execution=_success([[2]], 0),
            gold_execution=self.gold,
            order_sensitive=False,
        )
        slow_correct = score_vm_step_execution_pair(
            parse_result=self.parse,
            predicted_execution=_success([[1]], 1_000_000),
            gold_execution=self.gold,
            order_sensitive=False,
        )
        self.assertEqual(wrong.reward, -0.2)
        self.assertEqual(wrong.vm_bonus, 0.0)
        self.assertGreater(slow_correct.reward, wrong.reward)

    def test_correct_reward_is_bounded(self) -> None:
        config = VMStepRewardConfig(vm_weight=0.1, vm_clip=1.0)
        result = score_vm_step_execution_pair(
            parse_result=self.parse,
            predicted_execution=_success([[1]], 10_000_000),
            gold_execution=self.gold,
            order_sensitive=False,
            config=config,
        )
        self.assertAlmostEqual(result.reward, 0.9)

    def test_missing_prediction_measurement_is_not_silently_scored(self) -> None:
        prediction = _success([[1]], 1_000)
        prediction.vm_step_measurement_complete = None
        result = score_vm_step_execution_pair(
            parse_result=self.parse,
            predicted_execution=prediction,
            gold_execution=self.gold,
            order_sensitive=False,
        )
        self.assertIsNone(result.reward)
        self.assertEqual(result.status, "prediction_vm_measurement_error")

    def test_config_prevents_correctness_inversion(self) -> None:
        with self.assertRaises(ValueError):
            VMStepRewardConfig(
                result_mismatch_reward=0.0,
                vm_weight=2.0,
                vm_clip=1.0,
            )


if __name__ == "__main__":
    unittest.main()
