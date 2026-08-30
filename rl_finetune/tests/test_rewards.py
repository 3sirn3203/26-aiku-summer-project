from __future__ import annotations

import unittest

from text2sql.core.models import ExecutionResult, ParseResult
from rl_finetune.rewards import RewardConfig, score_execution_pair


def _success(rows, query_elapsed_ns: int) -> ExecutionResult:
    return ExecutionResult(
        status="success",
        rows=rows,
        columns=["value"],
        query_elapsed_ns=query_elapsed_ns,
    )


class RewardTests(unittest.TestCase):
    def test_correct_sql_always_beats_fast_mismatch(self) -> None:
        parse = ParseResult(status="success", sql="SELECT 1")
        gold = _success([[1]], query_elapsed_ns=1_000)
        slow_correct = _success([[1]], query_elapsed_ns=10_000_000)
        fast_wrong = _success([[2]], query_elapsed_ns=1)

        correct_reward = score_execution_pair(
            parse_result=parse,
            predicted_execution=slow_correct,
            gold_execution=gold,
            order_sensitive=False,
        )
        mismatch_reward = score_execution_pair(
            parse_result=parse,
            predicted_execution=fast_wrong,
            gold_execution=gold,
            order_sensitive=False,
        )

        self.assertEqual(correct_reward.status, "correct")
        self.assertEqual(mismatch_reward.status, "result_mismatch")
        self.assertGreater(correct_reward.reward, mismatch_reward.reward)

    def test_latency_only_orders_correct_predictions(self) -> None:
        parse = ParseResult(status="success", sql="SELECT 1")
        gold = _success([[1]], query_elapsed_ns=100_000)
        fast_correct = _success([[1]], query_elapsed_ns=10_000)
        slow_correct = _success([[1]], query_elapsed_ns=1_000_000)

        fast_reward = score_execution_pair(
            parse_result=parse,
            predicted_execution=fast_correct,
            gold_execution=gold,
            order_sensitive=False,
        )
        slow_reward = score_execution_pair(
            parse_result=parse,
            predicted_execution=slow_correct,
            gold_execution=gold,
            order_sensitive=False,
        )

        self.assertEqual(fast_reward.status, "correct")
        self.assertEqual(slow_reward.status, "correct")
        self.assertGreater(fast_reward.reward, slow_reward.reward)

    def test_gold_execution_error_is_skippable(self) -> None:
        result = score_execution_pair(
            parse_result=ParseResult(status="success", sql="SELECT 1"),
            predicted_execution=_success([[1]], query_elapsed_ns=100),
            gold_execution=ExecutionResult(
                status="execution_error",
                error_type="fixture_gold_error",
                error_message="gold failed",
            ),
            order_sensitive=False,
        )

        self.assertIsNone(result.reward)
        self.assertEqual(result.status, "gold_execution_error")

    def test_config_rejects_latency_weight_that_can_invert_correctness(self) -> None:
        with self.assertRaises(ValueError):
            RewardConfig(
                result_mismatch_reward=0.0,
                correct_base_reward=1.0,
                latency_weight=2.0,
                latency_clip=1.0,
            )


if __name__ == "__main__":
    unittest.main()


