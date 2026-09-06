from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from text2sql.config import ExecutionConfig
from text2sql.core.evaluation import RowFingerprintAccumulator
from text2sql.core.models import ExecutionResult
from rl_vm_step.reference import build_gold_reference
from rl_vm_step.trl_reward import TRLVMStepRewardRuntime, make_vm_step_reward_func


def _execution(rows, lower: int) -> ExecutionResult:
    return ExecutionResult(
        status="success",
        rows=rows,
        columns=["value"],
        vm_steps_lower_bound=lower,
        vm_steps_upper_bound_exclusive=lower + 1_000,
        vm_step_progress_interval=1_000,
        vm_step_measurement_complete=True,
    )


class TRLVMStepRewardTests(unittest.TestCase):
    def test_returns_one_float_per_completion(self) -> None:
        execution_config = ExecutionConfig(
            timeout_seconds=1.0,
            max_sql_bytes=10_000,
            max_result_rows=100,
            max_result_bytes=100_000,
            worker_memory_limit_bytes=100_000_000,
        )
        runtime = TRLVMStepRewardRuntime(execution=execution_config)
        reward_func = make_vm_step_reward_func(runtime)
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "db.sqlite"
            db_path.touch()
            record = {
                "example_id": "train:0",
                "db_path": str(db_path),
                "gold_sql": "SELECT 1",
                "order_sensitive": False,
            }
            gold = _execution([], 9_000)
            ordered, unordered = RowFingerprintAccumulator().fingerprints()
            gold.row_count = 0
            gold.row_fingerprint_version = 1
            gold.ordered_rows_fingerprint = ordered
            gold.unordered_rows_fingerprint = unordered
            with patch("rl_vm_step.reference.execute_sql", return_value=gold):
                reference = build_gold_reference(record, execution_config)
            with patch(
                "rl_vm_step.reward.execute_sql",
                side_effect=[_execution([], 1_000), _execution([], 20_000)],
            ):
                rewards = reward_func(
                    ["SELECT 1", "SELECT 1"],
                    db_path=[str(db_path), str(db_path)],
                    gold_sql=["SELECT 1", "SELECT 1"],
                    gold_reference=[reference, reference],
                    order_sensitive=[False, False],
                    example_id=["train:0", "train:0"],
                )
        self.assertEqual(len(rewards), 2)
        self.assertTrue(all(isinstance(value, float) for value in rewards))
        self.assertGreater(rewards[0], rewards[1])


if __name__ == "__main__":
    unittest.main()
