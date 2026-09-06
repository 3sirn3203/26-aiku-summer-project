from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from text2sql.config import ExecutionConfig
from rl_vm_step.reference import build_gold_reference, gold_execution_from_reference
from rl_vm_step.reward import score_vm_step_completion


class ExecutorIntegrationTests(unittest.TestCase):
    def test_semantically_equal_query_is_ranked_by_vm_steps(self) -> None:
        config = ExecutionConfig(
            timeout_seconds=5.0,
            max_sql_bytes=10_000,
            max_result_rows=100,
            max_result_bytes=100_000,
            worker_memory_limit_bytes=1_073_741_824,
        )
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "database.sqlite"
            connection = sqlite3.connect(db_path)
            connection.execute("CREATE TABLE items(value INTEGER)")
            connection.executemany(
                "INSERT INTO items VALUES (?)", ((value,) for value in range(5_000))
            )
            connection.commit()
            connection.close()
            record = {
                "example_id": "integration:0",
                "db_path": str(db_path),
                "gold_sql": "SELECT count(*) FROM items",
                "order_sensitive": False,
            }
            reference = build_gold_reference(record, config)
            result = score_vm_step_completion(
                raw_output="SELECT sum(1) FROM items",
                db_path=db_path,
                gold_execution=gold_execution_from_reference(reference),
                order_sensitive=False,
                execution_config=config,
            )
        self.assertEqual(reference["status"], "ready")
        self.assertTrue(result.correct)
        self.assertLess(result.vm_bonus, 0.0)


if __name__ == "__main__":
    unittest.main()
