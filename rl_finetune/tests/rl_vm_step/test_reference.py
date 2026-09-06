from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from text2sql.config import ExecutionConfig
from text2sql.core.models import ExecutionResult
from rl_vm_step.reference import (
    build_gold_reference,
    gold_execution_from_reference,
    validate_gold_reference,
)


def _config() -> ExecutionConfig:
    return ExecutionConfig(
        timeout_seconds=1.0,
        max_sql_bytes=10_000,
        max_result_rows=100,
        max_result_bytes=100_000,
        worker_memory_limit_bytes=100_000_000,
    )


def _gold_execution() -> ExecutionResult:
    return ExecutionResult(
        status="success",
        columns=["value"],
        row_count=1,
        row_fingerprint_version=1,
        ordered_rows_fingerprint="ordered",
        unordered_rows_fingerprint="unordered",
        vm_steps_lower_bound=1_000,
        vm_steps_upper_bound_exclusive=2_000,
        vm_step_progress_interval=1_000,
        vm_step_measurement_complete=True,
    )


class GoldReferenceTests(unittest.TestCase):
    def test_reference_round_trip_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "database.sqlite"
            db_path.touch()
            record = {
                "example_id": "train:0",
                "db_path": str(db_path),
                "gold_sql": "SELECT 1",
                "order_sensitive": False,
            }
            with patch(
                "rl_vm_step.reference.execute_sql", return_value=_gold_execution()
            ):
                reference = build_gold_reference(record, _config())
            validate_gold_reference(reference, record, _config())
            restored = gold_execution_from_reference(reference)

        self.assertEqual(reference["status"], "ready")
        self.assertEqual(reference["vm_step"]["estimate"], 1_500.0)
        self.assertEqual(restored.unordered_rows_fingerprint, "unordered")

    def test_reference_detects_database_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "database.sqlite"
            db_path.touch()
            record = {
                "example_id": "train:0",
                "db_path": str(db_path),
                "gold_sql": "SELECT 1",
                "order_sensitive": False,
            }
            with patch(
                "rl_vm_step.reference.execute_sql", return_value=_gold_execution()
            ):
                reference = build_gold_reference(record, _config())
            db_path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                validate_gold_reference(reference, record, _config())


if __name__ == "__main__":
    unittest.main()
