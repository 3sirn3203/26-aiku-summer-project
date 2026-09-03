from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from text2sql.single_turn.distributed import (
    DistributedGenerationError,
    load_generation_shards,
    parse_gpu_ids,
    round_robin_shards,
)
from text2sql.single_turn.evaluation_runner import _read_evaluation_records, run_full_evaluation
from text2sql.core.models import ExecutionResult

from text2sql.config import load_config


CODE_ROOT = Path(__file__).resolve().parents[1]


class DistributedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(
            CODE_ROOT / "configs" / "single_turn_zero_shot.json"
        )
        if not cls.config.spider.root.is_dir():
            raise unittest.SkipTest("Spider data is not installed")

    def test_gpu_list_and_round_robin_contract(self) -> None:
        self.assertEqual(parse_gpu_ids("1, 3,7"), (1, 3, 7))
        with self.assertRaises(ValueError):
            parse_gpu_ids("1,1")
        with self.assertRaises(ValueError):
            parse_gpu_ids("cuda:1")
        self.assertEqual(
            round_robin_shards(list(range(8)), 3),
            [[0, 3, 6], [1, 4, 7], [2, 5]],
        )

    def test_generation_loader_accepts_only_explicit_newer_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "attempt-000"
            second = root / "attempt-001"
            first.mkdir()
            second.mkdir()
            error_record = {
                "example_id": "dev:0",
                "generation": {"status": "error", "error_type": "generation_error"},
            }
            success_record = {
                "example_id": "dev:0",
                "generation": {"status": "success"},
            }
            (first / "worker-00.jsonl").write_text(
                json.dumps(error_record) + "\n", encoding="utf-8"
            )
            (second / "worker-00.jsonl").write_text(
                json.dumps(success_record) + "\n", encoding="utf-8"
            )
            records, warnings = load_generation_shards(root)
            self.assertEqual(records["dev:0"]["generation"]["status"], "success")
            self.assertTrue(any("selected retry" in warning for warning in warnings))

            (second / "worker-00.jsonl").unlink()
            (first / "worker-01.jsonl").write_text(
                json.dumps(success_record) + "\n", encoding="utf-8"
            )
            with self.assertRaises(DistributedGenerationError):
                load_generation_shards(root)

    def test_resume_repairs_only_an_incomplete_evaluation_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text('{"example_id":"dev:0"}\n{"example_id":', encoding="utf-8")
            records = _read_evaluation_records(path)
            self.assertEqual(records, [{"example_id": "dev:0"}])
            repaired_lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(repaired_lines), 1)
            self.assertEqual(json.loads(repaired_lines[0]), records[0])

    def test_distributed_smoke_uses_fixed_manifest_order(self) -> None:
        class RecordingProgress:
            def __init__(self):
                self.stage = None
                self.totals = {}
                self.updates = {}
                self.messages = []

            def start(self, stage, _label, total, initial=0):
                self.stage = stage
                self.totals[stage] = total
                self.updates.setdefault(stage, []).append(initial)

            def update(self, completed):
                self.updates[self.stage].append(completed)

            def finish(self):
                self.stage = None

            def message(self, message):
                self.messages.append(message)

        def fake_execute_sql(_db_path, _sql, **_kwargs):
            return ExecutionResult(
                status="success",
                rows=[[1]],
                columns=["value"],
                query_elapsed_ns=10,
                worker_elapsed_ns=20,
                parent_elapsed_ns=30,
            )

        with tempfile.TemporaryDirectory() as directory:
            progress = RecordingProgress()
            config = replace(
                self.config,
                official_evaluation=replace(
                    self.config.official_evaluation, enabled=False
                ),
                output=replace(self.config.output, directory=Path(directory)),
            )
            with mock.patch(
                "text2sql.single_turn.evaluation_runner.execute_sql", side_effect=fake_execute_sql
            ):
                result = run_full_evaluation(
                    config,
                    backend_name="mock",
                    gpu_ids=(0, 1),
                    selection="smoke",
                    run_name="distributed-smoke",
                    progress=progress,
                )
            self.assertTrue(result["summary"]["pipeline_pass"])
            self.assertEqual(result["summary"]["run_type"], "distributed_smoke")
            self.assertEqual(result["summary"]["selection"], "smoke")
            self.assertEqual(result["summary"]["total_examples"], 8)
            self.assertEqual(
                progress.totals,
                {"generation": 8, "sql_execution": 8},
            )
            self.assertEqual(progress.updates["generation"][-1], 8)
            self.assertEqual(progress.updates["sql_execution"][-1], 8)
            self.assertTrue(any("starting worker" in message for message in progress.messages))
            records = [
                json.loads(line)
                for line in (
                    Path(result["run_directory"]) / "generation_records.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [record["index"] for record in records],
                [sample.index for sample in self.config.smoke.samples],
            )

    def test_full_dev_mock_is_sharded_merged_and_resumable(self) -> None:
        executed_sql = []

        def fake_execute_sql(_db_path, sql, **_kwargs):
            executed_sql.append(sql)
            return ExecutionResult(
                status="success",
                rows=[[1]],
                columns=["value"],
                elapsed_seconds=0.0012,
                query_elapsed_ns=1_000_000,
                worker_elapsed_ns=1_100_000,
                parent_elapsed_ns=1_200_000,
            )

        official_preflight = {
            "ok": True,
            "errors": [],
            "evaluator_tree_sha256": "evaluator-hash",
            "databases": {},
        }

        def fake_evaluate_official(items, **_kwargs):
            return {
                "ok": True,
                "processed_examples": len(items),
                "policy": {"fixture": "all-match"},
                "evaluator": {"fixture": "pinned"},
                "results": [
                    {
                        "example_id": item.example_id,
                        "db_id": item.db_id,
                        "exact_set_match": {"status": "scored", "match": True},
                        "test_suite": {"status": "scored", "match": True},
                    }
                    for item in items
                ],
            }

        with tempfile.TemporaryDirectory() as directory:
            progress = mock.Mock()
            config = replace(
                self.config,
                output=replace(self.config.output, directory=Path(directory)),
            )
            with mock.patch(
                "text2sql.single_turn.evaluation_runner.execute_sql", side_effect=fake_execute_sql
            ), mock.patch(
                "text2sql.single_turn.evaluation_runner.validate_official_environment",
                return_value=official_preflight,
            ), mock.patch(
                "text2sql.single_turn.evaluation_runner.evaluate_official",
                side_effect=fake_evaluate_official,
            ):
                result = run_full_evaluation(
                    config,
                    backend_name="mock",
                    gpu_ids=(0, 1),
                    run_name="full-dev-mock",
                    progress=progress,
                )

            self.assertTrue(result["summary"]["pipeline_pass"])
            self.assertEqual(result["summary"]["total_examples"], 1034)
            self.assertEqual(result["summary"]["processed_examples"], 1034)
            self.assertEqual(result["summary"]["generation_worker_count"], 2)
            self.assertEqual(result["summary"]["generation_statuses"], {"success": 1034})
            self.assertEqual(result["summary"]["prediction_query_timing"]["count"], 1034)
            self.assertEqual(result["summary"]["exact_set_match_accuracy"], 1.0)
            self.assertEqual(result["summary"]["test_suite_accuracy"], 1.0)
            progress.start.assert_any_call(
                "generation", "LLM generation", 1034, initial=0
            )
            progress.start.assert_any_call(
                "sql_execution", "Sequential SQL execution", 1034, initial=0
            )
            progress.start.assert_any_call(
                "official_evaluation",
                "Official Spider evaluation",
                1034,
                initial=0,
            )
            self.assertEqual(len(executed_sql), 2068)
            for position in range(0, len(executed_sql), 2):
                self.assertEqual(
                    executed_sql[position].rstrip().rstrip(";"),
                    executed_sql[position + 1].rstrip().rstrip(";"),
                )

            run_dir = Path(result["run_directory"])
            generation_records = [
                json.loads(line)
                for line in (run_dir / "generation_records.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            records = [
                json.loads(line)
                for line in (run_dir / "records.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([record["index"] for record in generation_records], list(range(1034)))
            self.assertEqual([record["index"] for record in records], list(range(1034)))
            self.assertTrue(
                all("gold_sql" not in record for record in generation_records)
            )
            for record in generation_records:
                self.assertEqual(record["schema_version"], 5)
                self.assertNotIn("messages", record)
                self.assertNotIn("generation_worker", record)
                self.assertNotIn("schema_sha256", record)
                self.assertEqual(
                    set(record["generation"]),
                    {
                        "status",
                        "elapsed_seconds",
                        "raw_output",
                        "error_type",
                        "error_message",
                    },
                )
            for record in records:
                self.assertNotIn("messages", record)
                self.assertNotIn("generation_worker", record)
                self.assertNotIn("schema_sha256", record)
                self.assertFalse(
                    {
                        "input_tokens",
                        "output_tokens",
                        "model_id",
                        "requested_revision",
                        "resolved_revision",
                        "physical_gpu",
                        "logical_device",
                        "worker_id",
                        "attempt",
                    }
                    & set(record["generation"])
                )
            manifest = json.loads(
                (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], 6)
            persisted_summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                persisted_summary,
                {
                    "test_suite_accuracy": 1.0,
                    "exact_set_match_accuracy": 1.0,
                    "result_match_accuracy": 1.0,
                    "mean_vm_steps": None,
                    "mean_latency_ms": 1.0,
                },
            )
            self.assertEqual(
                {
                    status["physical_gpu"]
                    for status in manifest["generation"]["worker_statuses"]
                },
                {0, 1},
            )
            shard_paths = sorted((run_dir / "shards").glob("attempt-*/*.jsonl"))
            self.assertEqual(len(shard_paths), 2)
            shard_mtimes = {path: path.stat().st_mtime_ns for path in shard_paths}

            executed_sql.clear()
            with mock.patch(
                "text2sql.single_turn.evaluation_runner.execute_sql", side_effect=fake_execute_sql
            ), mock.patch(
                "text2sql.single_turn.evaluation_runner.launch_generation_attempt"
            ) as launch_mock, mock.patch(
                "text2sql.single_turn.evaluation_runner.validate_official_environment",
                return_value=official_preflight,
            ), mock.patch(
                "text2sql.single_turn.evaluation_runner.evaluate_official",
                side_effect=fake_evaluate_official,
            ) as official_mock:
                resumed = run_full_evaluation(
                    config,
                    backend_name="mock",
                    gpu_ids=(1, 0),
                    resume_run="full-dev-mock",
                )
            self.assertTrue(resumed["summary"]["pipeline_pass"])
            launch_mock.assert_not_called()
            official_mock.assert_not_called()
            self.assertEqual(executed_sql, [])
            self.assertEqual(
                {path: path.stat().st_mtime_ns for path in shard_paths},
                shard_mtimes,
            )


if __name__ == "__main__":
    unittest.main()
