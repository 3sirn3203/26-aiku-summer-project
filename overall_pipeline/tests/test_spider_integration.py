from __future__ import annotations

import builtins
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from text2sql.config import load_config
from text2sql.core.models import ExecutionResult, GenerationResult
from text2sql.core.executor import execute_sql
from text2sql.single_turn.smoke_runner import run_smoke
from text2sql.core.spider import SpiderDataset


CODE_ROOT = Path(__file__).resolve().parents[1]


class SpiderIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(
            CODE_ROOT / "configs" / "single_turn_zero_shot.json"
        )
        if not cls.config.spider.root.is_dir():
            raise unittest.SkipTest("Spider data is not installed")

    def test_dev_data_contract(self) -> None:
        report = SpiderDataset(self.config.spider).validate()
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.example_count, 1034)
        self.assertEqual(report.database_count, 20)

    def test_generation_view_does_not_expose_gold_metadata(self) -> None:
        dataset = SpiderDataset(self.config.spider, include_gold=False)
        example = dataset.get_example(0)
        self.assertEqual(example.gold_sql, "")
        self.assertEqual(example.parsed_sql, {})
        self.assertTrue(example.question)
        self.assertTrue(dataset.get_schema(example.db_id).table_names)

    def test_large_gold_result_streams_completely(self) -> None:
        dataset = SpiderDataset(self.config.spider)
        example = dataset.get_example(455)
        result = execute_sql(
            dataset.database_path(example.db_id),
            example.gold_sql,
            timeout_seconds=self.config.execution.timeout_seconds,
            max_sql_bytes=self.config.execution.max_sql_bytes,
            max_result_rows=self.config.execution.max_result_rows,
            max_result_bytes=self.config.execution.max_result_bytes,
            worker_memory_limit_bytes=(
                self.config.execution.worker_memory_limit_bytes
            ),
        )

        self.assertEqual(example.db_id, "wta_1")
        self.assertEqual(result.status, "success", result.to_dict())
        self.assertEqual(result.row_count, 20662)
        self.assertEqual(len(result.rows), 10000)
        self.assertTrue(result.truncated)
        self.assertTrue(result.vm_step_measurement_complete)

    def test_mock_smoke_never_imports_model_libraries(self) -> None:
        blocked = {"torch", "transformers", "huggingface_hub"}
        imported = []
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            root = name.split(".", 1)[0]
            if root in blocked:
                imported.append(name)
                raise AssertionError("local mock imported %s" % name)
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                self.config,
                output=replace(self.config.output, directory=Path(directory)),
            )
            builtins.__import__ = guarded_import
            try:
                result = run_smoke(config, backend_name="mock", run_name="unittest")
            finally:
                builtins.__import__ = original_import

            self.assertEqual(imported, [])
            self.assertTrue(result["summary"]["pipeline_pass"])
            self.assertFalse(result["summary"]["accuracy_measurement"])
            self.assertEqual(result["summary"]["executable_predictions"], 8)
            self.assertEqual(result["summary"]["gold_execution_statuses"], {"success": 8})
            self.assertTrue(result["summary"]["official_evaluation_ok"])
            self.assertTrue(result["summary"]["official_contract_met"])
            self.assertEqual(result["summary"]["exact_set_match_accuracy"], 1.0)
            self.assertEqual(result["summary"]["test_suite_accuracy"], 1.0)
            self.assertTrue(result["summary"]["selected_databases_unchanged"])
            run_dir = Path(result["run_directory"])
            self.assertTrue((run_dir / "run_manifest.json").is_file())
            self.assertTrue((run_dir / "records.jsonl").is_file())
            self.assertTrue((run_dir / "summary.json").is_file())
            persisted_summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(persisted_summary),
                {
                    "test_suite_accuracy",
                    "exact_set_match_accuracy",
                    "result_match_accuracy",
                    "mean_vm_steps",
                    "mean_latency_ms",
                },
            )
            self.assertEqual(persisted_summary["test_suite_accuracy"], 1.0)
            self.assertEqual(persisted_summary["exact_set_match_accuracy"], 1.0)
            self.assertEqual(persisted_summary["result_match_accuracy"], 1.0)
            manifest = json.loads(
                (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["config"]["output"]["directory"], directory)
            self.assertNotIn("source_config", manifest)
            self.assertEqual(len(manifest["source_config_sha256"]), 64)
            self.assertEqual(len(manifest["source"]["python_tree_sha256"]), 64)
            self.assertEqual(
                set(manifest["source"]),
                {"package_version", "python_tree_sha256", "scope"},
            )
            self.assertTrue(manifest["config"]["official_evaluation"]["enabled"])
            self.assertTrue(manifest["official_evaluation"]["preflight"]["ok"])
            self.assertTrue(manifest["official_evaluation"]["result"]["ok"])
            records = [
                json.loads(line)
                for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 8)
            for record in records:
                rendered_prompt = json.dumps(record["messages"], ensure_ascii=False)
                self.assertNotIn(record["gold_sql"], rendered_prompt)
                self.assertEqual(
                    record["official_evaluation"]["exact_set_match"]["status"],
                    "scored",
                )
                self.assertTrue(
                    record["official_evaluation"]["exact_set_match"]["match"]
                )
                self.assertEqual(
                    record["official_evaluation"]["test_suite"]["status"],
                    "scored",
                )
                self.assertTrue(
                    record["official_evaluation"]["test_suite"]["match"]
                )
                self.assertIsInstance(
                    record["predicted_execution"]["query_elapsed_ns"], int
                )

    def test_backend_initialization_failure_writes_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                self.config,
                output=replace(self.config.output, directory=Path(directory)),
            )
            with mock.patch(
                "text2sql.single_turn.smoke_runner._create_backend",
                side_effect=OSError("offline cache miss"),
            ):
                with self.assertRaisesRegex(RuntimeError, "diagnostics written"):
                    run_smoke(config, backend_name="hf", run_name="failed-init")

            run_dir = Path(directory) / "failed-init"
            manifest = json.loads(
                (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failure"]["stage"], "backend_initialization")
            self.assertEqual(
                summary,
                {
                    "test_suite_accuracy": None,
                    "exact_set_match_accuracy": None,
                    "result_match_accuracy": None,
                    "mean_vm_steps": None,
                    "mean_latency_ms": None,
                },
            )
            self.assertEqual((run_dir / "records.jsonl").read_text(), "")

    def test_mock_requires_every_prediction_to_execute_and_match(self) -> None:
        dataset = SpiderDataset(self.config.spider)
        responses = {
            "%s:%d" % (example.split, example.index): example.gold_sql
            for example in (
                dataset.get_example(sample.index) for sample in self.config.smoke.samples
            )
        }
        first_id = "dev:%d" % self.config.smoke.samples[0].index
        responses[first_id] = "SELECT missing_column FROM missing_table"

        class PartiallyFailingBackend:
            name = "mock"

            def generate(self, request):
                return GenerationResult(
                    status="success",
                    raw_output=responses[request.example_id],
                    model_id="test-backend",
                )

            def metadata(self):
                return {"backend": "mock", "fixture": "one-invalid-prediction"}

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                self.config,
                output=replace(self.config.output, directory=Path(directory)),
            )
            with mock.patch(
                "text2sql.single_turn.smoke_runner._create_backend",
                return_value=PartiallyFailingBackend(),
            ):
                result = run_smoke(
                    config, backend_name="mock", run_name="partial-prediction-failure"
                )

            self.assertFalse(result["summary"]["pipeline_pass"])
            self.assertFalse(result["summary"]["prediction_contract_met"])
            self.assertEqual(result["summary"]["executable_predictions"], 7)

    def test_prediction_runs_first_and_timing_is_recorded_and_summarized(self) -> None:
        class ProbeBackend:
            name = "mock"

            def generate(self, _request):
                return GenerationResult(
                    status="success",
                    raw_output="SELECT 1 AS predicted_probe",
                    model_id="test-backend",
                )

            def metadata(self):
                return {"backend": "mock", "fixture": "timing-and-order-probe"}

            def close(self):
                return None

        executed_sql = []

        def fake_execute_sql(_db_path, sql, **_kwargs):
            executed_sql.append(sql)
            call_number = len(executed_sql)
            query_elapsed_ns = call_number * 1_000_000
            worker_elapsed_ns = query_elapsed_ns + 100_000
            parent_elapsed_ns = worker_elapsed_ns + 100_000
            return ExecutionResult(
                status="success",
                rows=[[1]],
                columns=["value"],
                elapsed_seconds=parent_elapsed_ns / 1_000_000_000,
                query_elapsed_ns=query_elapsed_ns,
                worker_elapsed_ns=worker_elapsed_ns,
                parent_elapsed_ns=parent_elapsed_ns,
            )

        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                self.config,
                output=replace(self.config.output, directory=Path(directory)),
            )
            with mock.patch(
                "text2sql.single_turn.smoke_runner._create_backend", return_value=ProbeBackend()
            ), mock.patch(
                "text2sql.single_turn.smoke_runner.execute_sql", side_effect=fake_execute_sql
            ) as execute_mock:
                result = run_smoke(
                    config, backend_name="mock", run_name="timing-and-order"
                )

            self.assertEqual(execute_mock.call_count, 16)
            for offset, sample in enumerate(self.config.smoke.samples):
                prediction_call = executed_sql[offset * 2]
                gold_call = executed_sql[offset * 2 + 1]
                self.assertEqual(prediction_call, "SELECT 1 AS predicted_probe")
                expected_gold = SpiderDataset(self.config.spider).get_example(
                    sample.index
                ).gold_sql
                self.assertEqual(gold_call, expected_gold)

            records = [
                json.loads(line)
                for line in (
                    Path(result["run_directory"]) / "records.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            for offset, record in enumerate(records):
                predicted = record["predicted_execution"]
                gold = record["gold_execution"]
                predicted_query_ns = (offset * 2 + 1) * 1_000_000
                gold_query_ns = (offset * 2 + 2) * 1_000_000
                self.assertEqual(predicted["query_elapsed_ns"], predicted_query_ns)
                self.assertEqual(predicted["query_elapsed_ms"], predicted_query_ns / 1e6)
                self.assertEqual(
                    predicted["worker_elapsed_ms"],
                    predicted["worker_elapsed_ns"] / 1e6,
                )
                self.assertEqual(
                    predicted["parent_elapsed_ms"],
                    predicted["parent_elapsed_ns"] / 1e6,
                )
                self.assertEqual(gold["query_elapsed_ns"], gold_query_ns)
                self.assertEqual(gold["query_elapsed_ms"], gold_query_ns / 1e6)

            prediction_timing = result["summary"]["prediction_query_timing"]
            gold_timing = result["summary"]["gold_query_timing"]
            self.assertEqual(prediction_timing["total_executions"], 8)
            self.assertEqual(prediction_timing["successful_executions"], 8)
            self.assertEqual(prediction_timing["excluded_non_success"], 0)
            self.assertEqual(
                prediction_timing["excluded_success_without_query_timing"], 0
            )
            self.assertEqual(prediction_timing["count"], 8)
            self.assertEqual(
                prediction_timing["query_elapsed_ns"],
                {
                    "min": 1_000_000,
                    "max": 15_000_000,
                    "mean": 8_000_000.0,
                    "median": 8_000_000.0,
                },
            )
            self.assertEqual(
                prediction_timing["query_elapsed_ms"],
                {"min": 1.0, "max": 15.0, "mean": 8.0, "median": 8.0},
            )
            self.assertEqual(gold_timing["count"], 8)
            self.assertEqual(
                gold_timing["query_elapsed_ns"],
                {
                    "min": 2_000_000,
                    "max": 16_000_000,
                    "mean": 9_000_000.0,
                    "median": 9_000_000.0,
                },
            )
            self.assertEqual(
                gold_timing["query_elapsed_ms"],
                {"min": 2.0, "max": 16.0, "mean": 9.0, "median": 9.0},
            )

    def test_official_test_suite_timeout_is_a_scored_failure(self) -> None:
        def timed_out_test_suite(items, **_kwargs):
            return {
                "schema_version": 1,
                "ok": True,
                "total_examples": len(items),
                "processed_examples": len(items),
                "metrics": {
                    "exact_set_match": {
                        "valid": True,
                        "classified_examples": len(items),
                        "infrastructure_failures": 0,
                    },
                    "test_suite": {
                        "valid": True,
                        "classified_examples": len(items),
                        "infrastructure_failures": 0,
                    },
                },
                "policy": {},
                "evaluator": {},
                "results": [
                    {
                        "example_id": item.example_id,
                        "db_id": item.db_id,
                        "exact_set_match": {
                            "status": "scored",
                            "match": True,
                        },
                        "test_suite": {
                            "status": "evaluator_timeout",
                            "match": False,
                        },
                    }
                    for item in items
                ],
            }

        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                self.config,
                output=replace(self.config.output, directory=Path(directory)),
            )
            with mock.patch(
                "text2sql.single_turn.smoke_runner.evaluate_official",
                side_effect=timed_out_test_suite,
            ):
                result = run_smoke(
                    config, backend_name="mock", run_name="official-timeout"
                )

        summary = result["summary"]
        self.assertFalse(summary["pipeline_pass"])
        self.assertTrue(summary["exact_set_match_valid"])
        self.assertEqual(summary["exact_set_match_accuracy"], 1.0)
        self.assertTrue(summary["test_suite_accuracy_valid"])
        self.assertEqual(summary["test_suite_matches"], 0)
        self.assertEqual(summary["test_suite_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
