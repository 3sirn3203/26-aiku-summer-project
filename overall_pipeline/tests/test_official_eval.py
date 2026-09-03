from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from text2sql.config import load_config
from text2sql.core.executor import execute_sql
from text2sql.core.official_eval import (
    OfficialEvaluationError,
    OfficialEvaluationItem,
    _run_worker,
    _worker_environment,
    evaluate_official,
    validate_official_environment,
)
from text2sql.core.spider import SpiderDataset
from text2sql.core.official_eval_worker import _apply_worker_memory_limit


CODE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DEPENDENCIES = {
    "nltk": "3.9.1",
    "sqlparse": "0.5.3",
    "tqdm": "4.67.1",
}


def _real_official_assets_available() -> bool:
    try:
        for name, version in EXPECTED_DEPENDENCIES.items():
            if importlib.metadata.version(name) != version:
                return False
    except importlib.metadata.PackageNotFoundError:
        return False
    config = load_config(CODE_ROOT / "configs" / "single_turn_zero_shot.json")
    return (
        config.official_evaluation.evaluator_root.is_dir()
        and config.official_evaluation.test_suite_database_root.is_dir()
        and config.official_evaluation.nltk_data_dir is not None
        and config.official_evaluation.nltk_data_dir.is_dir()
    )


class OfficialEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(
            CODE_ROOT / "configs" / "single_turn_zero_shot.json"
        )
        cls.dataset = SpiderDataset(cls.config.spider)

    def _preflight(self, db_ids):
        return validate_official_environment(
            evaluator_root=self.config.official_evaluation.evaluator_root,
            database_root=self.config.official_evaluation.test_suite_database_root,
            tables_path=self.dataset.tables_path,
            expected_commit=self.config.official_evaluation.upstream_commit,
            nltk_data_dir=self.config.official_evaluation.nltk_data_dir,
            db_ids=db_ids,
        )

    @unittest.skipUnless(
        _real_official_assets_available(),
        "pinned official-evaluation dependencies/data are not installed",
    )
    def test_real_preflight_records_pinned_sources_and_variants(self) -> None:
        db_ids = [sample.db_id for sample in self.config.smoke.samples]
        report = self._preflight(db_ids)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(
            report["actual_commit"],
            "e97acc546ecbee8fa27fa8dbf025ef61493a876c",
        )
        self.assertEqual(report["dependencies"], EXPECTED_DEPENDENCIES)
        self.assertTrue(report["nltk_resources"]["punkt"])
        self.assertTrue(report["nltk_resources"]["punkt_tab"])
        self.assertEqual(set(report["databases"]), set(db_ids))
        for db_id in db_ids:
            self.assertGreater(report["databases"][db_id]["variant_count"], 1)
            self.assertEqual(
                len(report["databases"][db_id]["directory_sha256"]), 64
            )

    @unittest.skipUnless(
        _real_official_assets_available(),
        "pinned official-evaluation dependencies/data are not installed",
    )
    def test_gold_self_match_scores_both_official_metrics(self) -> None:
        example = self.dataset.get_example(self.config.smoke.samples[0].index)
        item = OfficialEvaluationItem(
            example_id="dev:%d" % example.index,
            db_id=example.db_id,
            gold_sql=example.gold_sql,
            predicted_sql=example.gold_sql,
        )
        report = self._preflight([example.db_id])
        result = evaluate_official(
            [item],
            evaluator_root=self.config.official_evaluation.evaluator_root,
            database_root=self.config.official_evaluation.test_suite_database_root,
            tables_path=self.dataset.tables_path,
            expected_commit=self.config.official_evaluation.upstream_commit,
            nltk_data_dir=self.config.official_evaluation.nltk_data_dir,
            timeout_seconds=10.0,
            preflight_report=report,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["processed_examples"], 1)
        scored = result["results"][0]
        self.assertEqual(scored["example_id"], item.example_id)
        self.assertEqual(scored["exact_set_match"]["status"], "scored")
        self.assertTrue(scored["exact_set_match"]["match"])
        self.assertEqual(scored["test_suite"]["status"], "scored")
        self.assertTrue(scored["test_suite"]["match"])
        self.assertEqual(
            result["policy"]["official_worker_thread_environment"],
            {
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
        )

    @unittest.skipUnless(
        _real_official_assets_available(),
        "pinned official-evaluation dependencies/data are not installed",
    )
    def test_adapter_itself_blocks_unsafe_test_suite_prediction(self) -> None:
        example = self.dataset.get_example(self.config.smoke.samples[0].index)
        item = OfficialEvaluationItem(
            example_id="dev:%d" % example.index,
            db_id=example.db_id,
            gold_sql=example.gold_sql,
            predicted_sql="DELETE FROM missing_table",
        )
        result = evaluate_official(
            [item],
            evaluator_root=self.config.official_evaluation.evaluator_root,
            database_root=self.config.official_evaluation.test_suite_database_root,
            tables_path=self.dataset.tables_path,
            expected_commit=self.config.official_evaluation.upstream_commit,
            nltk_data_dir=self.config.official_evaluation.nltk_data_dir,
            timeout_seconds=10.0,
            preflight_report=self._preflight([example.db_id]),
        )
        scored = result["results"][0]
        self.assertFalse(scored["exact_set_match"]["match"])
        self.assertFalse(scored["test_suite_eligibility"]["eligible"])
        self.assertEqual(
            scored["test_suite_eligibility"]["safety_error"]["status"],
            "unsafe_sql",
        )
        self.assertEqual(
            scored["test_suite"]["status"], "prediction_ineligible"
        )
        self.assertFalse(scored["test_suite"]["match"])

    @unittest.skipUnless(
        _real_official_assets_available(),
        "pinned official-evaluation dependencies/data are not installed",
    )
    def test_adapter_blocks_quoted_commented_memory_function(self) -> None:
        example = self.dataset.get_example(self.config.smoke.samples[0].index)
        item = OfficialEvaluationItem(
            example_id="dev:%d" % example.index,
            db_id=example.db_id,
            gold_sql=example.gold_sql,
            predicted_sql='SELECT "randomblob"/**/(1000000000)',
        )
        result = evaluate_official(
            [item],
            evaluator_root=self.config.official_evaluation.evaluator_root,
            database_root=self.config.official_evaluation.test_suite_database_root,
            tables_path=self.dataset.tables_path,
            expected_commit=self.config.official_evaluation.upstream_commit,
            nltk_data_dir=self.config.official_evaluation.nltk_data_dir,
            timeout_seconds=10.0,
            max_sql_bytes=self.config.execution.max_sql_bytes,
            preflight_report=self._preflight([example.db_id]),
        )
        scored = result["results"][0]
        self.assertFalse(scored["test_suite_eligibility"]["eligible"])
        self.assertEqual(
            scored["test_suite_eligibility"]["safety_error"]["denied_action"],
            "SQLITE_FUNCTION:randomblob",
        )
        self.assertEqual(scored["test_suite"]["status"], "prediction_ineligible")

    @unittest.skipUnless(
        _real_official_assets_available(),
        "pinned official-evaluation dependencies/data are not installed",
    )
    def test_official_value_normalization_is_not_gated_by_raw_execution(self) -> None:
        example = self.dataset.get_example(69)
        prediction = example.gold_sql.replace(">  1", "> value")
        raw_execution = execute_sql(
            self.dataset.database_path(example.db_id),
            prediction,
            timeout_seconds=self.config.execution.timeout_seconds,
            max_sql_bytes=self.config.execution.max_sql_bytes,
            max_result_rows=self.config.execution.max_result_rows,
            max_result_bytes=self.config.execution.max_result_bytes,
            worker_memory_limit_bytes=(
                self.config.execution.worker_memory_limit_bytes
            ),
        )
        self.assertEqual(raw_execution.status, "execution_error")

        result = evaluate_official(
            [
                OfficialEvaluationItem(
                    example_id="dev:69",
                    db_id=example.db_id,
                    gold_sql=example.gold_sql,
                    predicted_sql=prediction,
                )
            ],
            evaluator_root=self.config.official_evaluation.evaluator_root,
            database_root=self.config.official_evaluation.test_suite_database_root,
            tables_path=self.dataset.tables_path,
            expected_commit=self.config.official_evaluation.upstream_commit,
            nltk_data_dir=self.config.official_evaluation.nltk_data_dir,
            timeout_seconds=10.0,
            max_sql_bytes=self.config.execution.max_sql_bytes,
            preflight_report=self._preflight([example.db_id]),
        )
        scored = result["results"][0]
        self.assertTrue(scored["prediction_normalization"]["applied"])
        self.assertTrue(scored["test_suite_eligibility"]["eligible"])
        self.assertTrue(scored["exact_set_match"]["match"])
        self.assertTrue(scored["test_suite"]["match"])

    @unittest.skipUnless(
        _real_official_assets_available(),
        "pinned official-evaluation dependencies/data are not installed",
    )
    def test_metric_validity_is_independent_when_test_suite_times_out(self) -> None:
        example = self.dataset.get_example(self.config.smoke.samples[0].index)

        def fake_worker(operation, *_args, **_kwargs):
            if operation == "exact":
                return {
                    "operation": "exact",
                    "status": "scored",
                    "match": True,
                    "hardness": "medium",
                    "error": None,
                }
            return {
                "operation": "test_suite",
                "status": "evaluator_timeout",
                "match": False,
                "error": {"type": "evaluator_timeout", "message": "fixture"},
            }

        with mock.patch(
            "text2sql.core.official_eval._run_worker", side_effect=fake_worker
        ):
            result = evaluate_official(
                [
                    OfficialEvaluationItem(
                        example_id="dev:%d" % example.index,
                        db_id=example.db_id,
                        gold_sql=example.gold_sql,
                        predicted_sql=example.gold_sql,
                    )
                ],
                evaluator_root=self.config.official_evaluation.evaluator_root,
                database_root=(
                    self.config.official_evaluation.test_suite_database_root
                ),
                tables_path=self.dataset.tables_path,
                expected_commit=self.config.official_evaluation.upstream_commit,
                nltk_data_dir=self.config.official_evaluation.nltk_data_dir,
                timeout_seconds=10.0,
                max_sql_bytes=self.config.execution.max_sql_bytes,
                preflight_report=self._preflight([example.db_id]),
            )
        self.assertFalse(result["ok"])
        self.assertTrue(result["metrics"]["exact_set_match"]["valid"])
        self.assertFalse(result["metrics"]["test_suite"]["valid"])
        self.assertEqual(
            result["metrics"]["test_suite"]["infrastructure_failures"], 1
        )

    def test_duplicate_ids_and_path_traversal_are_rejected(self) -> None:
        first = OfficialEvaluationItem("dev:1", "db", "SELECT 1", "SELECT 1")
        duplicate = OfficialEvaluationItem(
            "dev:1", "db", "SELECT 1", "SELECT 1"
        )
        traversal = OfficialEvaluationItem(
            "dev:2", "../db", "SELECT 1", "SELECT 1"
        )
        common = {
            "evaluator_root": Path("."),
            "database_root": Path("."),
            "tables_path": Path("tables.json"),
            "expected_commit": "0" * 40,
            "nltk_data_dir": None,
            "timeout_seconds": 1.0,
            "preflight_report": {"ok": True},
        }
        with self.assertRaises(OfficialEvaluationError):
            evaluate_official([first, duplicate], **common)
        with self.assertRaises(OfficialEvaluationError):
            evaluate_official([traversal], **common)

    def test_worker_timeout_is_structured(self) -> None:
        with mock.patch(
            "text2sql.core.official_eval.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["python"], 0.01),
        ):
            result = _run_worker(
                "test_suite",
                {"operation": "test_suite"},
                timeout_seconds=0.01,
                nltk_data_dir=None,
            )
        self.assertEqual(result["status"], "evaluator_timeout")
        self.assertFalse(result["match"])
        self.assertGreaterEqual(result["parent_elapsed_ns"], 0)

    def test_official_worker_forces_single_thread_numerical_libraries(self) -> None:
        variable_names = (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
        inherited = {name: "64" for name in variable_names}
        inherited["PRESERVED_VARIABLE"] = "preserved"
        with mock.patch.dict(os.environ, inherited, clear=True):
            environment = _worker_environment(None)

        self.assertEqual(
            {name: environment[name] for name in variable_names},
            {name: "1" for name in variable_names},
        )
        self.assertEqual(environment["PRESERVED_VARIABLE"], "preserved")
        self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")

    def test_linux_official_worker_memory_limit_is_applied(self) -> None:
        fake_resource = types.ModuleType("resource")
        fake_resource.RLIMIT_AS = 9
        fake_resource.RLIM_INFINITY = -1
        limits = [(-1, -1)]

        def getrlimit(_resource):
            return limits[-1]

        def setrlimit(_resource, value):
            limits.append(value)

        fake_resource.getrlimit = getrlimit
        fake_resource.setrlimit = setrlimit
        with mock.patch("text2sql.core.official_eval_worker.sys.platform", "linux"):
            with mock.patch.dict(sys.modules, {"resource": fake_resource}):
                _apply_worker_memory_limit(
                    {"worker_memory_limit_bytes": 1024 * 1024}
                )
        self.assertEqual(limits[-1], (1024 * 1024, -1))

    def test_preflight_distinguishes_missing_generated_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator_root = root / "evaluator"
            evaluator_root.mkdir()
            for filename in (
                "evaluation.py",
                "process_sql.py",
                "exec_eval.py",
                "parse.py",
                "LICENSE",
            ):
                (evaluator_root / filename).write_text("fixture\n", encoding="utf-8")
            (evaluator_root / "UPSTREAM_COMMIT").write_text(
                "0" * 40 + "\n", encoding="utf-8"
            )
            database_root = root / "database"
            db_directory = database_root / "fixture_db"
            db_directory.mkdir(parents=True)
            (db_directory / "fixture_db.sqlite").write_bytes(b"not-a-real-db")
            tables_path = root / "tables.json"
            tables_path.write_text("[]\n", encoding="utf-8")
            report = validate_official_environment(
                evaluator_root=evaluator_root,
                database_root=database_root,
                tables_path=tables_path,
                expected_commit="0" * 40,
                nltk_data_dir=self.config.official_evaluation.nltk_data_dir,
                db_ids=["fixture_db"],
            )
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("generated test-suite variants are missing" in error for error in report["errors"])
        )
        self.assertTrue(
            any("source hash mismatch" in error for error in report["errors"])
        )


if __name__ == "__main__":
    unittest.main()
