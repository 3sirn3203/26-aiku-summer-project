from __future__ import annotations

import unittest

from text2sql.core.reporting import (
    SUMMARY_METRIC_KEYS,
    compact_run_manifest,
    empty_metric_summary,
    metric_summary,
)


class ReportingTests(unittest.TestCase):
    def test_metric_summary_contains_only_five_scalar_metrics(self) -> None:
        summary = metric_summary(
            test_suite_accuracy=0.4,
            exact_set_match_accuracy=0.5,
            result_match_accuracy=0.6,
            prediction_vm_steps={
                "complete_vm_steps_lower_bound": {"mean": 1234.5}
            },
            prediction_query_timing={"query_elapsed_ms": {"mean": 7.25}},
        )
        self.assertEqual(tuple(summary), SUMMARY_METRIC_KEYS)
        self.assertEqual(summary["mean_vm_steps"], 1234.5)
        self.assertEqual(summary["mean_latency_ms"], 7.25)

    def test_failed_summary_has_the_same_five_null_metrics(self) -> None:
        self.assertEqual(
            empty_metric_summary(),
            {key: None for key in SUMMARY_METRIC_KEYS},
        )

    def test_manifest_compaction_removes_duplicated_payloads(self) -> None:
        compact = compact_run_manifest(
            {
                "schema_version": 6,
                "run_id": "run",
                "status": "completed",
                "config": {"model": {"id": "model"}},
                "source_config": {"model": {"id": "model"}},
                "source_config_sha256": "source-config-hash",
                "local_checkpoint": {"path": "model", "files": ["weights"]},
                "source": {
                    "package_version": "1",
                    "python_tree_sha256": "source-tree-hash",
                    "python_files_sha256": {"a.py": "file-hash"},
                    "scope": ["**/*.py"],
                },
                "dataset": {
                    "split": "dev",
                    "selected_database_sha256_before": {"db": "hash"},
                    "selected_database_sha256_after": {"db": "hash"},
                    "selected_databases_unchanged": True,
                    "validation": {"ok": True},
                },
                "official_evaluation": {
                    "enabled": True,
                    "preflight_sha256": "preflight-hash",
                    "preflight": {
                        "ok": True,
                        "databases": {"db": {"large": "payload"}},
                    },
                },
                "generation": {
                    "contract": {"large": "payload"},
                    "contract_sha256": "generation-hash",
                    "attempts": [{"worker_statuses": [{"backend": {"large": 1}}]}],
                },
            }
        )
        self.assertNotIn("source_config", compact)
        self.assertNotIn("local_checkpoint", compact)
        self.assertNotIn("python_files_sha256", compact["source"])
        self.assertNotIn("selected_database_sha256_after", compact["dataset"])
        self.assertNotIn("validation", compact["dataset"])
        self.assertNotIn("databases", compact["official_evaluation"]["preflight"])
        self.assertNotIn("contract", compact["generation"])
        self.assertEqual(compact["generation"]["attempt_count"], 1)


if __name__ == "__main__":
    unittest.main()
