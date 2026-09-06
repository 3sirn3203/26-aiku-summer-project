from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sft_verifier.build_sft_dataset import build_examples, feedback_for
from sft_verifier.mutate_gold import generate_mutations
from sft_verifier.prepare_splits import build_manifest


class VerifierDataTests(unittest.TestCase):
    def test_database_splits_are_disjoint_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            tables = []
            for index in range(10):
                db_id = "db_%d" % index
                records.append(
                    {"db_id": db_id, "question": "question", "query": "SELECT 1", "sql": {}}
                )
                tables.append(
                    {
                        "db_id": db_id,
                        "table_names_original": ["items"],
                        "column_names_original": [[-1, "*"], [0, "id"]],
                        "column_types": ["text", "number"],
                        "primary_keys": [1],
                        "foreign_keys": [],
                    }
                )
            for filename in ("train_spider.json", "train_others.json"):
                (root / filename).write_text(json.dumps(records), encoding="utf-8")
            (root / "tables.json").write_text(json.dumps(tables), encoding="utf-8")
            first = build_manifest(root, 7)
            second = build_manifest(root, 7)
            self.assertEqual(first["database_splits"], second["database_splits"])
            groups = [set(first["database_splits"][name]) for name in ("train", "validation", "test")]
            self.assertTrue(all(groups))
            self.assertFalse(groups[0] & groups[1])
            self.assertFalse(groups[0] & groups[2])
            self.assertFalse(groups[1] & groups[2])

    def test_mutations_are_single_edit_candidates_with_feedback(self) -> None:
        mutations = generate_mutations(
            "SELECT DISTINCT name FROM singer WHERE age > 20 ORDER BY age DESC LIMIT 1"
        )
        categories = {item[0] for item in mutations}
        self.assertTrue(
            {"missing_distinct", "wrong_order_direction", "missing_limit", "missing_filter", "wrong_comparison"}
            <= categories
        )
        self.assertTrue(all(sql and feedback for _, sql, feedback in mutations))

    def test_sft_builder_strips_gold_and_balances_decisions(self) -> None:
        base = {
            "schema_version": 1,
            "base_example_id": "train:0",
            "db_id": "db",
            "split": "train",
            "question": "How many active employees are there?",
            "gold_sql": "SELECT COUNT(*) FROM employees WHERE active = 1",
            "serialized_schema": 'Database: db\nTable "employees"\n  - "active" NUMBER',
            "planner_output": {"plan": "Count only active employee records."},
            "candidate_raw_output": "SELECT COUNT(*) FROM employees",
            "candidate_sql": "SELECT COUNT(*) FROM employees",
            "sql_parsing": {"status": "success", "sql": "SELECT COUNT(*) FROM employees"},
            "execution_observation": {"status": "success", "columns": ["COUNT(*)"], "rows": [[2]], "truncated": False},
            "label_status": "accepted",
            "official_evaluation": {"test_suite": {"status": "scored", "match": False}},
        }
        stop = {
            **base,
            "example_id": "train:0:gold",
            "source": "gold",
            "decision_label": "stop",
        }
        cont = {
            **base,
            "example_id": "train:0:mutation",
            "source": "gold_mutation",
            "decision_label": "continue",
            "mutation": {"category": "missing_filter", "feedback": "Restore the active-employee filter."},
        }
        examples = build_examples([stop, cont], seed=1, balance=True)
        self.assertEqual(len(examples), 2)
        self.assertEqual({item["label"]["decision"] for item in examples}, {"stop", "continue"})
        self.assertTrue(all("gold_sql" not in item for item in examples))
        self.assertTrue(all(set(item["label"]) == {"decision", "feedback"} for item in examples))

    def test_unknown_semantic_mismatch_is_not_given_guessed_feedback(self) -> None:
        self.assertIsNone(
            feedback_for(
                {
                    "decision_label": "continue",
                    "sql_parsing": {"status": "success"},
                    "execution_observation": {"status": "success"},
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
