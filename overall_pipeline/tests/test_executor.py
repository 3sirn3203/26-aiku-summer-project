from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from text2sql.core.executor import execute_sql


class ExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="sqlite fixture # ")
        self.db_path = Path(self.temporary.name) / "fixture ?.sqlite"
        connection = sqlite3.connect(str(self.db_path))
        connection.executescript(
            """
            CREATE TABLE items(id INTEGER PRIMARY KEY, name TEXT, score REAL, payload BLOB);
            INSERT INTO items(name, score, payload) VALUES
              ('alpha', 1.5, X'00FF'),
              ('beta', 2.5, X'0102'),
              ('gamma', NULL, NULL);
            """
        )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self, sql: str, **overrides):
        settings = {
            "timeout_seconds": 2.0,
            "max_sql_bytes": 100000,
            "max_result_rows": 100,
            "max_result_bytes": 100000,
        }
        settings.update(overrides)
        return execute_sql(self.db_path, sql, **settings)

    def _hash(self) -> str:
        return hashlib.sha256(self.db_path.read_bytes()).hexdigest()

    def test_select_cte_empty_and_blob(self) -> None:
        selected = self._run("SELECT id, name, payload FROM items ORDER BY id")
        self.assertEqual(selected.status, "success")
        self.assertEqual(len(selected.rows), 3)
        self.assertEqual(selected.rows[0][2], {"__bytes_base64__": "AP8="})

        cte = self._run(
            "WITH RECURSIVE n(x) AS "
            "(VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<3) SELECT sum(x) FROM n"
        )
        self.assertEqual(cte.status, "success")
        self.assertEqual(cte.rows, [[6]])

        empty = self._run("SELECT * FROM items WHERE id < 0")
        self.assertEqual(empty.status, "success")
        self.assertEqual(empty.rows, [])

    def test_timing_boundaries_are_recorded_for_success(self) -> None:
        result = self._run("SELECT id, name FROM items ORDER BY id")

        self.assertEqual(result.status, "success")
        self.assertIsNotNone(result.query_elapsed_ns)
        self.assertIsNotNone(result.worker_elapsed_ns)
        self.assertIsNotNone(result.parent_elapsed_ns)
        self.assertGreater(result.query_elapsed_ns, 0)
        self.assertGreaterEqual(result.worker_elapsed_ns, result.query_elapsed_ns)
        self.assertGreaterEqual(result.parent_elapsed_ns, result.worker_elapsed_ns)
        self.assertEqual(result.vm_step_progress_interval, 1000)
        self.assertIsNotNone(result.vm_steps_lower_bound)
        self.assertGreaterEqual(result.vm_steps_lower_bound, 0)
        self.assertEqual(
            result.vm_steps_upper_bound_exclusive,
            result.vm_steps_lower_bound + result.vm_step_progress_interval,
        )
        self.assertTrue(result.vm_step_measurement_complete)
        self.assertAlmostEqual(
            result.elapsed_seconds,
            result.parent_elapsed_ns / 1_000_000_000,
            places=9,
        )
        self.assertEqual(result.to_dict()["query_elapsed_ns"], result.query_elapsed_ns)

    def test_syntax_and_execution_errors_are_distinct(self) -> None:
        syntax = self._run("SELEC 1")
        execution = self._run("SELECT missing FROM items")
        self.assertEqual(syntax.status, "syntax_error")
        self.assertEqual(execution.status, "execution_error")
        self.assertIsNone(syntax.query_elapsed_ns)
        self.assertIsNone(syntax.worker_elapsed_ns)
        self.assertIsNone(syntax.parent_elapsed_ns)
        self.assertIsNotNone(execution.query_elapsed_ns)
        self.assertGreater(execution.query_elapsed_ns, 0)
        self.assertIsNotNone(execution.worker_elapsed_ns)
        self.assertIsNotNone(execution.parent_elapsed_ns)

    def test_mutations_and_pragma_are_blocked_and_db_is_unchanged(self) -> None:
        before = self._hash()
        statements = (
            "INSERT INTO items(name) VALUES ('bad')",
            "UPDATE items SET name='bad'",
            "DELETE FROM items",
            "DELETE FROM missing_table",
            "WITH x AS (SELECT 1) DELETE FROM missing_table",
            "DROP TABLE items",
            "CREATE TABLE bad(id INTEGER)",
            "ALTER TABLE items ADD COLUMN bad TEXT",
            "PRAGMA user_version",
            "ATTACH DATABASE ':memory:' AS other",
            "SELECT load_extension('bad')",
        )
        for statement in statements:
            with self.subTest(statement=statement):
                result = self._run(statement)
                self.assertEqual(result.status, "unsafe_sql", result.to_dict())
                self.assertIsNotNone(result.denied_action)
        self.assertEqual(before, self._hash())

    def test_multiple_statements_are_blocked_before_execution(self) -> None:
        result = self._run("SELECT 1; DROP TABLE items;")
        self.assertEqual(result.status, "unsafe_sql")
        self.assertEqual(result.error_type, "multiple_statements")

    def test_timeout_and_followup_query(self) -> None:
        result = self._run(
            "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n) "
            "SELECT sum(x) FROM n",
            timeout_seconds=0.25,
        )
        self.assertEqual(result.status, "execution_timeout")
        self.assertIsNotNone(result.parent_elapsed_ns)
        self.assertGreater(result.parent_elapsed_ns, 0)
        self.assertAlmostEqual(
            result.elapsed_seconds,
            result.parent_elapsed_ns / 1_000_000_000,
            places=9,
        )
        self.assertIsNotNone(result.query_elapsed_ns)
        self.assertGreater(result.query_elapsed_ns, 0)
        self.assertIsNotNone(result.vm_steps_lower_bound)
        self.assertGreater(result.vm_steps_lower_bound, 0)
        self.assertEqual(result.vm_step_progress_interval, 1000)
        self.assertFalse(result.vm_step_measurement_complete)
        self.assertIsNone(result.vm_steps_upper_bound_exclusive)
        self.assertEqual(self._run("SELECT count(*) FROM items").rows, [[3]])

    def test_result_limits(self) -> None:
        rows = self._run("SELECT * FROM items", max_result_rows=1)
        self.assertEqual(rows.status, "result_limit")
        self.assertTrue(rows.truncated)
        self.assertFalse(rows.vm_step_measurement_complete)
        self.assertIsNone(rows.vm_steps_upper_bound_exclusive)
        size = self._run("SELECT '%s'" % ("x" * 1000), max_result_bytes=20)
        self.assertEqual(size.status, "result_limit")
        self.assertTrue(size.truncated)

    def test_memory_amplifying_functions_are_blocked(self) -> None:
        for sql in (
            "SELECT randomblob(1000000000)",
            "SELECT zeroblob(1000000000)",
            "SELECT printf('%1000000000s', 'x')",
            "SELECT format('%1000000000s', 'x')",
            "SELECT eval('DROP TABLE items')",
            'SELECT "randomblob"(4)',
            "SELECT [zeroblob](4)",
            "SELECT `printf`('%4s', 'x')",
            "SELECT randomblob/**/(4)",
            "SELECT randomblob -- comment\n (4)",
            'SELECT "fts3_tokenizer"/* comment */("simple")',
        ):
            with self.subTest(sql=sql):
                result = self._run(sql)
                self.assertEqual(result.status, "unsafe_sql")
                self.assertTrue(result.denied_action.startswith("SQLITE_FUNCTION:"))


if __name__ == "__main__":
    unittest.main()
