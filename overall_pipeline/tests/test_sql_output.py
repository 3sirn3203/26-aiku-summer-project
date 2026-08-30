from __future__ import annotations

import unittest

from text2sql.core.sql_output import extract_sql
from text2sql.core.sql_text import split_sql_statements, top_level_statement_keyword


class SqlOutputTests(unittest.TestCase):
    def test_plain_and_fenced_sql(self) -> None:
        self.assertEqual(extract_sql("SELECT 1;").sql, "SELECT 1")
        self.assertEqual(extract_sql("```sql\nSELECT 2;\n```").sql, "SELECT 2")

    def test_semicolon_in_strings_and_comments(self) -> None:
        statements = split_sql_statements("SELECT '; DROP TABLE x'; -- done\n")
        self.assertEqual(statements, ["SELECT '; DROP TABLE x'"])

    def test_multiple_statements_are_rejected(self) -> None:
        parsed = extract_sql("SELECT 1; DROP TABLE x;")
        self.assertEqual(parsed.status, "error")
        self.assertEqual(parsed.error_type, "multiple_statements")

    def test_prose_is_not_silently_repaired(self) -> None:
        parsed = extract_sql("Here is your query: SELECT 1")
        self.assertEqual(parsed.status, "error")
        self.assertEqual(parsed.error_type, "sql_parse_error")

    def test_unsafe_statement_reaches_executor_for_classification(self) -> None:
        parsed = extract_sql("DROP TABLE people")
        self.assertEqual(parsed.status, "success")
        self.assertEqual(parsed.sql, "DROP TABLE people")

    def test_main_statement_keyword_skips_cte_bodies_and_quoted_text(self) -> None:
        self.assertEqual(
            top_level_statement_keyword(
                "WITH x AS (SELECT 'DELETE FROM hidden') SELECT * FROM x"
            ),
            "SELECT",
        )
        self.assertEqual(
            top_level_statement_keyword(
                "WITH x AS (SELECT 1) DELETE FROM people WHERE id IN (SELECT * FROM x)"
            ),
            "DELETE",
        )
        self.assertEqual(
            top_level_statement_keyword(
                "/* SELECT */ WITH RECURSIVE x(v) AS (VALUES(1)) UPDATE people SET id=1"
            ),
            "UPDATE",
        )


if __name__ == "__main__":
    unittest.main()
