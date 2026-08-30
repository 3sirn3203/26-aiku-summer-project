from __future__ import annotations

import unittest

from text2sql.core.models import SchemaMetadata
from text2sql.single_turn.prompt import build_messages
from text2sql.core.schema import serialize_schema


class SchemaPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = SchemaMetadata(
            db_id="fixture",
            table_names=("people", "visits"),
            column_names=(
                (-1, "*"),
                (0, "person_id"),
                (0, "name"),
                (1, "visit_id"),
                (1, "person_id"),
            ),
            column_types=("text", "number", "text", "number", "number"),
            primary_keys=(1, 3),
            foreign_keys=((4, 1),),
        )

    def test_schema_uses_original_metadata(self) -> None:
        serialized = serialize_schema(self.metadata)
        self.assertIn('Table "people"', serialized)
        self.assertIn('"person_id" NUMBER PRIMARY KEY', serialized)
        self.assertIn('"visits"."person_id" -> "people"."person_id"', serialized)
        self.assertNotIn("CREATE TABLE", serialized)

    def test_prompt_contains_only_question_and_schema(self) -> None:
        serialized = serialize_schema(self.metadata)
        messages = build_messages("List every person.", serialized)
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        rendered = "\n".join(message["content"] for message in messages)
        self.assertIn("List every person.", rendered)
        self.assertIn(serialized, rendered)
        self.assertNotIn("gold_sql", rendered)
        self.assertNotIn("execution result", rendered.casefold())


if __name__ == "__main__":
    unittest.main()

