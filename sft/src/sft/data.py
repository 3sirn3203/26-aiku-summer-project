import json
from pathlib import Path

from overall_pipeline.core.data import serialize_schema
from overall_pipeline.core.protocol import action_messages
from overall_pipeline.core.rollout import answer_only_messages


def read_rows(path):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Empty or invalid dataset: {path}")
    return rows


def schema_map(path):
    return {row["db_id"]: serialize_schema(row) for row in read_rows(path)}


def build_examples(data_path, tables_path):
    schemas = schema_map(tables_path)
    examples = []
    for index, row in enumerate(read_rows(data_path)):
        if row["db_id"] not in schemas:
            raise ValueError(f"Missing schema for db_id={row['db_id']}")
        examples.append({
            "example_id": f"{Path(data_path).stem}:{index}",
            "messages": action_messages(answer_only_messages(row, schemas[row["db_id"]]), "answer"),
            "response": f"<answer>\n{row['query']}\n</answer>",
        })
    return examples
