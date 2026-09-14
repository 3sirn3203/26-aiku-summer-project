"""Adapter for the public Spider test-suite evaluator's structural metrics."""
from copy import deepcopy
import json
from pathlib import Path
import sys

from .data import database_path


class ExactMatchScorer:
    """Persistent Spider structural exact-match scorer with reusable parses."""

    def __init__(self, evaluator_path, tables, database_dir, nltk_data=None):
        sys.path.insert(0, str(Path(evaluator_path).resolve()))
        import evaluation as ev
        from process_sql import Schema, get_schema, get_sql
        if nltk_data:
            import nltk
            nltk.data.path.insert(0, str(Path(nltk_data).resolve()))
        self.ev, self.Schema = ev, Schema
        self.get_schema, self.get_sql = get_schema, get_sql
        self.database_dir = database_dir
        self.kmaps = ev.build_foreign_key_map_from_json(tables)
        self.evaluator = ev.Evaluator()
        self.schemas, self.gold = {}, {}

    def _schema(self, db_id):
        if db_id not in self.schemas:
            path = database_path(self.database_dir, db_id)
            self.schemas[db_id] = self.Schema(self.get_schema(str(path)))
        return self.schemas[db_id]

    def _normalize(self, sql, schema, db_id):
        sql = deepcopy(sql)
        valid = self.ev.build_valid_col_units(sql["from"]["table_units"], schema)
        return self.ev.rebuild_sql_col(valid, self.ev.rebuild_sql_val(sql), self.kmaps[db_id])

    def score(self, example, prediction):
        db_id, query = example["db_id"], example["query"]
        schema = self._schema(db_id)
        key = (db_id, query)
        if key not in self.gold:
            parsed = self.get_sql(schema, query)
            self.gold[key] = (self._normalize(parsed, schema, db_id),
                              self.evaluator.eval_hardness(parsed))
        gold, difficulty = self.gold[key]
        try:
            pred = self._normalize(self.get_sql(schema, prediction), schema, db_id)
            exact = bool(self.evaluator.eval_exact_match(pred, deepcopy(gold)))
        except (AssertionError, ValueError, IndexError, KeyError):
            exact = False
        return {"difficulty": difficulty, "exact_match": exact}


def run(payload):
    scorer = ExactMatchScorer(payload["evaluator_path"], payload["tables"],
                              payload["database_dir"], payload.get("nltk_data"))
    return [scorer.score(row, row["prediction"]) for row in payload["rows"]]


if __name__ == "__main__":
    print(json.dumps(run(json.load(sys.stdin))))
