"""Adapter for the public Spider test-suite evaluator's structural metrics."""
from copy import deepcopy
import json
from pathlib import Path
import sys


def run(payload):
    sys.path.insert(0, str(Path(payload["evaluator_path"]).resolve()))
    import evaluation as ev
    from process_sql import Schema, get_schema, get_sql
    if payload.get("nltk_data"):
        import nltk
        nltk.data.path.insert(0, str(Path(payload["nltk_data"]).resolve()))
    from .data import database_path
    kmaps = ev.build_foreign_key_map_from_json(payload["tables"])
    evaluator = ev.Evaluator()
    results = []
    for row in payload["rows"]:
        schema = Schema(get_schema(str(database_path(payload["database_dir"], row["db_id"]))))
        gold = get_sql(schema, row["query"])
        difficulty = evaluator.eval_hardness(gold)
        try:
            pred = get_sql(schema, row["prediction"])
        except (AssertionError, ValueError, IndexError, KeyError):
            results.append({"difficulty": difficulty, "exact_match": False})
            continue
        normalized = []
        for sql in (pred, gold):
            sql = deepcopy(sql)
            valid = ev.build_valid_col_units(sql["from"]["table_units"], schema)
            normalized.append(ev.rebuild_sql_col(valid, ev.rebuild_sql_val(sql), kmaps[row["db_id"]]))
        results.append({"difficulty": difficulty, "exact_match": bool(evaluator.eval_exact_match(*normalized))})
    return results


if __name__ == "__main__":
    print(json.dumps(run(json.load(sys.stdin))))
