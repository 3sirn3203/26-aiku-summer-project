from dataclasses import asdict
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import sqlite3
import subprocess
import sys
import time

import torch
import transformers

from rrcm_sql.data import database_path, read_json, schema_map
from rrcm_sql.sql import Executor, Judge, reward

from .worker_pool import GenerationPool


def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _signature(cfg, rows):
    payload = {
        "config": asdict(cfg),
        "test_json_sha256": _hash_file(cfg.data.test_json),
        "test_tables_sha256": _hash_file(cfg.data.test_tables),
        "selected": [int(row["_index"]) for row in rows],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def _prepare_output(cfg, output, rows, devices):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    shards = output / "generation_shards"
    manifest_path = output / "run_manifest.json"
    signature, payload = _signature(cfg, rows)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("signature") != signature:
            raise ValueError("Existing baseline output was created with different config/data")
    else:
        if any(output.iterdir()):
            raise FileExistsError(f"Output directory is not an existing compatible run: {output}")
        manifest = {
            "signature": signature,
            **payload,
            "devices": list(devices),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sqlite": sqlite3.sqlite_version,
            "argv": sys.argv,
        }
        _atomic_json(manifest_path, manifest)
        cfg.save(output / "config.json")
    shards.mkdir(exist_ok=True)
    return output, shards


def _load_shards(shards, rows):
    records, expected = {}, {int(row["_index"]) for row in rows}
    for path in sorted(shards.glob("*.json")):
        try:
            index = int(path.stem)
            record = json.loads(path.read_text())
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid generation shard: {path}") from exc
        if index not in expected or index in records or record.get("dataset_index") != index:
            raise ValueError(f"Unexpected or duplicate generation shard: {path}")
        records[index] = record
    return records


def _validate_suite(root, rows):
    if not root:
        return
    root = Path(root)
    missing = []
    for db_id in sorted({row["db_id"] for row in rows}):
        if len(list((root / db_id).glob("*.sqlite"))) < 2:
            missing.append(db_id)
    if missing:
        raise ValueError(f"Incomplete test-suite database bundle; missing variants for {missing[:10]}")


def _score_records(cfg, rows, records):
    sql_cfg = deepcopy(cfg.sql)
    sql_cfg.evaluator_path = cfg.evaluation.evaluator_path
    sql_cfg.nltk_data = cfg.evaluation.nltk_data
    sql_cfg.suite_database_dir = cfg.evaluation.test_suite_database_dir
    sql_cfg.reward_metric = "execution"
    _validate_suite(sql_cfg.suite_database_dir, rows)
    executor = Executor(sql_cfg)
    judge = Judge(executor, cfg.data.test_database_dir, cfg.data.test_tables)
    evaluate_suite = bool(sql_cfg.suite_database_dir)
    for position, row in enumerate(rows, 1):
        record = records[int(row["_index"])]
        if record.get("final_sql"):
            scores = judge.score(row, record["final_sql"], evaluate_suite=evaluate_suite)
        else:
            scores = {
                "outcome": "non_executable",
                "execution_correct": False,
                "exact_match": False,
                "test_suite_correct": False if evaluate_suite else None,
                "final_execution": {"ok": False, "kind": record.get("termination", "generation_failure"),
                                    "error": "No final SQL was generated"},
            }
        record["scores"] = scores
        record["outcome"] = scores["outcome"]
        record["reward"] = reward(
            scores["outcome"], record["intermediate_count"],
            cfg.generation.max_intermediate, exact_match=scores["exact_match"])
        print(f"{position}/{len(rows)} scored {record['outcome']}", flush=True)


def _summarize(records):
    n = len(records)
    correct = [r for r in records if r["outcome"] == "correct"]
    intermediate = [q for r in records for q in r["intermediate"]]
    duplicates = 0
    for record in records:
        sql = [" ".join(q["sql"].lower().split()) for q in record["intermediate"]]
        duplicates += len(sql) - len(set(sql))
    return {
        "count": n,
        "execution_accuracy": sum(bool(r["scores"]["execution_correct"]) for r in records) / n,
        "correct_rate": len(correct) / n,
        "executable_incorrect_rate": sum(r["outcome"] == "executable_incorrect" for r in records) / n,
        "non_executable_rate": sum(r["outcome"] == "non_executable" for r in records) / n,
        "intermediate_mean": len(intermediate) / n,
        "correct_intermediate_mean": (sum(r["intermediate_count"] for r in correct) / len(correct)
                                      if correct else None),
        "direct_answer_rate": sum(r["termination"] == "answer" and not r["intermediate"] for r in records) / n,
        "no_intermediate_rate": sum(not r["intermediate"] for r in records) / n,
        "llm_calls_mean": sum(len(r["turns"]) for r in records) / n,
        "input_tokens_mean": sum(r["input_tokens"] for r in records) / n,
        "output_tokens_mean": sum(r["output_tokens"] for r in records) / n,
        "input_tokens_total": sum(r["input_tokens"] for r in records),
        "output_tokens_total": sum(r["output_tokens"] for r in records),
        "invalid_intermediate_rate": (sum(not q["result"]["ok"] for q in intermediate) / len(intermediate)
                                      if intermediate else 0),
        "duplicate_intermediate_rate": duplicates / len(intermediate) if intermediate else 0,
        "generation_seconds_mean": sum(r["elapsed"] for r in records) / n,
        "final_sql_seconds_mean": sum(r["scores"]["final_execution"].get("elapsed", 0)
                                      for r in records) / n,
    }


def _structural_metrics(cfg, rows, records):
    if not cfg.evaluation.evaluator_path:
        return None
    payload = {
        "evaluator_path": cfg.evaluation.evaluator_path,
        "tables": cfg.data.test_tables,
        "nltk_data": cfg.evaluation.nltk_data,
        "database_dir": cfg.data.test_database_dir,
        "rows": [{**row, "prediction": record.get("final_sql") or ""}
                 for row, record in zip(rows, records)],
    }
    result = subprocess.run(
        [sys.executable, "-m", "rrcm_sql.spider_metrics"],
        input=json.dumps(payload), text=True, capture_output=True, check=True, timeout=600)
    return json.loads(result.stdout)


def _write_reports(cfg, output, rows, records, devices, limit, started):
    summary = _summarize(records)
    summary.update({
        "split": "test",
        "mode": cfg.generation.mode,
        "model_name_or_path": cfg.model.name_or_path,
        "evaluation_devices": list(devices),
        "is_subset": limit is not None,
        "test_suite_accuracy": (sum(bool(r["scores"]["test_suite_correct"]) for r in records) / len(records)
                                if cfg.evaluation.test_suite_database_dir else None),
        "test_suite_status": ("computed" if cfg.evaluation.test_suite_database_dir
                              else "not_configured"),
        "correctness_backend": ("spider_result_eq" if cfg.evaluation.evaluator_path
                                else "strict_execution_proxy"),
    })
    if cfg.generation.mode == "single_turn" and any(len(r["turns"]) != 1 for r in records):
        raise RuntimeError("Single-turn invariant violated: every sample must make exactly one LLM call")
    structural = _structural_metrics(cfg, rows, records)
    if structural is None:
        summary["exact_match"] = None
    else:
        summary["exact_match"] = sum(r["exact_match"] for r in structural) / len(records)
        (output / "structural_metrics.json").write_text(json.dumps(structural, indent=2) + "\n")
        difficulty = []
        for name in sorted({r["difficulty"] for r in structural}):
            positions = [i for i, item in enumerate(structural) if item["difficulty"] == name]
            group = [records[i] for i in positions]
            difficulty.append({
                "difficulty": name,
                **_summarize(group),
                "exact_match": sum(structural[i]["exact_match"] for i in positions) / len(positions),
                "test_suite_accuracy": (sum(bool(records[i]["scores"]["test_suite_correct"])
                                            for i in positions) / len(positions)
                                        if cfg.evaluation.test_suite_database_dir else None),
            })
        with (output / "difficulty.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(difficulty[0]))
            writer.writeheader()
            writer.writerows(difficulty)
    summary["seconds"] = time.monotonic() - started
    (output / "trajectories.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    (output / "predictions.sql").write_text(
        "\n".join(" ".join((r.get("final_sql") or "SELECT").split()) for r in records) + "\n")
    (output / "gold.sql").write_text(
        "\n".join(" ".join(row["query"].split()) + "\t" + row["db_id"] for row in rows) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def evaluate(cfg, output_dir, devices, limit=None):
    started = time.monotonic()
    rows = read_json(cfg.data.test_json)
    if not rows:
        raise ValueError("Empty Spider test data")
    indexed = [dict(row, _index=index, example_id=f"test:{index}") for index, row in enumerate(rows)]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        indexed = random.Random(cfg.evaluation.seed).sample(indexed, min(limit, len(indexed)))
        indexed.sort(key=lambda row: row["_index"])
    schemas = schema_map(cfg.data.test_tables)
    missing_schema = {row["db_id"] for row in indexed} - schemas.keys()
    if missing_schema:
        raise ValueError(f"Missing test schemas: {sorted(missing_schema)[:10]}")
    for db_id in {row["db_id"] for row in indexed}:
        database_path(cfg.data.test_database_dir, db_id)
    if cfg.evaluation.evaluator_path:
        preflight = {
            "evaluator_path": cfg.evaluation.evaluator_path,
            "tables": cfg.data.test_tables,
            "nltk_data": cfg.evaluation.nltk_data,
            "database_dir": cfg.data.test_database_dir,
            "rows": [{**indexed[0], "prediction": indexed[0]["query"]}],
        }
        subprocess.run([sys.executable, "-m", "rrcm_sql.spider_metrics"],
                       input=json.dumps(preflight), text=True, capture_output=True,
                       check=True, timeout=60)
    output, shards = _prepare_output(cfg, output_dir, indexed, devices)
    records = _load_shards(shards, indexed)
    missing = [row for row in indexed if row["_index"] not in records]
    if missing:
        jobs = [(row["_index"], {"example_id": row["example_id"], "db_id": row["db_id"],
                                 "question": row["question"]}, schemas[row["db_id"]])
                for row in missing]
        pool = GenerationPool(cfg, devices)
        completed = len(records)
        try:
            def save(index, record):
                nonlocal completed
                record["dataset_index"] = index
                _atomic_json(shards / f"{index:06d}.json", record)
                records[index] = record
                completed += 1
                print(f"{completed}/{len(indexed)} generated", flush=True)
            pool.run(jobs, save)
        finally:
            pool.close()
    if set(records) != {row["_index"] for row in indexed}:
        raise RuntimeError("Generation shards are incomplete")
    ordered = [records[row["_index"]] for row in indexed]
    _score_records(cfg, indexed, records)
    return _write_reports(cfg, output, indexed, ordered, devices, limit, started)


def rescore(cfg, output_dir, test_suite_database_dir):
    """Recompute all metrics from saved generation shards without model inference."""
    started = time.monotonic()
    output = Path(output_dir)
    manifest_path = output / "run_manifest.json"
    shards = output / "generation_shards"
    if not manifest_path.is_file() or not shards.is_dir():
        raise FileNotFoundError(
            f"Existing run must contain run_manifest.json and generation_shards: {output}")

    manifest = json.loads(manifest_path.read_text())
    current_config = asdict(cfg)
    original_config = manifest.get("config", {})
    for section in ("model", "data", "generation", "sql"):
        if original_config.get(section) != current_config.get(section):
            raise ValueError(
                f"Existing run used a different {section} configuration")
    if manifest.get("test_json_sha256") != _hash_file(cfg.data.test_json):
        raise ValueError("Existing run used a different test JSON")
    if manifest.get("test_tables_sha256") != _hash_file(cfg.data.test_tables):
        raise ValueError("Existing run used different test tables")

    all_rows = read_json(cfg.data.test_json)
    selected = manifest.get("selected")
    if not isinstance(selected, list) or any(
            not isinstance(index, int) or index < 0 or index >= len(all_rows)
            for index in selected):
        raise ValueError("Existing run manifest has an invalid selected index list")
    indexed = [dict(all_rows[index], _index=index, example_id=f"test:{index}")
               for index in selected]
    records = _load_shards(shards, indexed)
    if set(records) != set(selected):
        raise RuntimeError("Generation shards are incomplete; cannot rescore")

    score_cfg = deepcopy(cfg)
    score_cfg.evaluation.test_suite_database_dir = str(test_suite_database_dir)
    ordered = [records[index] for index in selected]
    _score_records(score_cfg, indexed, records)
    devices = manifest.get("devices", [])
    limit = len(selected) if len(selected) != len(all_rows) else None
    summary = _write_reports(score_cfg, output, indexed, ordered, devices, limit, started)
    score_cfg.save(output / "rescore_config.json")
    return summary
