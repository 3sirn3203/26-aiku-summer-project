import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .data import load_split, read_json, schema_map, split_config
from .model import HFPolicy, load_model
from .rollout import rollout
from .sql import Executor, Judge
from .train import append_json, seed_all, summarize


def evaluate(cfg, output_dir, checkpoint=None, data_file=None, limit=None, split="dev", devices=None):
    cfg = split_config(cfg, split)
    seed_all(cfg.train.seed)
    pool = None
    if devices:
        from .runtime.evaluation_pool import EvaluationPool
        policy = None
    else:
        model, tokenizer = load_model(cfg.model, trainable=False, checkpoint=checkpoint)
        policy = HFPolicy(model, tokenizer, cfg.model, cfg.rollout)
    from .tracking import ExperimentTracker
    tracker = ExperimentTracker()
    state = {"step": 0, "groups": 0}
    if checkpoint and (Path(checkpoint) / "complete.json").exists():
        state.update(read_json(Path(checkpoint) / "complete.json"))
    try:
        if devices:
            pool = EvaluationPool(cfg, devices, checkpoint, state["step"])
        summary = evaluate_policy(cfg, output_dir, policy=policy, data_file=data_file, limit=limit,
                                  split=split, policy_version=state["step"], pool=pool)
        tracker.start(cfg, output_dir, state, job_type=f"{split}-evaluation", checkpoint=checkpoint)
        tracker.log(split, summary, state)
        tracker.summary({f"{split}/{k}": v for k, v in summary.items()})
        tracker.evaluation_files(output_dir, split)
        return summary
    finally:
        if pool is not None:
            pool.close()
        tracker.finish(failed=sys.exc_info()[0] is not None)


def evaluate_policy(cfg, output_dir, policy=None, pool=None, policy_version=0,
                    data_file=None, limit=None, split="dev"):
    started = time.monotonic()
    rows = read_json(data_file) if data_file else load_split(cfg.data, split)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        # Stable shuffled subset avoids selecting only the first database.
        import random
        rows = random.Random(cfg.train.seed).sample(rows, min(limit, len(rows)))
    if not rows:
        raise ValueError("Empty evaluation data")
    if cfg.sql.evaluator_path:
        preflight = {"evaluator_path": cfg.sql.evaluator_path, "tables": cfg.data.tables,
                     "nltk_data": cfg.sql.nltk_data, "database_dir": cfg.data.database_dir,
                     "rows": [{**rows[0], "prediction": rows[0]["query"]}]}
        subprocess.run([sys.executable, "-m", "rrcm_sql.spider_metrics"], input=json.dumps(preflight),
                       text=True, capture_output=True, check=True, timeout=60)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    cfg.save(output / "config.json")
    schemas = schema_map(cfg.data.tables)
    executor = Executor(cfg.sql)
    judge = Judge(executor, cfg.data.database_dir, cfg.data.tables)
    trajectories = []
    worker_rows = [dict(row, example_id=str(row.get("example_id", index)))
                   for index, row in enumerate(rows)]
    generated = pool.evaluate(worker_rows, schemas, policy_version, cfg) if pool else None
    for index, row in enumerate(rows):
        if time.monotonic() - started > cfg.evaluation.timeout_seconds:
            raise TimeoutError("Evaluation exceeded timeout_seconds")
        row = worker_rows[index]
        t = generated[index] if generated is not None else rollout(
            policy, row, schemas[row["db_id"]], executor, judge, cfg.rollout,
            mode="free", sample=False, evaluate_suite=bool(cfg.sql.suite_database_dir),
            policy_version=policy_version)
        trajectories.append(t)
        append_json(output / "trajectories.jsonl", t.record())
        print(f"{index + 1}/{len(rows)} {t.outcome}", flush=True)
    summary = summarize(trajectories)
    summary["split"] = split
    summary["policy_version"] = policy_version
    summary["data_sha256"] = hashlib.sha256(
        json.dumps({"rows": rows, "schemas": schemas}, sort_keys=True).encode()).hexdigest()
    summary["is_subset"] = limit is not None
    summary["selection_metric"] = cfg.evaluation.selection_metric
    summary["evaluation_devices"] = list(pool.devices) if pool else [cfg.model.device]
    summary["execution_accuracy"] = sum(t.scores.get("execution_correct", False) for t in trajectories) / len(rows)
    summary["test_suite_accuracy"] = (sum(bool(t.scores.get("test_suite_correct")) for t in trajectories) / len(rows)
                                      if cfg.sql.suite_database_dir else None)
    summary["correctness_backend"] = "spider_result_eq" if cfg.sql.evaluator_path else "strict_execution_proxy"
    summary["max_intermediate_reached_rate"] = sum(len(t.intermediate) == cfg.rollout.max_intermediate for t in trajectories) / len(rows)
    # Official structural exact match + hardness in an isolated process (no SQL prediction execution).
    if cfg.sql.evaluator_path:
        payload = {"evaluator_path": cfg.sql.evaluator_path, "tables": cfg.data.tables,
                   "nltk_data": cfg.sql.nltk_data,
                   "database_dir": cfg.data.database_dir,
                   "rows": [{**row, "prediction": t.final_sql or ""} for row, t in zip(rows, trajectories)]}
        result = subprocess.run([sys.executable, "-m", "rrcm_sql.spider_metrics"],
                                input=json.dumps(payload), text=True, capture_output=True, check=True, timeout=300)
        structural = json.loads(result.stdout)
        summary["exact_match"] = sum(x["exact_match"] for x in structural) / len(rows)
        buckets = {}
        for t, entry in zip(trajectories, structural):
            buckets.setdefault(entry["difficulty"], []).append(t)
        per_difficulty = []
        for key, group in sorted(buckets.items()):
            scores = [entry for entry in structural if entry["difficulty"] == key]
            per_difficulty.append({"difficulty": key, **summarize(group),
                                   "exact_match": sum(entry["exact_match"] for entry in scores) / len(scores),
                                   "test_suite_accuracy": (sum(bool(t.scores.get("test_suite_correct")) for t in group) / len(group)
                                                           if cfg.sql.suite_database_dir else None)})
        with (output / "difficulty.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_difficulty[0]))
            writer.writeheader()
            writer.writerows(per_difficulty)
        (output / "structural_metrics.json").write_text(json.dumps(structural, indent=2) + "\n")
    else:
        summary["exact_match"] = None
    summary["seconds"] = time.monotonic() - started
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    (output / "predictions.sql").write_text("\n".join(" ".join((t.final_sql or "SELECT").split()) for t in trajectories) + "\n")
    (output / "gold.sql").write_text("\n".join(" ".join(r["query"].split()) + "\t" + r["db_id"] for r in rows) + "\n")
    return summary
