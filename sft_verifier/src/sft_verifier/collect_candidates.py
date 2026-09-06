from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from text2sql.core.executor import execute_sql
from text2sql.core.sql_output import extract_sql
from text2sql.multi_turn_agent.contracts import ContractError, parse_planner_output
from text2sql.multi_turn_agent.observation import bound_execution_observation
from text2sql.multi_turn_agent.prompts import build_coder_messages, build_planner_messages

from .common import (
    DEFAULT_SPIDER_ROOT,
    append_jsonl,
    load_schemas,
    read_json,
    read_jsonl,
    select_examples,
    write_json,
)
from .generation import ModelGenerator


def _execute(spider_root: Path, db_id: str, sql: Optional[str]) -> Dict[str, Any]:
    if not sql:
        return {
            "status": "not_executed",
            "error_type": "sql_parse_error",
            "error_message": "No parsed SQL was available",
            "columns": [],
            "row_count": None,
            "rows": [],
            "truncated": False,
        }
    result = execute_sql(
        spider_root / "database" / db_id / (db_id + ".sqlite"),
        sql,
        timeout_seconds=5.0,
        max_sql_bytes=100000,
        max_result_rows=10000,
        max_result_bytes=2000000,
        worker_memory_limit_bytes=1024 * 1024 * 1024,
    )
    return bound_execution_observation(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect real planner/coder candidates")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--spider-root", type=Path, default=DEFAULT_SPIDER_ROOT)
    parser.add_argument("--planner-model", required=True)
    parser.add_argument("--planner-revision")
    parser.add_argument("--planner-device", default="cuda:0")
    parser.add_argument("--coder-model", required=True)
    parser.add_argument("--coder-revision")
    parser.add_argument("--coder-device", default="cuda:1")
    parser.add_argument("--samples-per-question", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    if args.samples_per_question < 1:
        parser.error("--samples-per-question must be positive")
    if args.output.exists() and not args.resume:
        parser.error("output exists; use --resume or choose a new path")

    spider_root = args.spider_root.resolve()
    split_manifest = read_json(args.split_manifest)
    schemas = load_schemas(spider_root / "tables.json")
    completed = set()
    if args.resume and args.output.exists():
        completed = {str(item["example_id"]) for item in read_jsonl(args.output)}
    progress_output = args.progress_output or args.output.with_suffix(".progress.jsonl")
    completed_examples = set()
    if args.resume and progress_output.exists():
        completed_examples = {
            str(item["base_example_id"]) for item in read_jsonl(progress_output)
        }

    planner = ModelGenerator(
        args.planner_model,
        revision=args.planner_revision,
        device=args.planner_device,
        allow_download=args.allow_model_download,
    )
    coder = ModelGenerator(
        args.coder_model,
        revision=args.coder_revision,
        device=args.coder_device,
        allow_download=args.allow_model_download,
    )
    written = 0
    failures = 0
    for example in select_examples(split_manifest, args.split, args.limit):
        base_id = str(example["example_id"])
        if base_id in completed_examples:
            continue
        schema = schemas[str(example["db_id"])]
        planner_messages = build_planner_messages(
            str(example["question"]), schema, 1, []
        )
        planner_generation = planner.generate(
            planner_messages, max_new_tokens=512, seed=args.seed + int(example["index"])
        )
        try:
            plan = parse_planner_output(planner_generation["raw_output"])
            planner_error = None
        except ContractError as exc:
            plan = None
            planner_error = str(exc)
            failures += 1
        if plan is None:
            failure_id = base_id + ":planner_error"
            if failure_id not in completed:
                append_jsonl(
                    args.output,
                    {
                        "schema_version": 1,
                        "example_id": failure_id,
                        "base_example_id": base_id,
                        "split": args.split,
                        "db_id": example["db_id"],
                        "question": example["question"],
                        "gold_sql": example["gold_sql"],
                        "serialized_schema": schema,
                        "source": "planner_error",
                        "planner_generation": planner_generation,
                        "planner_contract_error": planner_error,
                    },
                )
                written += 1
            append_jsonl(progress_output, {"base_example_id": base_id, "status": "planner_error"})
            continue

        common: Dict[str, Any] = {
            "schema_version": 1,
            "base_example_id": base_id,
            "split": args.split,
            "db_id": example["db_id"],
            "question": example["question"],
            "gold_sql": example["gold_sql"],
            "serialized_schema": schema,
            "planner_output": plan.to_dict(),
            "planner_generation": planner_generation,
        }
        gold_id = base_id + ":gold"
        if gold_id not in completed:
            gold_parse = extract_sql(str(example["gold_sql"]))
            append_jsonl(
                args.output,
                {
                    **common,
                    "example_id": gold_id,
                    "source": "gold",
                    "candidate_raw_output": example["gold_sql"],
                    "candidate_sql": gold_parse.sql,
                    "sql_parsing": gold_parse.to_dict(),
                    "execution_observation": _execute(
                        spider_root, str(example["db_id"]), gold_parse.sql
                    ),
                },
            )
            written += 1
        coder_messages = build_coder_messages(
            str(example["question"]), schema, 1, plan
        )
        seen_sql = set()
        for sample_index in range(args.samples_per_question):
            candidate_id = "%s:coder:%d" % (base_id, sample_index)
            if candidate_id in completed:
                continue
            generation = coder.generate(
                coder_messages,
                max_new_tokens=512,
                do_sample=sample_index > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + int(example["index"]) * 100 + sample_index,
            )
            parsing = extract_sql(generation["raw_output"])
            normalized = " ".join((parsing.sql or generation["raw_output"]).split()).casefold()
            if normalized in seen_sql:
                continue
            seen_sql.add(normalized)
            append_jsonl(
                args.output,
                {
                    **common,
                    "example_id": candidate_id,
                    "source": "coder_greedy" if sample_index == 0 else "coder_sample",
                    "candidate_raw_output": generation["raw_output"],
                    "candidate_sql": parsing.sql,
                    "sql_parsing": parsing.to_dict(),
                    "execution_observation": _execute(
                        spider_root, str(example["db_id"]), parsing.sql
                    ),
                    "coder_generation": generation,
                },
            )
            written += 1
        append_jsonl(progress_output, {"base_example_id": base_id, "status": "complete"})

    manifest_output = args.manifest_output or args.output.with_suffix(".manifest.json")
    write_json(
        manifest_output,
        {
            "schema_version": 1,
            "split": args.split,
            "output": str(args.output.resolve()),
            "records_written_this_invocation": written,
            "progress_output": str(progress_output.resolve()),
            "planner_failures_this_invocation": failures,
            "planner_model": args.planner_model,
            "planner_revision": args.planner_revision,
            "coder_model": args.coder_model,
            "coder_revision": args.coder_revision,
            "samples_per_question": args.samples_per_question,
            "seed": args.seed,
        },
    )
    print("wrote %d candidate records to %s" % (written, args.output))


if __name__ == "__main__":
    main()
