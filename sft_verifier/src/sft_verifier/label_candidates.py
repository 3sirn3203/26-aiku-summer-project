from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from text2sql.core.official_eval import (
    OfficialEvaluationItem,
    evaluate_official,
    is_official_infrastructure_failure,
    validate_official_environment,
)

from .common import (
    DEFAULT_EVALUATOR_ROOT,
    DEFAULT_NLTK_DATA,
    DEFAULT_SPIDER_ROOT,
    DEFAULT_TEST_SUITE_ROOT,
    UPSTREAM_COMMIT,
    read_jsonl,
    write_json,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label candidate SQL with Spider test-suite")
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--spider-root", type=Path, default=DEFAULT_SPIDER_ROOT)
    parser.add_argument("--test-suite-root", type=Path, default=DEFAULT_TEST_SUITE_ROOT)
    parser.add_argument("--evaluator-root", type=Path, default=DEFAULT_EVALUATOR_ROOT)
    parser.add_argument("--nltk-data", type=Path, default=DEFAULT_NLTK_DATA)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    records: List[Dict[str, Any]] = []
    for path in args.candidates:
        records.extend(read_jsonl(path))
    records = [item for item in records if item.get("source") != "planner_error"]
    if args.limit is not None:
        records = records[: args.limit]
    db_ids = sorted({str(item["db_id"]) for item in records})
    setup = validate_official_environment(
        evaluator_root=args.evaluator_root.resolve(),
        database_root=args.test_suite_root.resolve(),
        tables_path=(args.spider_root / "tables.json").resolve(),
        expected_commit=UPSTREAM_COMMIT,
        nltk_data_dir=args.nltk_data.resolve(),
        db_ids=db_ids,
    )
    if not setup["ok"]:
        raise RuntimeError("official evaluator preflight failed: %s" % setup["errors"])

    scored: Dict[str, Dict[str, Any]] = {}
    eligible = [item for item in records if item.get("candidate_sql")]
    for start in range(0, len(eligible), args.batch_size):
        batch = eligible[start : start + args.batch_size]
        result = evaluate_official(
            [
                OfficialEvaluationItem(
                    example_id=str(item["example_id"]),
                    db_id=str(item["db_id"]),
                    gold_sql=str(item["gold_sql"]),
                    predicted_sql=str(item["candidate_sql"]),
                )
                for item in batch
            ],
            evaluator_root=args.evaluator_root.resolve(),
            database_root=args.test_suite_root.resolve(),
            tables_path=(args.spider_root / "tables.json").resolve(),
            expected_commit=UPSTREAM_COMMIT,
            nltk_data_dir=args.nltk_data.resolve(),
            timeout_seconds=args.timeout_seconds,
            preflight_report=setup,
        )
        for item in result["results"]:
            scored[str(item["example_id"])] = item

    output = []
    excluded = 0
    for record in records:
        official = scored.get(str(record["example_id"]))
        if official is None:
            decision = "continue"
            label_status = "accepted"
            evidence = {"test_suite": {"status": "prediction_unavailable", "match": False}}
        else:
            test_suite = official["test_suite"]
            status = test_suite["status"]
            infrastructure_failure = is_official_infrastructure_failure("test_suite", status)
            label_status = "excluded" if infrastructure_failure else "accepted"
            decision = "stop" if test_suite.get("match") is True else "continue"
            evidence = official
        if label_status == "excluded":
            excluded += 1
        output.append(
            {
                **record,
                "decision_label": decision,
                "label_status": label_status,
                "official_evaluation": evidence,
            }
        )
    write_jsonl(args.output, output)
    write_json(
        args.output.with_suffix(".manifest.json"),
        {
            "schema_version": 1,
            "records": len(output),
            "accepted": len(output) - excluded,
            "excluded": excluded,
            "database_count": len(db_ids),
            "preflight": setup,
        },
    )
    print("wrote %d labeled records (%d excluded)" % (len(output), excluded))


if __name__ == "__main__":
    main()
