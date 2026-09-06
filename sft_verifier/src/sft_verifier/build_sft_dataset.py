from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from text2sql.multi_turn_agent.contracts import PlannerOutput
from text2sql.multi_turn_agent.prompts import build_verifier_messages

from .common import dataset_summary, read_jsonl, write_json, write_jsonl


def feedback_for(record: Mapping[str, Any]) -> Optional[str]:
    decision = record.get("decision_label")
    if decision == "stop":
        return (
            "The candidate satisfies the question's requested result, conditions, "
            "and aggregation or ordering requirements, so no correction is needed."
        )
    parsing = record.get("sql_parsing")
    if isinstance(parsing, Mapping) and parsing.get("status") != "success":
        return (
            "The coder did not produce one complete SQL query. Produce one complete "
            "read-only query that answers the question using the supplied schema."
        )
    mutation = record.get("mutation")
    if isinstance(mutation, Mapping):
        value = mutation.get("feedback")
        if isinstance(value, str) and value.strip():
            return value.strip()
    observation = record.get("execution_observation")
    if isinstance(observation, Mapping) and observation.get("status") != "success":
        message = observation.get("error_message")
        detail = (" The database reported: %s" % message) if message else ""
        return (
            "The candidate cannot be executed against the supplied schema.%s Revise "
            "the referenced tables, columns, joins, or SQL structure before retrying."
            % detail
        )
    # A test-suite mismatch confirms that the query is wrong, but does not identify
    # an actionable correction. Keep these records for offline decision evaluation;
    # do not train on a guessed explanation in the high-confidence first milestone.
    return None


def _stable_rank(record: Mapping[str, Any], seed: int) -> str:
    value = "%d\0%s" % (seed, record["example_id"])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_examples(
    records: Sequence[Mapping[str, Any]], *, seed: int, balance: bool
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    seen = set()
    for record in records:
        if record.get("label_status") != "accepted":
            continue
        decision = record.get("decision_label")
        if decision not in ("stop", "continue"):
            continue
        if record.get("source") == "gold_mutation" and decision != "continue":
            continue
        feedback = feedback_for(record)
        if feedback is None:
            continue
        planner = record.get("planner_output")
        if not isinstance(planner, Mapping) or not isinstance(planner.get("plan"), str):
            continue
        parsing = record.get("sql_parsing")
        observation = record.get("execution_observation")
        if not isinstance(parsing, Mapping) or not isinstance(observation, Mapping):
            continue
        messages = build_verifier_messages(
            str(record["question"]),
            str(record["serialized_schema"]),
            1,
            PlannerOutput(plan=planner["plan"]),
            "success",
            str(record.get("candidate_raw_output", "")),
            parsing,
            observation,
        )
        target = json.dumps(
            {"feedback": feedback, "decision": decision},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        dedup_key = hashlib.sha256(
            json.dumps([messages, target], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        mutation = record.get("mutation")
        category = (
            str(mutation.get("category"))
            if isinstance(mutation, Mapping)
            else (
                "correct"
                if decision == "stop"
                else str(observation.get("error_type") or parsing.get("error_type") or "semantic")
            )
        )
        examples.append(
            {
                "schema_version": 1,
                "example_id": record["example_id"],
                "base_example_id": record["base_example_id"],
                "db_id": record["db_id"],
                "split": record["split"],
                "source": record["source"],
                "error_category": category,
                "messages": messages,
                "response": target,
                "label": {"decision": decision, "feedback": feedback},
                "label_evidence": {
                    "test_suite": record.get("official_evaluation", {}).get("test_suite"),
                    "exact_set_match": record.get("official_evaluation", {}).get("exact_set_match"),
                },
            }
        )

    if not balance:
        return sorted(examples, key=lambda item: (item["split"], _stable_rank(item, seed)))
    balanced: List[Dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        stop = [item for item in examples if item["split"] == split and item["label"]["decision"] == "stop"]
        cont = [item for item in examples if item["split"] == split and item["label"]["decision"] == "continue"]
        keep = min(len(stop), len(cont))
        stop.sort(key=lambda item: _stable_rank(item, seed))
        cont.sort(key=lambda item: _stable_rank(item, seed))
        balanced.extend(stop[:keep])
        balanced.extend(cont[:keep])
    return sorted(balanced, key=lambda item: (item["split"], _stable_rank(item, seed)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a high-confidence verifier SFT dataset")
    parser.add_argument("--labeled", type=Path, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records: List[Dict[str, Any]] = []
    for path in args.labeled:
        records.extend(read_jsonl(path))
    examples = build_examples(records, seed=args.seed, balance=not args.no_balance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        write_jsonl(
            args.output_dir / (split + ".jsonl"),
            [item for item in examples if item["split"] == split],
        )
    summary = dataset_summary(examples)
    summary.update(
        {
            "seed": args.seed,
            "balanced": not args.no_balance,
            "input_records": len(records),
            "excluded_from_sft": len(records) - len(examples),
            "separate_gold_reference_field_in_messages": False,
            "prompt_builder": "text2sql.multi_turn_agent.prompts.build_verifier_messages",
        }
    )
    write_json(args.output_dir / "dataset_manifest.json", summary)
    print("wrote %d SFT examples to %s" % (len(examples), args.output_dir))


if __name__ == "__main__":
    main()
