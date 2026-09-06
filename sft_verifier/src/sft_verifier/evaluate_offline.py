from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from text2sql.multi_turn_agent.contracts import ContractError, parse_verifier_output

from . import BASE_MODEL_ID, BASE_MODEL_REVISION
from .common import read_jsonl, write_json, write_jsonl
from .generation import ModelGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a verifier on fixed labeled candidates")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--base-revision", default=BASE_MODEL_REVISION)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = read_jsonl(args.dataset)
    if args.max_examples is not None:
        records = records[: args.max_examples]
    generator = ModelGenerator(
        args.base_model,
        revision=args.base_revision,
        device=args.device,
        adapter=args.adapter,
        allow_download=args.allow_model_download,
    )
    predictions: List[Dict[str, Any]] = []
    confusion = {"stop_stop": 0, "stop_continue": 0, "continue_stop": 0, "continue_continue": 0}
    category_counts: Dict[str, Dict[str, int]] = {}
    valid = 0
    contradictions = 0
    for position, record in enumerate(records):
        generation = generator.generate(
            record["messages"], max_new_tokens=384, seed=20260905 + position
        )
        try:
            parsed = parse_verifier_output(generation["raw_output"])
            predicted = parsed.decision
            feedback = parsed.feedback
            contract_error = None
            valid += 1
            lower = feedback.casefold()
            if predicted == "stop" and any(
                marker in lower for marker in ("must correct", "needs correction", "incorrect", "cannot be executed")
            ):
                contradictions += 1
            if predicted == "continue" and any(
                marker in lower for marker in ("no correction", "is correct", "satisfies the question")
            ):
                contradictions += 1
        except ContractError as exc:
            predicted = None
            feedback = None
            contract_error = str(exc)
        gold = str(record["label"]["decision"])
        if predicted in ("stop", "continue"):
            confusion[gold + "_" + predicted] += 1
        category = str(record.get("error_category") or "none")
        category_metric = category_counts.setdefault(
            category, {"examples": 0, "contract_valid": 0, "correct": 0}
        )
        category_metric["examples"] += 1
        category_metric["contract_valid"] += int(predicted is not None)
        category_metric["correct"] += int(predicted == gold)
        predictions.append(
            {
                "example_id": record["example_id"],
                "db_id": record["db_id"],
                "gold_decision": gold,
                "predicted_decision": predicted,
                "feedback": feedback,
                "contract_error": contract_error,
                "generation": generation,
                "error_category": record.get("error_category"),
            }
        )
    total = len(records)
    stop_total = confusion["stop_stop"] + confusion["stop_continue"]
    continue_total = confusion["continue_stop"] + confusion["continue_continue"]
    stop_recall = confusion["stop_stop"] / stop_total if stop_total else None
    continue_recall = confusion["continue_continue"] / continue_total if continue_total else None
    predicted_stop = confusion["stop_stop"] + confusion["continue_stop"]
    predicted_continue = confusion["stop_continue"] + confusion["continue_continue"]
    stop_precision = confusion["stop_stop"] / predicted_stop if predicted_stop else None
    continue_precision = (
        confusion["continue_continue"] / predicted_continue if predicted_continue else None
    )

    def f1(precision: float | None, recall: float | None) -> float | None:
        if precision is None or recall is None:
            return None
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    stop_f1 = f1(stop_precision, stop_recall)
    continue_f1 = f1(continue_precision, continue_recall)
    balanced = (
        (stop_recall + continue_recall) / 2
        if stop_recall is not None and continue_recall is not None
        else None
    )
    metrics = {
        "examples": total,
        "contract_valid": valid,
        "contract_valid_rate": valid / total if total else None,
        "confusion_gold_predicted": confusion,
        "stop_precision": stop_precision,
        "stop_recall": stop_recall,
        "stop_f1": stop_f1,
        "continue_precision": continue_precision,
        "continue_recall": continue_recall,
        "continue_f1": continue_f1,
        "macro_f1": (
            (stop_f1 + continue_f1) / 2
            if stop_f1 is not None and continue_f1 is not None
            else None
        ),
        "balanced_accuracy": balanced,
        "false_stop_rate": confusion["continue_stop"] / continue_total if continue_total else None,
        "false_continue_rate": confusion["stop_continue"] / stop_total if stop_total else None,
        "decision_feedback_contradictions": contradictions,
        "decision_feedback_contradiction_rate": contradictions / valid if valid else None,
        "by_error_category": {
            category: {
                **values,
                "accuracy": values["correct"] / values["examples"],
                "contract_valid_rate": values["contract_valid"] / values["examples"],
            }
            for category, values in sorted(category_counts.items())
        },
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_json(args.output_dir / "metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
