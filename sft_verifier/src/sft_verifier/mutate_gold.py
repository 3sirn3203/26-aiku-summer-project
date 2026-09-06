from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from text2sql.core.sql_output import extract_sql

from .collect_candidates import _execute
from .common import DEFAULT_SPIDER_ROOT, read_jsonl, write_jsonl


Mutation = Tuple[str, str, str]


def generate_mutations(sql: str) -> Sequence[Mutation]:
    """Return conservative, single-edit SQL mutations with known feedback."""

    mutations: List[Mutation] = []

    def add(category: str, candidate: str, feedback: str) -> None:
        candidate = candidate.strip().rstrip(";").strip()
        if candidate and candidate.casefold() != sql.strip().rstrip(";").casefold():
            mutations.append((category, candidate, feedback))

    if re.search(r"\bDISTINCT\b", sql, re.IGNORECASE):
        add(
            "missing_distinct",
            re.sub(r"\bDISTINCT\s+", "", sql, count=1, flags=re.IGNORECASE),
            "The candidate does not remove duplicate result rows. Preserve the question's uniqueness requirement when selecting the result.",
        )

    order_match = re.search(r"\b(ASC|DESC)\b", sql, re.IGNORECASE)
    if order_match:
        replacement = "DESC" if order_match.group(1).upper() == "ASC" else "ASC"
        add(
            "wrong_order_direction",
            sql[: order_match.start()] + replacement + sql[order_match.end() :],
            "The candidate orders the results in the opposite direction from the question. Reverse the ordering direction.",
        )

    limit_match = re.search(r"\s+LIMIT\s+\d+\s*;?\s*$", sql, re.IGNORECASE)
    if limit_match:
        add(
            "missing_limit",
            sql[: limit_match.start()],
            "The candidate does not restrict the result to the requested number of rows. Restore the requested result limit after ordering.",
        )

    where_pattern = re.compile(
        r"\s+WHERE\s+.*?(?=\s+(?:GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|UNION|INTERSECT|EXCEPT)\b|\s*$)",
        re.IGNORECASE | re.DOTALL,
    )
    if where_pattern.search(sql):
        add(
            "missing_filter",
            where_pattern.sub("", sql, count=1),
            "The candidate omits a condition required by the question. Restore the missing filter using the relevant field and requested value or comparison.",
        )

    aggregate_match = re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", sql, re.IGNORECASE)
    if aggregate_match:
        original = aggregate_match.group(1).upper()
        replacement = "SUM" if original == "COUNT" else "COUNT"
        add(
            "wrong_aggregation",
            sql[: aggregate_match.start(1)] + replacement + sql[aggregate_match.end(1) :],
            "The candidate uses the wrong aggregation for the requested quantity. Use the aggregation implied by the question.",
        )

    comparator_match = re.search(r"(?<![<>!])(?:>=|<=|<>|!=|=|>|<)(?!=)", sql)
    if comparator_match:
        original = comparator_match.group(0)
        replacement = {"=": "!=", "!=": "=", "<>": "=", ">": "<", "<": ">", ">=": "<", "<=": ">"}[original]
        add(
            "wrong_comparison",
            sql[: comparator_match.start()] + replacement + sql[comparator_match.end() :],
            "The candidate applies the wrong comparison in a required condition. Correct the comparison so it matches the question.",
        )

    deduplicated: Dict[str, Mutation] = {}
    for mutation in mutations:
        key = " ".join(mutation[1].split()).casefold()
        deduplicated.setdefault(key, mutation)
    return tuple(deduplicated.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Create single-edit gold SQL negatives")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--spider-root", type=Path, default=DEFAULT_SPIDER_ROOT)
    parser.add_argument("--max-per-question", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_per_question < 1:
        parser.error("--max-per-question must be positive")

    gold_records = [item for item in read_jsonl(args.candidates) if item.get("source") == "gold"]
    output = []
    for gold in gold_records:
        for position, (category, sql, feedback) in enumerate(
            generate_mutations(str(gold["gold_sql"]))[: args.max_per_question]
        ):
            parsing = extract_sql(sql)
            output.append(
                {
                    **gold,
                    "example_id": "%s:mutation:%s:%d"
                    % (gold["base_example_id"], category, position),
                    "source": "gold_mutation",
                    "candidate_raw_output": sql,
                    "candidate_sql": parsing.sql,
                    "sql_parsing": parsing.to_dict(),
                    "execution_observation": _execute(
                        args.spider_root.resolve(), str(gold["db_id"]), parsing.sql
                    ),
                    "mutation": {
                        "category": category,
                        "feedback": feedback,
                    },
                }
            )
    write_jsonl(args.output, output)
    print("wrote %d mutations to %s" % (len(output), args.output))


if __name__ == "__main__":
    main()
