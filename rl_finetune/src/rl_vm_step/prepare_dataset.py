from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from text2sql.config import ConfigError, load_config
from rl_finetune.dataset import read_indices_json, write_jsonl
from rl_vm_step.dataset import build_vm_step_records, gold_reference_rows
from rl_vm_step.prepared_dataset import build_prepared_manifest, write_json
from rl_vm_step.train_grpo_lora import _validate_base_provenance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rl_vm_step.prepare_dataset",
        description="Prepare immutable Spider gold VM-step references",
    )
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--examples-file", default="train_spider.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--indices-file", type=Path)
    parser.add_argument(
        "--fail-on-rejected-gold-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise ConfigError("--limit must be positive")
    if args.offset < 0 or args.progress_every < 1:
        raise ConfigError("offset must be non-negative and progress interval positive")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ConfigError("prepared dataset directory already exists: %s" % output_dir)
    baseline = load_config(args.baseline_config)
    config = replace(
        baseline,
        spider=replace(
            baseline.spider, examples_file=args.examples_file, split=args.split
        ),
    )
    _validate_base_provenance(config)
    indices = read_indices_json(args.indices_file) if args.indices_file else None

    def progress(position: int, total: int) -> None:
        if position == 1 or position == total or position % args.progress_every == 0:
            print(
                "gold VM reference: %d/%d" % (position, total),
                file=sys.stderr,
                flush=True,
            )

    records, rejected = build_vm_step_records(
        config,
        limit=args.limit,
        offset=args.offset,
        indices=indices,
        progress_callback=progress,
    )
    train_path = output_dir / "train_dataset.jsonl"
    gold_path = output_dir / "gold_reference.jsonl"
    rejected_path = output_dir / "rejected_gold_reference.jsonl"
    write_jsonl(train_path, records)
    write_jsonl(gold_path, gold_reference_rows(records))
    write_jsonl(rejected_path, rejected)
    manifest = build_prepared_manifest(
        config=config,
        records=records,
        rejected=rejected,
        train_dataset_path=train_path,
        gold_reference_path=gold_path,
        rejected_reference_path=rejected_path,
        selection={
            "examples_file": args.examples_file,
            "split": args.split,
            "limit": args.limit,
            "offset": args.offset,
            "indices_file": str(args.indices_file.resolve())
            if args.indices_file
            else None,
        },
    )
    if not records:
        manifest["status"] = "invalid"
        manifest["error"] = "no eligible records"
    elif rejected and args.fail_on_rejected_gold_reference:
        manifest["status"] = "invalid"
        manifest["error"] = "%d gold references were rejected" % len(rejected)
    write_json(output_dir / "dataset_manifest.json", manifest)
    if manifest["status"] != "ready":
        raise ConfigError(str(manifest["error"]))
    print(
        json.dumps(
            {
                "prepared_dataset": str(output_dir),
                "record_count": len(records),
                "rejected_count": len(rejected),
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConfigError, RuntimeError, ValueError, OSError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
