from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from text2sql.config import ConfigError, load_config
from text2sql.core.progress import ProgressReporter
from text2sql.core.spider import SpiderDataError
from text2sql.single_turn.distributed import parse_gpu_ids
from text2sql.single_turn.evaluation_runner import run_full_evaluation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rl_finetune.evaluate_two_turn_base",
        description=(
            "Distributed draft-execute-observe-final evaluation for the frozen "
            "base model or one PEFT adapter"
        ),
    )
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    naming = parser.add_mutually_exclusive_group(required=True)
    naming.add_argument("--run-name")
    naming.add_argument("--resume-run")
    parser.add_argument("--selection", choices=("smoke", "all"), default="smoke")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--max-draft-tokens", type=int, default=256)
    parser.add_argument("--max-final-tokens", type=int, default=256)
    parser.add_argument("--max-observation-rows", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)
    return parser


def _source_tree_sha256() -> str:
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    digest.update(b"rl-finetune-source-v1\0")
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _workflow_contract(args: argparse.Namespace) -> dict:
    if args.max_draft_tokens < 1 or args.max_final_tokens < 1:
        raise ConfigError("two-turn token limits must be positive")
    if args.max_observation_rows < 1:
        raise ConfigError("--max-observation-rows must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ConfigError("--temperature must be positive and finite")
    if args.seed < 0:
        raise ConfigError("--seed must be non-negative")
    return {
        "schema_version": 1,
        "turns": 2,
        "max_draft_tokens": args.max_draft_tokens,
        "max_final_tokens": args.max_final_tokens,
        "max_observation_rows": args.max_observation_rows,
        "temperature": args.temperature,
        "seed": args.seed,
        "seed_policy": "base_seed_plus_spider_index",
        "final_sql_is_scored": True,
        "gold_available_to_generation": False,
        "rl_source_tree_sha256": _source_tree_sha256(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if (
            not math.isfinite(args.progress_interval_seconds)
            or args.progress_interval_seconds <= 0
        ):
            raise ConfigError("--progress-interval-seconds must be positive and finite")
        config = load_config(args.baseline_config)
        config = replace(
            config,
            model=replace(config.model, device="cuda:0"),
            output=replace(config.output, directory=args.output_dir.resolve()),
        )
        gpu_ids = parse_gpu_ids(args.gpus)
        workflow = _workflow_contract(args)
        progress = ProgressReporter(
            enabled=not args.no_progress,
            interval_seconds=args.progress_interval_seconds,
            stream=sys.stderr,
        )
        try:
            result = run_full_evaluation(
                config,
                backend_name="two_turn",
                gpu_ids=gpu_ids,
                allow_model_download=args.allow_model_download,
                run_name=args.run_name,
                resume_run=args.resume_run,
                selection=args.selection,
                progress=progress,
                adapter_dir=(
                    args.adapter_dir.resolve() if args.adapter_dir is not None else None
                ),
                workflow_contract=workflow,
                invocation={
                    "interface": "rl_finetune_module",
                    "command": "evaluate_two_turn_base",
                    "config": str(config.source_path),
                    "selection": args.selection,
                    "physical_gpu_ids": list(gpu_ids),
                    "adapter_directory": (
                        str(args.adapter_dir.resolve())
                        if args.adapter_dir is not None
                        else None
                    ),
                    "workflow": workflow,
                },
            )
        finally:
            progress.close()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["summary"]["pipeline_pass"] else 1
    except (ConfigError, SpiderDataError, RuntimeError, ValueError, OSError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
