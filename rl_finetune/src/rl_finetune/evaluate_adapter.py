"""Compatibility CLI delegating adapter evaluation to the current pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from text2sql.cli import main as text2sql_main


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rl_finetune.evaluate_adapter",
        description="Distributed PEFT adapter evaluation via `text2sql evaluate`",
    )
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    naming = parser.add_mutually_exclusive_group(required=True)
    naming.add_argument("--run-name")
    naming.add_argument("--resume-run")
    parser.add_argument("--selection", choices=("smoke", "all"), default="smoke")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    delegated = [
        "evaluate",
        "--config",
        str(args.baseline_config),
        "--backend",
        "peft",
        "--adapter-dir",
        str(args.adapter_dir),
        "--gpus",
        args.gpus,
        "--selection",
        args.selection,
        "--output-dir",
        str(args.output_dir),
        "--progress-interval-seconds",
        str(args.progress_interval_seconds),
    ]
    if args.run_name is not None:
        delegated.extend(("--run-name", args.run_name))
    else:
        delegated.extend(("--resume-run", args.resume_run))
    if args.allow_model_download:
        delegated.append("--allow-model-download")
    if args.no_progress:
        delegated.append("--no-progress")
    return text2sql_main(delegated)


if __name__ == "__main__":
    raise SystemExit(main())
