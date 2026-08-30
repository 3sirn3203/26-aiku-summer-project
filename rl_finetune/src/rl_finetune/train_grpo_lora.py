from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from text2sql.config import AppConfig, ConfigError, load_config
from rl_finetune.dataset import (
    build_prompt_records,
    read_indices_json,
    write_jsonl,
)
from rl_finetune.rewards import RewardConfig
from rl_finetune.training_runtime import (
    check_training_dependencies,
    create_lora_config,
    load_model_and_tokenizer,
    write_adapter_provenance,
)
from rl_finetune.trl_reward import (
    TRLRewardRuntime,
    make_text2sql_reward_func,
)

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rl_finetune.train_grpo_lora",
        description="GRPO + LoRA fine-tuning entrypoint for Spider Text-to-SQL",
    )
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--examples-file", default="train_spider.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--indices-file", type=Path)
    parser.add_argument(
        "--dry-run-dataset",
        action="store_true",
        help="write the prompt dataset and manifest without importing training libraries",
    )

    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="all-linear")

    parser.add_argument("--latency-weight", type=float, default=0.1)
    parser.add_argument("--latency-clip", type=float, default=1.0)
    parser.add_argument("--gold-error-reward", type=float, default=0.0)
    parser.add_argument(
        "--reward-trace",
        action="store_true",
        help="append per-completion reward diagnostics under the run directory",
    )
    return parser


def _training_config(config: AppConfig, *, examples_file: str, split: str) -> AppConfig:
    return replace(
        config,
        spider=replace(config.spider, examples_file=examples_file, split=split),
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _accepted_kwargs(callable_obj: Any, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    parameters = inspect.signature(callable_obj).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in parameters}


def _check_training_dependencies() -> Dict[str, str]:
    return check_training_dependencies()


def _create_grpo_config(args: argparse.Namespace) -> Any:
    from trl import GRPOConfig

    kwargs = {
        "output_dir": str(args.run_dir / "trainer"),
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "remove_unused_columns": False,
        "report_to": [],
    }
    return GRPOConfig(**_accepted_kwargs(GRPOConfig.__init__, kwargs))


def _load_model_and_tokenizer(config: AppConfig) -> Any:
    return load_model_and_tokenizer(config)


def _create_lora_config(args: argparse.Namespace) -> Any:
    return create_lora_config(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )


def _create_trainer(
    *,
    model: Any,
    tokenizer: Any,
    training_args: Any,
    train_dataset: Any,
    reward_func: Any,
    lora_config: Any,
) -> Any:
    from trl import GRPOTrainer

    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "reward_funcs": reward_func,
        "peft_config": lora_config,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    return GRPOTrainer(**_accepted_kwargs(GRPOTrainer.__init__, kwargs))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise ConfigError("--limit must be positive")
    if args.offset < 0:
        raise ConfigError("--offset must be non-negative")
    dependency_versions: Optional[Dict[str, str]] = None
    if not args.dry_run_dataset:
        dependency_versions = _check_training_dependencies()

    baseline_config = load_config(args.baseline_config)
    train_config = _training_config(
        baseline_config, examples_file=args.examples_file, split=args.split
    )
    run_dir = (args.output_dir / args.run_name).resolve()
    if run_dir.exists():
        raise ConfigError("run directory already exists: %s" % run_dir)
    args.run_dir = run_dir

    indices = read_indices_json(args.indices_file) if args.indices_file else None
    records = build_prompt_records(
        train_config,
        limit=args.limit,
        offset=args.offset,
        indices=indices,
    )
    if not records:
        raise ConfigError("training selection produced no records")

    started = datetime.now(timezone.utc)
    dataset_path = run_dir / "train_dataset.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    write_jsonl(dataset_path, records)
    manifest = {
        "schema_version": 1,
        "run_type": "rl_grpo_lora",
        "status": "dataset_ready" if args.dry_run_dataset else "initializing_trainer",
        "started_at": started.isoformat(),
        "baseline_config": str(args.baseline_config.resolve()),
        "config": {
            "model": {
                "id": train_config.model.model_id,
                "revision": train_config.model.revision,
                "dtype": train_config.model.dtype,
                "attention_implementation": train_config.model.attention_implementation,
            },
            "spider": {
                "root": str(train_config.spider.root),
                "examples_file": train_config.spider.examples_file,
                "split": train_config.spider.split,
            },
            "selection": {
                "limit": args.limit,
                "offset": args.offset,
                "indices_file": str(args.indices_file) if args.indices_file else None,
                "record_count": len(records),
            },
            "grpo": {
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "num_generations": args.num_generations,
                "temperature": args.temperature,
                "max_prompt_length": args.max_prompt_length,
                "max_completion_length": args.max_completion_length,
            },
            "lora": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": args.lora_target_modules,
            },
            "reward": {
                "latency_weight": args.latency_weight,
                "latency_clip": args.latency_clip,
                "gold_error_reward": args.gold_error_reward,
                "correctness_first": True,
            },
        },
        "dependencies": dependency_versions,
        "artifacts": {
            "train_dataset": "train_dataset.jsonl",
            "reward_trace": "reward_trace.jsonl" if args.reward_trace else None,
            "trainer_output": "trainer/",
            "adapter": "adapter/",
        },
    }
    _write_json(manifest_path, manifest)
    if args.dry_run_dataset:
        print(json.dumps({"run_directory": str(run_dir), "record_count": len(records)}))
        return 0

    from datasets import Dataset

    reward_config = RewardConfig(
        latency_weight=args.latency_weight,
        latency_clip=args.latency_clip,
    )
    trace_path = run_dir / "reward_trace.jsonl" if args.reward_trace else None
    reward_func = make_text2sql_reward_func(
        TRLRewardRuntime(
            execution=train_config.execution,
            reward=reward_config,
            gold_error_reward=args.gold_error_reward,
            trace_path=trace_path,
        )
    )
    train_dataset = Dataset.from_list(records)
    training_args = _create_grpo_config(args)
    model, tokenizer = _load_model_and_tokenizer(train_config)
    lora_config = _create_lora_config(args)
    trainer = _create_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        reward_func=reward_func,
        lora_config=lora_config,
    )
    manifest["status"] = "training"
    _write_json(manifest_path, manifest)
    trainer.train()
    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    write_adapter_provenance(
        adapter_dir,
        train_config,
        training_run_type="rl_grpo_lora",
    )
    manifest["status"] = "completed"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)
    print(json.dumps({"run_directory": str(run_dir), "record_count": len(records)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConfigError, RuntimeError, ValueError, OSError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
