from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from text2sql.config import AppConfig, ConfigError, load_config
from rl_finetune.dataset import read_indices_json, write_jsonl
from rl_finetune.training_runtime import (
    check_training_dependencies,
    create_lora_config,
    load_model_and_tokenizer,
    write_adapter_provenance,
)
from rl_vm_step.config import VMStepRewardConfig
from rl_vm_step.dataset import build_vm_step_records, gold_reference_rows
from rl_vm_step.distributed import (
    DistributedContext,
    merge_reward_traces,
    reward_trace_path,
)
from rl_vm_step.prepared_dataset import load_prepared_dataset
from rl_vm_step.trl_reward import TRLVMStepRewardRuntime, make_vm_step_reward_func


VM_STEP_V1_BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
VM_STEP_V1_BASE_REVISION = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rl_vm_step.train_grpo_lora",
        description="Single-turn VM-step GRPO + LoRA training for Spider Text-to-SQL",
    )
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--examples-file", default="train_spider.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--indices-file", type=Path)
    parser.add_argument("--prepared-dataset-dir", type=Path)
    parser.add_argument("--dry-run-dataset", action="store_true")
    parser.add_argument(
        "--fail-on-rejected-gold-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--kl-beta", type=float, default=0.04)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--fp16", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa"),
        default="sdpa",
    )

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="all-linear")

    parser.add_argument("--vm-weight", type=float, default=0.1)
    parser.add_argument("--vm-clip", type=float, default=1.0)
    parser.add_argument("--vm-epsilon-steps", type=float, default=1_000.0)
    parser.add_argument(
        "--reward-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "learning_rate",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "num_generations",
        "temperature",
        "max_grad_norm",
        "max_prompt_length",
        "max_completion_length",
        "logging_steps",
        "save_steps",
        "lora_r",
        "lora_alpha",
        "vm_epsilon_steps",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ConfigError("--%s must be positive" % name.replace("_", "-"))
    if args.num_generations < 2:
        raise ConfigError("--num-generations must be at least 2 for GRPO")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ConfigError("--max-steps must be -1 or positive")
    if args.limit is not None and args.limit < 1:
        raise ConfigError("--limit must be positive")
    if args.offset < 0:
        raise ConfigError("--offset must be non-negative")
    if args.lora_dropout < 0:
        raise ConfigError("--lora-dropout must be non-negative")
    if args.kl_beta < 0:
        raise ConfigError("--kl-beta must be non-negative")
    if args.prepared_dataset_dir is not None and (
        args.limit is not None or args.offset != 0 or args.indices_file is not None
    ):
        raise ConfigError(
            "prepared datasets cannot be combined with limit/offset/indices"
        )
    if args.prepared_dataset_dir is not None and args.dry_run_dataset:
        raise ConfigError("prepared datasets cannot be combined with dataset dry-run")


def _training_config(
    config: AppConfig,
    *,
    examples_file: str,
    split: str,
    attention_implementation: Optional[str] = None,
) -> AppConfig:
    return replace(
        config,
        spider=replace(config.spider, examples_file=examples_file, split=split),
        model=(
            replace(
                config.model,
                attention_implementation=attention_implementation,
            )
            if attention_implementation is not None
            else config.model
        ),
    )


def _validate_base_provenance(config: AppConfig) -> None:
    if (
        config.model.model_id != VM_STEP_V1_BASE_MODEL_ID
        or config.model.revision != VM_STEP_V1_BASE_REVISION
    ):
        raise ConfigError(
            "VM-step v1 requires %s@%s"
            % (VM_STEP_V1_BASE_MODEL_ID, VM_STEP_V1_BASE_REVISION)
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _accepted_kwargs(callable_obj: Any, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    parameters = inspect.signature(callable_obj).parameters
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in parameters}


def _create_grpo_config(args: argparse.Namespace) -> Any:
    from trl import GRPOConfig

    kwargs = {
        "output_dir": str(args.run_dir / "trainer"),
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "beta": args.kl_beta,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "fp16": args.fp16,
        "bf16": False,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": (
            {"use_reentrant": False} if args.gradient_checkpointing else None
        ),
        "ddp_find_unused_parameters": False,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "remove_unused_columns": False,
        "report_to": [],
    }
    return GRPOConfig(**_accepted_kwargs(GRPOConfig.__init__, kwargs))


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

    from unittest.mock import patch

    from trl.trainer import grpo_trainer as grpo_module
    from rl_vm_step.fsdp_compat import unwrap_model_for_fsdp_generation

    original_unwrap = grpo_module.unwrap_model_for_generation

    class GenerationOptimizedGRPOTrainer(GRPOTrainer):
        def compute_loss(self, *compute_args: Any, **compute_kwargs: Any) -> Any:
            replacement = lambda model, accelerator: (
                unwrap_model_for_fsdp_generation(
                    model, accelerator, fallback=original_unwrap
                )
            )
            with patch.object(
                grpo_module,
                "unwrap_model_for_generation",
                replacement,
            ):
                return super().compute_loss(*compute_args, **compute_kwargs)

    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "reward_funcs": reward_func,
        "peft_config": lora_config,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    return GenerationOptimizedGRPOTrainer(
        **_accepted_kwargs(GenerationOptimizedGRPOTrainer.__init__, kwargs)
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    context = DistributedContext.from_environment()
    if context.world_size > 1 and args.prepared_dataset_dir is None:
        raise ConfigError(
            "multi-GPU training requires --prepared-dataset-dir; run "
            "python -m rl_vm_step.prepare_dataset first"
        )
    dependencies = None if args.dry_run_dataset else check_training_dependencies()
    baseline = load_config(args.baseline_config)
    config = _training_config(
        baseline,
        examples_file=args.examples_file,
        split=args.split,
        attention_implementation=args.attention_implementation,
    )
    _validate_base_provenance(config)
    reward_config = VMStepRewardConfig(
        vm_weight=args.vm_weight,
        vm_clip=args.vm_clip,
        vm_epsilon_steps=args.vm_epsilon_steps,
    )
    run_dir = (args.output_dir / args.run_name).resolve()
    args.run_dir = run_dir
    prepared_manifest: Optional[Dict[str, Any]] = None
    rejected: Sequence[Mapping[str, Any]] = []
    if args.prepared_dataset_dir is not None:
        records, prepared_manifest = load_prepared_dataset(
            args.prepared_dataset_dir, config
        )
        selection = prepared_manifest["selection"]
        reference_summary = {
            "selected_count": selection["selected_count"],
            "eligible_count": selection["eligible_count"],
            "rejected_count": selection["rejected_count"],
            "measurement": prepared_manifest["measurement"],
        }
    else:
        indices = read_indices_json(args.indices_file) if args.indices_file else None
        records, rejected = build_vm_step_records(
            config, limit=args.limit, offset=args.offset, indices=indices
        )
        reference_summary = {
            "selected_count": len(records) + len(rejected),
            "eligible_count": len(records),
            "rejected_count": len(rejected),
            "measurement": "sqlite_progress_handler_interval_midpoint",
        }

    state = None
    if context.world_size > 1:
        from accelerate import PartialState

        state = PartialState()
    if context.is_main_process:
        if run_dir.exists():
            raise ConfigError("run directory already exists: %s" % run_dir)
        run_dir.mkdir(parents=True)
        if args.prepared_dataset_dir is None:
            write_jsonl(run_dir / "train_dataset.jsonl", records)
            write_jsonl(
                run_dir / "gold_reference.jsonl", gold_reference_rows(records)
            )
            write_jsonl(run_dir / "rejected_gold_reference.jsonl", rejected)
            _write_json(
                run_dir / "gold_reference_summary.json", reference_summary
            )
    if state is not None:
        state.wait_for_everyone()

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "run_type": "single_turn_vm_step_grpo_lora",
        "status": "dataset_ready",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "baseline_config": str(args.baseline_config.resolve()),
        "config": {
            "model": {
                "id": config.model.model_id,
                "revision": config.model.revision,
                "dtype": config.model.dtype,
                "attention_implementation": config.model.attention_implementation,
            },
            "selection": {
                "examples_file": args.examples_file,
                "split": args.split,
                "limit": args.limit,
                "offset": args.offset,
                "indices_file": str(args.indices_file) if args.indices_file else None,
                **reference_summary,
            },
            "grpo": {
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "max_steps": args.max_steps,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "num_generations": args.num_generations,
                "temperature": args.temperature,
                "kl_beta": args.kl_beta,
                "max_grad_norm": args.max_grad_norm,
                "seed": args.seed,
                "max_prompt_length": args.max_prompt_length,
                "max_completion_length": args.max_completion_length,
                "fp16": args.fp16,
                "bf16": False,
                "gradient_checkpointing": args.gradient_checkpointing,
                "gradient_checkpointing_use_reentrant": (
                    False if args.gradient_checkpointing else None
                ),
            },
            "lora": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": args.lora_target_modules,
            },
            "reward": asdict(reward_config),
            "distributed": {
                "world_size": context.world_size,
                "launcher": "accelerate" if context.world_size > 1 else "python",
                "strategy": (
                    "fsdp"
                    if os.environ.get("ACCELERATE_USE_FSDP") == "true"
                    else "ddp" if context.world_size > 1 else "single_process"
                ),
            },
            "prepared_dataset": str(args.prepared_dataset_dir.resolve())
            if args.prepared_dataset_dir
            else None,
        },
        "dependencies": dependencies,
        "artifacts": {
            "train_dataset": (
                str(args.prepared_dataset_dir.resolve() / "train_dataset.jsonl")
                if args.prepared_dataset_dir
                else "train_dataset.jsonl"
            ),
            "gold_reference": (
                str(args.prepared_dataset_dir.resolve() / "gold_reference.jsonl")
                if args.prepared_dataset_dir
                else "gold_reference.jsonl"
            ),
            "reward_trace": "reward_trace.jsonl" if args.reward_trace else None,
            "rank_reward_traces": "reward_trace/" if context.world_size > 1 else None,
            "trainer_output": "trainer/",
            "adapter": "adapter/",
        },
    }
    manifest_path = run_dir / "run_manifest.json"
    if context.is_main_process:
        _write_json(manifest_path, manifest)
    if not records:
        if context.is_main_process:
            manifest["status"] = "failed"
            manifest["error"] = {"message": "no eligible VM-step training records"}
            _write_json(manifest_path, manifest)
        raise ConfigError("training selection produced no eligible VM-step records")
    if rejected and args.fail_on_rejected_gold_reference:
        if context.is_main_process:
            manifest["status"] = "failed"
            manifest["error"] = {
                "message": "%d gold references were rejected" % len(rejected)
            }
            _write_json(manifest_path, manifest)
        raise ConfigError(
            "%d gold references were rejected; inspect rejected_gold_reference.jsonl"
            % len(rejected)
        )
    if args.dry_run_dataset:
        if context.is_main_process:
            print(
                json.dumps(
                    {
                        "run_directory": str(run_dir),
                        "record_count": len(records),
                        "rejected_count": len(rejected),
                    }
                )
            )
        return 0

    from datasets import Dataset

    trace_path = reward_trace_path(run_dir, context) if args.reward_trace else None
    reward_func = make_vm_step_reward_func(
        TRLVMStepRewardRuntime(
            execution=config.execution,
            reward=reward_config,
            trace_path=trace_path,
            strict_measurement=True,
        )
    )
    training_args = _create_grpo_config(args)
    try:
        import torch

        model, tokenizer = load_model_and_tokenizer(
            config,
            training_dtype=torch.float16 if args.fp16 else None,
        )
        if args.gradient_checkpointing:
            model.config.use_cache = False
            model.enable_input_require_grads()
        lora_config = create_lora_config(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        trainer = _create_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            train_dataset=Dataset.from_list(records),
            reward_func=reward_func,
            lora_config=lora_config,
        )
        if context.is_main_process:
            manifest["status"] = "training"
            _write_json(manifest_path, manifest)
        trainer.train()
        adapter_dir = run_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        if state is not None:
            state.wait_for_everyone()
        if context.is_main_process:
            tokenizer.save_pretrained(str(adapter_dir))
            write_adapter_provenance(
                adapter_dir,
                config,
                training_run_type="single_turn_vm_step_grpo_lora",
            )
            if args.reward_trace and context.world_size > 1:
                merge_reward_traces(run_dir, context.world_size)
            manifest["status"] = "completed"
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(manifest_path, manifest)
        if state is not None:
            state.wait_for_everyone()
    except Exception as exc:
        if context.is_main_process:
            manifest["status"] = "failed"
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            manifest["error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:1000],
            }
            _write_json(manifest_path, manifest)
        raise
    if context.is_main_process:
        print(
            json.dumps(
                {"run_directory": str(run_dir), "record_count": len(records)}
            )
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConfigError, RuntimeError, ValueError, OSError) as exc:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise
        print("error: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
