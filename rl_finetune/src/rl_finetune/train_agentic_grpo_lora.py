from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from text2sql.config import AppConfig, ConfigError, load_config
from rl_finetune.agentic_runtime.environment import SQLWorkflowEnvironment
from rl_finetune.agentic_runtime.rollout import (
    TransformersWorkflowPolicy,
    WorkflowRolloutRunner,
)
from rl_finetune.agentic_grpo_trainer import (
    AgenticGRPOConfig,
    AgenticGRPOTrainer,
    restore_rng_state,
)
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
from rl_finetune.trajectory_reward import TrajectoryRewardRuntime


_AGENTIC_V1_BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
_AGENTIC_V1_BASE_REVISION = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rl_finetune.train_agentic_grpo_lora",
        description="Two-turn workflow GRPO + LoRA training for Spider Text-to-SQL",
    )
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--examples-file", default="train_spider.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--indices-file", type=Path)
    parser.add_argument("--dry-run-dataset", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path)

    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-draft-tokens", type=int, default=256)
    parser.add_argument("--max-final-tokens", type=int, default=256)
    parser.add_argument("--max-observation-rows", type=int, default=20)
    parser.add_argument(
        "--credit-mode", choices=("all_actions", "final_only"), default="all_actions"
    )
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--trajectory-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write full context/action token trajectory traces (default: enabled)",
    )

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", default="all-linear")
    return parser


def _training_config(config: AppConfig, *, examples_file: str, split: str) -> AppConfig:
    return replace(
        config,
        spider=replace(config.spider, examples_file=examples_file, split=split),
    )


def _validate_base_provenance(config: AppConfig) -> None:
    if (
        config.model.model_id != _AGENTIC_V1_BASE_MODEL_ID
        or config.model.revision != _AGENTIC_V1_BASE_REVISION
    ):
        raise ConfigError(
            "agentic v1 requires the pinned Coder base %s@%s"
            % (_AGENTIC_V1_BASE_MODEL_ID, _AGENTIC_V1_BASE_REVISION)
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _validate_args(args: argparse.Namespace) -> None:
    if args.dry_run_dataset and args.resume_from_checkpoint is not None:
        raise ConfigError(
            "--dry-run-dataset cannot be combined with --resume-from-checkpoint"
        )
    positive_names = (
        "max_steps",
        "gradient_accumulation_steps",
        "num_generations",
        "max_draft_tokens",
        "max_final_tokens",
        "max_observation_rows",
        "lora_r",
        "lora_alpha",
    )
    for name in positive_names:
        if getattr(args, name) < 1:
            raise ConfigError("--%s must be positive" % name.replace("_", "-"))
    if args.num_generations < 2:
        raise ConfigError("--num-generations must be at least 2 for GRPO")
    if args.limit is not None and args.limit < 1:
        raise ConfigError("--limit must be positive")
    if args.offset < 0:
        raise ConfigError("--offset must be non-negative")
    if args.temperature <= 0 or args.learning_rate <= 0:
        raise ConfigError("temperature and learning rate must be positive")
    if args.lora_dropout < 0:
        raise ConfigError("--lora-dropout must be non-negative")


def _dataset_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    identity = [
        {
            "example_id": record.get("example_id"),
            "db_id": record.get("db_id"),
            "prompt_sha256": record.get("prompt_sha256"),
            "schema_sha256": record.get("schema_sha256"),
            "gold_sql": record.get("gold_sql"),
        }
        for record in records
    ]
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _new_checkpoint_metadata(
    config: AppConfig,
    args: argparse.Namespace,
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "model": {
            "id": config.model.model_id,
            "revision": config.model.revision,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
        },
        "rollout": _rollout_contract(args),
        "dataset_sha256": _dataset_fingerprint(records),
    }


def _rollout_contract(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "max_draft_tokens": args.max_draft_tokens,
        "max_final_tokens": args.max_final_tokens,
        "max_observation_rows": args.max_observation_rows,
    }


def _checkpoint_metadata(
    checkpoint: Path,
    *,
    config: AppConfig,
    records: Sequence[Mapping[str, Any]],
    trainer_config: AgenticGRPOConfig,
    rollout_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    state_path = checkpoint / "trainer_state.json"
    required = {
        "trainer_state": state_path,
        "optimizer": checkpoint / "optimizer.pt",
        "scheduler": checkpoint / "scheduler.pt",
        "rng": checkpoint / "rng_state.pt",
        "adapter": checkpoint / "adapter_config.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ConfigError(
            "resume checkpoint is incomplete; missing: %s" % ", ".join(missing)
        )
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        adapter_config = json.loads(required["adapter"].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ConfigError("resume checkpoint metadata is invalid: %s" % exc) from exc
    metadata = state.get("checkpoint_metadata")
    if not isinstance(metadata, dict):
        raise ConfigError(
            "resume checkpoint has no verifiable model/dataset provenance"
        )
    if metadata.get("schema_version") != 1:
        raise ConfigError("resume checkpoint provenance schema is unsupported")
    expected_model = {
        "id": config.model.model_id,
        "revision": config.model.revision,
    }
    if metadata.get("model") != expected_model:
        raise ConfigError(
            "resume checkpoint model provenance does not match baseline config"
        )
    if metadata.get("dataset_sha256") != _dataset_fingerprint(records):
        raise ConfigError(
            "resume checkpoint dataset selection does not match this run"
        )
    lora = metadata.get("lora")
    if not isinstance(lora, dict):
        raise ConfigError("resume checkpoint has no LoRA provenance")
    if metadata.get("rollout") != dict(rollout_contract):
        raise ConfigError(
            "resume checkpoint rollout configuration does not match this run"
        )
    saved_trainer_config = state.get("config")
    if not isinstance(saved_trainer_config, dict):
        raise ConfigError("resume checkpoint has no trainer configuration")
    expected_trainer_config = asdict(trainer_config)
    for key, expected_value in expected_trainer_config.items():
        if key == "max_steps":
            continue
        if saved_trainer_config.get(key) != expected_value:
            raise ConfigError(
                "resume checkpoint trainer setting %r does not match this run" % key
            )
    adapter_base = adapter_config.get("base_model_name_or_path")
    if adapter_base and adapter_base != config.model.model_id:
        raise ConfigError(
            "resume adapter base model %r does not match %r"
            % (adapter_base, config.model.model_id)
        )
    adapter_lora_pairs = {
        "r": "r",
        "alpha": "lora_alpha",
        "dropout": "lora_dropout",
    }
    for metadata_key, adapter_key in adapter_lora_pairs.items():
        if adapter_key in adapter_config and adapter_config[adapter_key] != lora.get(
            metadata_key
        ):
            raise ConfigError(
                "resume adapter LoRA setting %r does not match checkpoint provenance"
                % metadata_key
            )
    try:
        global_step = int(state["global_step"])
        record_index = int(state.get("record_index", 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError("resume checkpoint trainer state is invalid: %s" % exc) from exc
    return {
        "checkpoint": checkpoint,
        "global_step": global_step,
        "record_index": record_index,
        "optimizer_path": required["optimizer"],
        "scheduler_path": required["scheduler"],
        "rng_path": required["rng"],
        "metadata": metadata,
    }


def _prepare_peft_model(
    base_model: Any, args: argparse.Namespace, resume_checkpoint: Optional[Path]
) -> Any:
    if resume_checkpoint is not None:
        from peft import PeftModel

        return PeftModel.from_pretrained(
            base_model, str(resume_checkpoint.resolve()), is_trainable=True
        )
    from peft import get_peft_model

    return get_peft_model(
        base_model,
        create_lora_config(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        ),
    )


def _make_runner(
    *,
    model: Any,
    tokenizer: Any,
    config: AppConfig,
    args: argparse.Namespace,
    device: Any,
    num_generations: Optional[int] = None,
) -> WorkflowRolloutRunner:
    policy = TransformersWorkflowPolicy(
        model,
        tokenizer,
        device=device,
        max_input_tokens=config.generation.max_input_tokens,
        max_time_seconds=config.generation.max_time_seconds,
    )
    environment = SQLWorkflowEnvironment(
        config.execution, max_observation_rows=args.max_observation_rows
    )
    return WorkflowRolloutRunner(
        policy,
        environment,
        num_generations=num_generations or args.num_generations,
        temperature=args.temperature,
        max_draft_tokens=args.max_draft_tokens,
        max_final_tokens=args.max_final_tokens,
    )


def _round_trip_adapter(
    *,
    adapter_dir: Path,
    config: AppConfig,
    args: argparse.Namespace,
    record: Mapping[str, Any],
) -> Dict[str, Any]:
    import torch
    from peft import PeftModel

    base_model, tokenizer = load_model_and_tokenizer(config)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
    device = torch.device(config.model.device)
    model.to(device)
    runner = _make_runner(
        model=model,
        tokenizer=tokenizer,
        config=config,
        args=args,
        device=device,
        num_generations=1,
    )
    trajectory = runner.collect_group(record)[0]
    return {
        "adapter_reload": True,
        "workflow_inference": True,
        "example_id": trajectory.example_id,
        "draft_status": trajectory.steps[0].observation.status,
        "final_parse_status": trajectory.terminal_status,
    }


def _execute_training(
    *,
    args: argparse.Namespace,
    train_config: AppConfig,
    records: Sequence[Mapping[str, Any]],
    trainer_config: AgenticGRPOConfig,
    run_dir: Path,
    resume_state: Optional[Mapping[str, Any]],
    checkpoint_metadata: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    import numpy
    import torch

    if not torch.cuda.is_available():
        raise ConfigError("agentic GRPO training requires CUDA")
    device = torch.device(train_config.model.device)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    resume_checkpoint = (
        Path(str(resume_state["checkpoint"])) if resume_state is not None else None
    )
    base_model, tokenizer = load_model_and_tokenizer(train_config)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _prepare_peft_model(base_model, args, resume_checkpoint)
    model.to(device)
    if resume_state is not None:
        restore_rng_state(Path(str(resume_state["rng_path"])))

    runner = _make_runner(
        model=model,
        tokenizer=tokenizer,
        config=train_config,
        args=args,
        device=device,
    )
    reward_runtime = TrajectoryRewardRuntime(
        train_config.execution,
        reward_config=RewardConfig(latency_weight=0.0),
    )
    trainer = AgenticGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        rollout_runner=runner,
        reward_runtime=reward_runtime,
        records=records,
        config=trainer_config,
        run_dir=run_dir,
        device=device,
        initial_global_step=(
            int(resume_state["global_step"]) if resume_state is not None else 0
        ),
        initial_record_index=(
            int(resume_state["record_index"]) if resume_state is not None else 0
        ),
        optimizer_state_path=(
            Path(str(resume_state["optimizer_path"]))
            if resume_state is not None
            else None
        ),
        scheduler_state_path=(
            Path(str(resume_state["scheduler_path"]))
            if resume_state is not None
            else None
        ),
        checkpoint_metadata=checkpoint_metadata,
    )
    result = trainer.train(write_trajectory_trace=args.trajectory_trace)
    adapter_dir = run_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    write_adapter_provenance(
        adapter_dir,
        train_config,
        training_run_type="two_turn_workflow_grpo_lora",
    )

    del trainer, runner, model, base_model
    torch.cuda.empty_cache()
    round_trip = _round_trip_adapter(
        adapter_dir=adapter_dir,
        config=train_config,
        args=args,
        record=records[0],
    )
    return result, round_trip


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    dependency_versions: Optional[Dict[str, str]] = None
    if not args.dry_run_dataset:
        dependency_versions = check_training_dependencies()

    baseline_config = load_config(args.baseline_config)
    train_config = _training_config(
        baseline_config, examples_file=args.examples_file, split=args.split
    )
    _validate_base_provenance(train_config)
    run_dir = (args.output_dir / args.run_name).resolve()
    if run_dir.exists():
        raise ConfigError("run directory already exists: %s" % run_dir)

    indices = read_indices_json(args.indices_file) if args.indices_file else None
    records = build_prompt_records(
        train_config, limit=args.limit, offset=args.offset, indices=indices
    )
    if not records:
        raise ConfigError("training selection produced no records")

    trainer_config = AgenticGRPOConfig(
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_epsilon=args.clip_epsilon,
        kl_beta=args.kl_beta,
        max_grad_norm=args.max_grad_norm,
        credit_mode=args.credit_mode,
        policy_temperature=args.temperature,
    )
    rollout_contract = _rollout_contract(args)
    resume_state: Optional[Dict[str, Any]] = None
    if args.resume_from_checkpoint is not None and not args.dry_run_dataset:
        resume_state = _checkpoint_metadata(
            args.resume_from_checkpoint,
            config=train_config,
            records=records,
            trainer_config=trainer_config,
            rollout_contract=rollout_contract,
        )
        checkpoint_metadata = dict(resume_state["metadata"])
    else:
        checkpoint_metadata = _new_checkpoint_metadata(train_config, args, records)
    effective_lora = dict(checkpoint_metadata["lora"])

    write_jsonl(run_dir / "train_dataset.jsonl", records)
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "run_type": "two_turn_workflow_grpo_lora",
        "status": "dataset_ready" if args.dry_run_dataset else "initializing_trainer",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "baseline_config": str(args.baseline_config.resolve()),
        "config": {
            "model": {
                "id": train_config.model.model_id,
                "revision": train_config.model.revision,
                "dtype": train_config.model.dtype,
                "attention_implementation": train_config.model.attention_implementation,
            },
            "selection": {
                "examples_file": args.examples_file,
                "split": args.split,
                "limit": args.limit,
                "offset": args.offset,
                "indices_file": str(args.indices_file) if args.indices_file else None,
                "record_count": len(records),
            },
            "rollout": {
                "turns": 2,
                **rollout_contract,
            },
            "grpo": asdict(trainer_config),
            "lora": effective_lora,
            "reward": {"latency_weight": 0.0, "final_only": True},
            "seed": args.seed,
            "resume_from_checkpoint": str(args.resume_from_checkpoint.resolve())
            if args.resume_from_checkpoint
            else None,
        },
        "dependencies": dependency_versions,
        "artifacts": {
            "train_dataset": "train_dataset.jsonl",
            "trajectory_trace": "trajectory_trace.jsonl"
            if args.trajectory_trace
            else None,
            "reward_trace": "reward_trace.jsonl",
            "trainer_output": "trainer/",
            "adapter": "adapter/",
        },
    }
    manifest_path = run_dir / "run_manifest.json"
    _write_json(manifest_path, manifest)
    if args.dry_run_dataset:
        print(json.dumps({"run_directory": str(run_dir), "record_count": len(records)}))
        return 0
    try:
        manifest["status"] = "training"
        _write_json(manifest_path, manifest)
        result, round_trip = _execute_training(
            args=args,
            train_config=train_config,
            records=records,
            trainer_config=trainer_config,
            run_dir=run_dir,
            resume_state=resume_state,
            checkpoint_metadata=checkpoint_metadata,
        )
        manifest["status"] = "completed"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["training_result"] = result
        manifest["round_trip_verification"] = round_trip
        _write_json(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        _write_json(manifest_path, manifest)
        raise

    print(json.dumps({"run_directory": str(run_dir), "record_count": len(records)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConfigError, RuntimeError, ValueError, OSError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
