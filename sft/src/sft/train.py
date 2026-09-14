from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainerCallback, TrainingArguments, set_seed)

from .collator import AnswerOnlyCollator, pretokenize
from .data import build_examples


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _rank():
    return int(os.environ.get("RANK", "0"))


def _world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def _distributed_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _release_distributed_training():
    """Release DDP replicas before rank zero starts one inference worker per GPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _distributed_barrier()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    # EvaluationPool workers are standalone processes, not members of the torchrun job.
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                 "ROLE_RANK", "ROLE_WORLD_SIZE"):
        os.environ.pop(name, None)


class ResourceCallback(TrainerCallback):
    def __init__(self, cfg):
        self.cfg = cfg

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available() and state.is_world_process_zero:
            free, total = torch.cuda.mem_get_info()
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            effective_free = free + max(reserved - allocated, 0)
            print(f"resource step={state.global_step} allocated_gib={allocated / 2**30:.2f} "
                  f"device_used_pct={(total-effective_free) / total * 100:.1f}", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        reason = None
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            effective_free = free + max(reserved - allocated, 0)
            if ((total - effective_free) / total > self.cfg.max_vram_fraction or
                    effective_free / 2**20 < self.cfg.min_free_vram_mb):
                reason = (f"GPU resource threshold reached at step {state.global_step}: "
                          f"effective_free_mib={effective_free / 2**20:.0f}")
        try:
            import psutil
            available = psutil.virtual_memory().available / 2**30
            if available < self.cfg.min_free_ram_gb:
                reason = (f"RAM resource threshold reached at step {state.global_step}: "
                          f"available_gib={available:.2f}")
        except ImportError:
            pass
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            stop = torch.tensor(int(reason is not None), device=args.device)
            torch.distributed.all_reduce(stop, op=torch.distributed.ReduceOp.MAX)
            if stop.item() and reason is None:
                reason = (f"Resource threshold reached on another rank at step "
                          f"{state.global_step}")
        if reason:
            if state.is_world_process_zero:
                print(reason + "; requesting a checkpointed graceful stop", flush=True)
            control.should_save = True
            control.should_training_stop = True
        return control


def load_tokenizer(cfg):
    shared = {"trust_remote_code": cfg.model.trust_remote_code,
              "local_files_only": cfg.model.local_files_only}
    if cfg.model.revision:
        shared["revision"] = cfg.model.revision
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_name_or_path or cfg.model.name_or_path, **shared)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires pad_token or eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if cfg.model.chat_template:
        tokenizer.chat_template = Path(cfg.model.chat_template).read_text()
    if tokenizer.chat_template is None:
        raise ValueError("Tokenizer needs a chat template")
    return tokenizer, shared


def load_model(cfg, shared):
    kwargs = dict(shared)
    kwargs["torch_dtype"] = getattr(torch, cfg.model.dtype)
    if cfg.model.attn_implementation:
        kwargs["attn_implementation"] = cfg.model.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, **kwargs)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=cfg.lora.rank, lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout, target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias))
    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def _manifest(cfg, model):
    targets = sorted({name.split(".lora_", 1)[0] for name, _ in model.named_modules()
                      if name.endswith(("lora_A.default", "lora_B.default"))})
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "wandb"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "config": asdict(cfg),
        "data_sha256": {name: _sha256(getattr(cfg.data, name))
                        for name in ("train_json", "dev_json", "tables")},
        "environment": {"python": platform.python_version(), "packages": packages,
                        "argv": sys.argv},
        "lora": {"resolved_target_modules": targets, "trainable_parameters": trainable,
                 "total_parameters": total},
    }


def train(cfg):
    cfg.validate()
    world_size = _world_size()
    if world_size != cfg.runtime.expected_world_size:
        raise ValueError(
            f"Expected {cfg.runtime.expected_world_size} training processes, got {world_size}; "
            "launch with torchrun --nproc_per_node matching runtime.expected_world_size")
    if world_size > 1 and (not torch.cuda.is_available()
                           or torch.cuda.device_count() < world_size):
        raise ValueError(f"DDP needs {world_size} visible CUDA devices")
    is_main = _rank() == 0
    if cfg.runtime.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = cfg.runtime.wandb_project
    set_seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    output = Path(cfg.training.output_dir)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists() and not cfg.training.resume_from_checkpoint:
        raise FileExistsError(
            f"Run directory already contains a manifest; configure resume_from_checkpoint: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer, shared = load_tokenizer(cfg)
    train_rows = build_examples(cfg.data.train_json, cfg.data.tables)
    dev_rows = build_examples(cfg.data.dev_json, cfg.data.tables)
    model = load_model(cfg, shared)
    if world_size == 1:
        model.to(torch.device(cfg.runtime.device))
    collator = AnswerOnlyCollator(tokenizer, cfg.training.max_length,
                                  cfg.model.chat_template_kwargs)
    train_rows, skipped_train = pretokenize(train_rows, collator)
    dev_rows, skipped_dev = pretokenize(dev_rows, collator)
    if is_main:
        _atomic_json(output / "skipped_overlength.json", {
            "train": skipped_train, "dev": skipped_dev,
            "train_kept": len(train_rows), "dev_kept": len(dev_rows)})

    validation = cfg.validation.enabled and cfg.validation.strategy != "no"
    args = TrainingArguments(
        output_dir=str(output / "checkpoints"), seed=cfg.training.seed,
        data_seed=cfg.training.seed, num_train_epochs=cfg.training.epochs,
        max_steps=cfg.training.max_steps,
        per_device_train_batch_size=cfg.training.per_device_batch_size,
        per_device_eval_batch_size=cfg.validation.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        max_grad_norm=cfg.training.max_grad_norm,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.training.logging_steps,
        save_strategy=(cfg.validation.strategy if validation else "steps"),
        save_steps=(cfg.validation.steps if validation and cfg.validation.strategy == "steps"
                    else cfg.training.save_steps),
        save_total_limit=cfg.training.save_total_limit,
        eval_strategy=(cfg.validation.strategy if validation else "no"),
        eval_steps=cfg.validation.steps,
        load_best_model_at_end=validation,
        metric_for_best_model="eval_loss", greater_is_better=False,
        fp16=cfg.model.dtype == "float16", bf16=cfg.model.dtype == "bfloat16",
        optim="adamw_torch", dataloader_num_workers=cfg.training.dataloader_num_workers,
        report_to=cfg.runtime.report_to, run_name=cfg.runtime.wandb_run_name,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_rows,
                      eval_dataset=dev_rows if validation else None,
                      data_collator=collator, processing_class=tokenizer,
                      callbacks=[ResourceCallback(cfg.runtime)])
    if trainer.is_world_process_zero():
        manifest = _manifest(cfg, model)
        manifest["distributed"] = {
            "world_size": world_size,
            "per_device_batch_size": cfg.training.per_device_batch_size,
            "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
            "effective_batch_size": (world_size * cfg.training.per_device_batch_size
                                     * cfg.training.gradient_accumulation_steps),
            "generation_devices": cfg.runtime.generation_devices,
        }
        manifest["data_counts"] = {"train_kept": len(train_rows), "dev_kept": len(dev_rows),
                                   "train_skipped": len(skipped_train),
                                   "dev_skipped": len(skipped_dev)}
        _atomic_json(output / "run_manifest.json", manifest)
        cfg.save(output / "config.json")
    result = trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
    metrics = dict(result.metrics)
    if validation:
        metrics.update(trainer.evaluate())
    final_adapter = output / "final_adapter"
    trainer.save_model(str(final_adapter))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_adapter)
        _atomic_json(output / "metrics.json", metrics)
    generation_metrics = None
    if cfg.export.merge and trainer.is_world_process_zero():
        merged = trainer.model.merge_and_unload()
        merged_dir = output / cfg.export.merged_subdir
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        del merged
    _distributed_barrier()
    del trainer
    del model
    _release_distributed_training()
    if not is_main:
        return None
    if cfg.validation.enabled and cfg.validation.generation_at_end:
        from rl.config import (Config as RLConfig, DataConfig, EvaluationConfig,
                               ModelConfig, RolloutConfig, RuntimeConfig, SQLConfig)
        from rl.evaluate import evaluate_policy
        from rl.runtime.evaluation_pool import EvaluationPool
        evaluation_cfg = RLConfig(
            model=ModelConfig(
                name_or_path=cfg.model.name_or_path, revision=cfg.model.revision,
                dtype=cfg.model.dtype, device=cfg.runtime.device, mode="lora",
                chat_format="chat", chat_template=cfg.model.chat_template,
                chat_template_kwargs=cfg.model.chat_template_kwargs),
            data=DataConfig(split_mode="official", train_json=cfg.data.train_json,
                            dev_json=cfg.data.dev_json, database_dir=cfg.data.database_dir,
                            tables=cfg.data.tables),
            sql=SQLConfig(evaluator_path=cfg.validation.evaluator_path,
                          nltk_data=cfg.validation.nltk_data),
            rollout=RolloutConfig(max_intermediate=0,
                                  max_context_tokens=cfg.training.max_length,
                                  max_new_tokens=cfg.validation.max_new_tokens),
            evaluation=EvaluationConfig(enabled=True, selection_metric="execution_accuracy",
                                        limit=cfg.validation.generation_limit,
                                        dev_suite_database_dir=cfg.validation.dev_suite_database_dir),
            runtime=RuntimeConfig())
        devices = cfg.runtime.generation_devices or [cfg.runtime.device]
        pool = EvaluationPool(evaluation_cfg, devices, checkpoint=final_adapter)
        try:
            generation_metrics = evaluate_policy(
                evaluation_cfg, output / "dev_generation", pool=pool,
                limit=cfg.validation.generation_limit, split="dev")
        finally:
            pool.close()
    return {"output_dir": str(output), "adapter": str(final_adapter),
            "merged": str(output / cfg.export.merged_subdir) if cfg.export.merge else None,
            "metrics": metrics, "dev_generation": generation_metrics,
            }
