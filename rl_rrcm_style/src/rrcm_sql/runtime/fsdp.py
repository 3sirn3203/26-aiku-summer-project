"""Two-rank FSDP updater with rollout on dedicated HF worker devices."""
from copy import deepcopy
from functools import partial
import json
import multiprocessing as python_mp
import random
import socket
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from torch.distributed.fsdp import (
    FullOptimStateDictConfig, FullStateDictConfig, FullyShardedDataParallel as FSDP,
    MixedPrecision, ShardingStrategy, StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

from ..config import Config
from ..data import load_split, schema_map, official_manifest
from ..model import action_log_probs, load_model
from ..sql import Executor, Judge
from ..train import advantages, append_json, exploration_summary, grpo_terms, seed_all, summarize, update_rl_amp_state
from .rollout_pool import RolloutPool


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wrap_policy(model, cfg):
    if cfg.model.mode == "lora":
        try:
            from peft.utils.other import fsdp_auto_wrap_policy
            return fsdp_auto_wrap_policy(model), False
        except ImportError as exc:
            raise RuntimeError("PEFT with FSDP requires peft.utils.other.fsdp_auto_wrap_policy") from exc
    names = set(cfg.runtime.fsdp_wrap_classes or getattr(model, "_no_split_modules", []) or [])
    classes = {module.__class__ for module in model.modules() if module.__class__.__name__ in names}
    if classes:
        return partial(transformer_auto_wrap_policy, transformer_layer_cls=classes), True
    return partial(size_based_auto_wrap_policy, min_num_params=1_000_000), True


def _full_state(model):
    options = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, options):
        return model.state_dict()


def _export(directory, model, tokenizer, optimizer, scaler, cfg, state, data_hash):
    directory = Path(directory)
    model_state = _full_state(model)
    optim_state = FSDP.full_optim_state_dict(model, optimizer, rank0_only=True)
    cuda_rng = [None] * dist.get_world_size()
    dist.all_gather_object(cuda_rng, torch.cuda.get_rng_state(torch.cuda.current_device()).cpu())
    if dist.get_rank() != 0:
        return None
    directory.mkdir(parents=True, exist_ok=False)
    export_cfg = deepcopy(cfg.model)
    export_cfg.device = "cpu"
    from ..validation import preserve_inference_state
    with preserve_inference_state():
        exported, _ = load_model(export_cfg, trainable=False)
    incompatible = exported.load_state_dict(model_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"FSDP export mismatch: missing={incompatible.missing_keys[:5]}, "
                           f"unexpected={incompatible.unexpected_keys[:5]}")
    exported.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    cfg.save(directory / "run_config.json")
    torch.save({"optimizer": optim_state, "scaler": scaler.state_dict(),
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": cuda_rng, "data_hash": data_hash,
                "backend": "fsdp", **state}, directory / "training_state.pt")
    (directory / "complete.json").write_text(json.dumps(state) + "\n")
    return model_state


def _sync_pool(model, pool, cfg, version):
    state = _full_state(model)
    if dist.get_rank() == 0:
        if cfg.model.mode == "lora":
            state = {name: value for name, value in state.items()
                     if ".lora_" in name or ".modules_to_save." in name}
        pool.sync_state(state, version)


def _optimizer_step(model, optimizer, scaler, max_norm, device):
    scaler.unscale_(optimizer)
    norm = model.clip_grad_norm_(max_norm)
    bad = torch.tensor([not torch.isfinite(norm)], dtype=torch.int32, device=device)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    before = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
    if bad.item():
        optimizer.zero_grad(set_to_none=True)
        if not scaler.is_enabled():
            raise FloatingPointError("Nonfinite distributed RL gradient")
        scaler.update(new_scale=max(before / 2, 1.0))
        after, skipped = float(scaler.get_scale()), True
    else:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        after, skipped = (float(scaler.get_scale()) if scaler.is_enabled() else 1.0), False
    return {"grad_norm": float(norm), "amp_step_skipped": skipped,
            "loss_scale_before": before, "loss_scale_after": after}


def _rank_main(rank, cfg, resume, port, result_queue):
    device = cfg.runtime.update_devices[rank]
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=cfg.evaluation.timeout_seconds))
    pool = None
    from ..tracking import ExperimentTracker
    from ..validation import Validation
    tracker = ExperimentTracker()
    try:
        seed_all(cfg.train.seed)
        rows, schemas = load_split(cfg.data, "train"), schema_map(cfg.data.tables)
        data_hash = __import__("hashlib").sha256(
            json.dumps({"rows": rows, "schemas": schemas}, sort_keys=True).encode()).hexdigest()
        model_cfg = deepcopy(cfg.model)
        model_cfg.device = "cpu"
        model, tokenizer = load_model(model_cfg, checkpoint=resume)
        auto_wrap, use_orig = _wrap_policy(model, cfg)
        dtype = getattr(torch, cfg.model.dtype, None)
        mixed = (MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
                 if dtype in {torch.float16, torch.bfloat16} else None)
        model = FSDP(model, auto_wrap_policy=auto_wrap, device_id=torch.device(device),
                     sharding_strategy=ShardingStrategy.FULL_SHARD, mixed_precision=mixed,
                     limit_all_gathers=True, use_orig_params=use_orig, sync_module_states=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.model.dtype == "float16")
        state = {"step": 0, "groups": 0, "equal_groups": 0,
                 "amp_skipped_updates": 0, "consecutive_amp_skips": 0,
                 "validation": {}, "tracking": {}}
        if resume:
            saved = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=False)
            if saved.get("backend") != "fsdp":
                raise ValueError("Exact FSDP resume requires an FSDP checkpoint; use it as a model warm-start instead")
            if saved["data_hash"] != data_hash:
                raise ValueError("Dataset/schema changed since checkpoint")
            full_optim = saved["optimizer"] if rank == 0 else None
            sharded = FSDP.scatter_full_optim_state_dict(full_optim, model, optim=optimizer)
            optimizer.load_state_dict(sharded)
            scaler.load_state_dict(saved["scaler"])
            state = {key: saved.get(key, value) for key, value in state.items()}
            random.setstate(saved["python_rng"])
            torch.set_rng_state(saved["torch_rng"])
            if saved.get("cuda_rng"):
                torch.cuda.set_rng_state(saved["cuda_rng"][rank], device=device)
        output = Path(cfg.train.output_dir)
        local_trainable = torch.tensor(
            [sum(p.numel() for p in model.parameters() if p.requires_grad)],
            dtype=torch.int64, device=device)
        dist.all_reduce(local_trainable)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            if not resume and any(output.iterdir()):
                raise ValueError(f"Output directory is not empty: {output}")
            cfg.save(output / "config.json")
            if cfg.data.split_mode == "official":
                (output / "data_manifest.json").write_text(json.dumps(official_manifest(cfg), indent=2) + "\n")
            tracker.start(cfg, output, state, resume=resume)
            (output / "runtime.json").write_text(json.dumps({
                "torch": torch.__version__, "update_backend": "fsdp",
                "update_devices": cfg.runtime.update_devices,
                "rollout_devices": cfg.runtime.rollout_devices,
                "reference_device": cfg.model.reference_device if cfg.train.kl_coefficient else None,
                "trainable_parameter_shards_total": int(local_trainable.item()),
                "world_size": 2,
            }, indent=2) + "\n")
            pool = RolloutPool(cfg, cfg.runtime.rollout_devices)
        dist.barrier()
        _sync_pool(model, pool, cfg, state["step"])
        reference = reference_cfg = None
        if rank == 0 and cfg.train.kl_coefficient:
            reference_cfg = deepcopy(cfg.model)
            reference_cfg.device = cfg.model.reference_device or "cpu"
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(cfg.train.seed)
                reference, _ = load_model(reference_cfg, trainable=False)
            reference.eval()
        validation = Validation(cfg, output, state, tracker) if rank == 0 else None

        def validate(phase):
            decision = [None]
            if rank == 0:
                try:
                    improved = validation.due(phase) and validation.run(pool=pool)
                    decision[0] = {"improved": improved, "validation": state["validation"],
                                   "tracking": state["tracking"]}
                except Exception as exc:
                    decision[0] = {"error": f"{type(exc).__name__}: {exc}"}
            dist.broadcast_object_list(decision, src=0, device=torch.device(device))
            if "error" in decision[0]:
                raise RuntimeError(f"Distributed validation failed: {decision[0]['error']}")
            state["validation"] = decision[0]["validation"]
            state["tracking"] = decision[0]["tracking"]
            if rank == 0:
                validation.record = state["validation"]
            if decision[0]["improved"]:
                _export(state["validation"]["best_checkpoint"], model, tokenizer,
                        optimizer, scaler, cfg, state, data_hash)
                if rank == 0:
                    validation.publish_best()

        validate("start")
        while state["step"] < cfg.train.max_steps and state["groups"] < cfg.train.max_groups:
            payload = [None]
            if rank == 0:
                example = random.choice(rows)
                group_id = state["groups"] + 1
                group = pool.generate(example, schemas[example["db_id"]], group_id, state["step"])
                if reference is not None:
                    for trajectory in group:
                        for turn in trajectory.turns:
                            with torch.no_grad():
                                turn.reference_log_probs = action_log_probs(
                                    reference, turn, cfg.rollout.temperature,
                                    reference_cfg).detach().cpu().tolist()
                payload[0] = group
            dist.broadcast_object_list(payload, src=0, device=torch.device(device))
            group = payload[0]
            state["groups"] += 1
            adv = advantages([t.reward for t in group])
            state["equal_groups"] += int(adv is None)
            if rank == 0:
                for index, trajectory in enumerate(group):
                    append_json(output / "trajectories.jsonl", {"group": state["groups"], "sample": index,
                                "step": state["step"], **trajectory.record()})
                metrics = {**state, **summarize(group), **exploration_summary(group),
                           "all_equal_reward_group_rate": state["equal_groups"] / state["groups"],
                           "max_intermediate_reached_rate": sum(
                               len(t.intermediate) == cfg.rollout.max_intermediate for t in group) / len(group)}
                append_json(output / "rollout_metrics.jsonl", metrics)
                tracker.log("rollout", metrics, state)
                print(json.dumps(metrics), flush=True)
            if adv is None:
                continue
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_total = kl_total = 0.0
            for trajectory, advantage in zip(group, adv.tolist()):
                tokens = sum(len(turn.action_ids) for turn in trajectory.turns)
                if not tokens:
                    continue
                denominator = cfg.rollout.group_size * tokens
                for turn in trajectory.turns:
                    current = action_log_probs(model, turn, cfg.rollout.temperature, cfg.model)
                    old = torch.tensor(turn.old_log_probs, device=current.device)
                    ref = (torch.tensor(turn.reference_log_probs, device=current.device)
                           if turn.reference_log_probs is not None else None)
                    terms, kl = grpo_terms(current, old, ref, advantage,
                                           cfg.train.clip_ratio, cfg.train.kl_coefficient)
                    loss = terms.sum() / denominator
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite distributed GRPO loss")
                    scaler.scale(loss).backward()
                    loss_total += loss.item()
                    kl_total += kl.detach().sum().item() / denominator
            update = _optimizer_step(model, optimizer, scaler, cfg.train.max_grad_norm,
                                     torch.device(device))
            applied = update_rl_amp_state(state, update)
            if rank == 0:
                metric = {**state, "loss": loss_total, "kl": kl_total, **update,
                          "optimizer_update_applied": applied, "update_epoch": 0}
                append_json(output / "train_metrics.jsonl", metric)
                tracker.log("train", metric, state)
                print(json.dumps(metric), flush=True)
            if update["amp_step_skipped"] and state["consecutive_amp_skips"] >= cfg.train.max_consecutive_amp_skips:
                raise FloatingPointError("Too many consecutive distributed AMP skips")
            if applied:
                _sync_pool(model, pool, cfg, state["step"])
                validate("interval")
            if applied and state["step"] % cfg.train.save_steps == 0:
                _export(output / f"checkpoint-{state['step']}", model, tokenizer,
                        optimizer, scaler, cfg, state, data_hash)
        validate("end")
        final = output / f"final-step-{state['step']}-groups-{state['groups']}"
        _export(final, model, tokenizer, optimizer, scaler, cfg, state, data_hash)
        if rank == 0:
            status = {**state, "checkpoint": str(final),
                      "max_steps_reached": state["step"] >= cfg.train.max_steps,
                      "update_backend": "fsdp"}
            (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
            tracker.summary({"final_step": state["step"], "final_checkpoint": str(final), **state["validation"]})
            tracker.checkpoint(final, "final")
            result_queue.put(status)
    finally:
        import sys
        tracker.finish(failed=sys.exc_info()[0] is not None)
        if pool:
            pool.close()
        dist.destroy_process_group()


def launch_fsdp(cfg: Config, resume=None):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("FSDP backend requires two visible CUDA GPUs")
    context = python_mp.get_context("spawn")
    result = context.Queue()
    torch_mp.spawn(_rank_main, args=(cfg, resume, _free_port(), result), nprocs=2, join=True)
    return result.get()
