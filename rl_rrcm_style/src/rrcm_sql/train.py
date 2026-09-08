from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from contextlib import ExitStack

import torch

from .data import load_split, schema_map, official_manifest
from .model import HFPolicy, Turn, action_log_probs, load_model
from .rollout import initial_messages, rollout
from .exploration import group_modes, task_seed
from .sql import Executor, Judge


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def advantages(rewards):
    values = torch.tensor(rewards, dtype=torch.float32)
    std = values.std(unbiased=False)
    return None if std.item() < 1e-8 else (values - values.mean()) / (std + 1e-8)


def grpo_terms(current, old, reference, advantage, clip_ratio, kl_coefficient):
    log_ratio = current - old
    ratio = log_ratio.exp()
    surrogate = torch.minimum(ratio * advantage,
                              ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage)
    if reference is None:
        kl = torch.zeros_like(current)
    else:
        delta = reference - current
        kl = delta.exp() - delta - 1
    return -surrogate + kl_coefficient * kl, kl


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def summarize(trajectories):
    n = len(trajectories)
    correct = [t for t in trajectories if t.outcome == "correct"]
    calls = sum(len(t.intermediate) for t in trajectories)
    queries = [[" ".join(q["sql"].lower().split()) for q in t.intermediate] for t in trajectories]
    return {
        "count": n,
        "execution_accuracy": sum(t.scores.get("execution_correct", False) for t in trajectories) / n,
        "exact_match": sum(t.scores.get("exact_match", False) for t in trajectories) / n,
        **{f"{kind}_rate": sum(t.outcome == kind for t in trajectories) / n
           for kind in ("correct", "executable_incorrect", "non_executable")},
        "reward_mean": sum(t.reward for t in trajectories) / n,
        **{f"reward_{key}_mean": sum(t.reward_components.get(key, 0.0)
                                      for t in trajectories) / n
           for key in ("execution_reward", "exact_match_bonus", "intermediate_penalty",
                       "failure_penalty")},
        "intermediate_mean": calls / n,
        "correct_intermediate_mean": (sum(len(t.intermediate) for t in correct) / len(correct) if correct else None),
        "direct_answer_rate": sum(t.termination == "answer" and not t.intermediate for t in trajectories) / n,
        "no_intermediate_rate": sum(not t.intermediate for t in trajectories) / n,
        "llm_calls_mean": sum(len(t.turns) for t in trajectories) / n,
        "input_tokens_mean": sum(t.record()["input_tokens"] for t in trajectories) / n,
        "output_tokens_mean": sum(t.record()["output_tokens"] for t in trajectories) / n,
        "input_tokens_total": sum(t.record()["input_tokens"] for t in trajectories),
        "output_tokens_total": sum(t.record()["output_tokens"] for t in trajectories),
        "invalid_intermediate_rate": (sum(not q["result"]["ok"] for t in trajectories for q in t.intermediate) / calls if calls else 0),
        "duplicate_intermediate_rate": sum(len(q) - len(set(q)) for q in queries) / calls if calls else 0,
        "sql_seconds_mean": sum(sum(q["result"].get("elapsed", 0) for q in t.intermediate)
                                + t.scores.get("final_execution", {}).get("elapsed", 0) for t in trajectories) / n,
    }


def exploration_summary(trajectories):
    result = {}
    for mode in ("free", "prompted_random"):
        selected = [t for t in trajectories if t.mode == mode]
        if not selected:
            continue
        for key, value in summarize(selected).items():
            result[f"{mode}_{key}"] = value
        requested = [a for t in selected for a in t.action_trace if a["requested"]]
        result[f"{mode}_instruction_compliance_rate"] = (
            sum(a["requested"] == a["actual"] for a in requested) / len(requested) if requested else None)
    return result


def local_group(policy, example, schema, executor, judge, cfg, group_id, policy_version):
    trajectories = []
    for index, mode in enumerate(group_modes(cfg.rollout)):
        trajectories.append(rollout(
            policy, example, schema, executor, judge, cfg.rollout, mode=mode,
            seed=task_seed(cfg.train.seed, group_id, index), policy_version=policy_version))
    return trajectories


def save_checkpoint(directory, model, tokenizer, optimizer, scaler, cfg, state, data_hash):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    cfg.save(directory / "run_config.json")
    torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "data_hash": data_hash, "backend": "single", **state}, directory / "training_state.pt")
    # Written last: interrupted saves are not resumable checkpoints.
    (directory / "complete.json").write_text(json.dumps(state) + "\n")


def training_setup(cfg, resume=None):
    seed_all(cfg.train.seed)
    rows = load_split(cfg.data, "train")
    schemas = schema_map(cfg.data.tables)
    model, tokenizer = load_model(cfg.model, checkpoint=resume)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.model.dtype == "float16" and
                                  next(model.parameters()).device.type == "cuda")
    output = Path(cfg.train.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Use a new run directory.")
    cfg.save(output / "config.json")
    if cfg.data.split_mode == "official":
        (output / "data_manifest.json").write_text(json.dumps(official_manifest(cfg), indent=2) + "\n")
    data_hash = hashlib.sha256(json.dumps({"rows": rows, "schemas": schemas}, sort_keys=True).encode()).hexdigest()
    state = {"step": 0, "groups": 0, "equal_groups": 0,
             "amp_skipped_updates": 0, "consecutive_amp_skips": 0,
             "validation": {}, "tracking": {}}
    if resume:
        if not (Path(resume) / "complete.json").exists():
            raise ValueError("Checkpoint is incomplete")
        # Only load training_state.pt from your own trusted checkpoints (pickle).
        saved = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=False)
        if saved.get("backend", "single") != "single":
            raise ValueError("Single-GPU exact resume requires a single-GPU checkpoint")
        if saved["data_hash"] != data_hash:
            raise ValueError("Dataset/schema changed since checkpoint")
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        state = {key: saved.get(key, value) for key, value in state.items()}
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return rows, schemas, model, tokenizer, optimizer, scaler, output, data_hash, state


def optimizer_step(model, optimizer, scaler, max_norm):
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm)
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite gradient norm; reduce learning rate or use float32/bfloat16")
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def rl_optimizer_step(model, optimizer, scaler, max_norm):
    """Apply one RL update, allowing GradScaler to recover from FP16 overflow."""
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm)
    scale_before = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
    if not torch.isfinite(norm) and not scaler.is_enabled():
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Nonfinite RL gradient norm without AMP recovery")
    # GradScaler skips optimizer.step when unscale_ found Inf/NaN, then lowers its scale.
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
    optimizer.zero_grad(set_to_none=True)
    return {"grad_norm": float(norm), "amp_step_skipped": scale_after < scale_before,
            "loss_scale_before": scale_before, "loss_scale_after": scale_after}


def update_rl_amp_state(state, result):
    if result["amp_step_skipped"]:
        state["amp_skipped_updates"] += 1
        state["consecutive_amp_skips"] += 1
        return False
    state["consecutive_amp_skips"] = 0
    state["step"] += 1
    return True


def train(cfg, resume=None):
    if cfg.runtime.update_backend == "fsdp":
        from .runtime.fsdp import launch_fsdp
        return launch_fsdp(cfg, resume)
    from .tracking import ExperimentTracker
    tracker = ExperimentTracker()
    try:
        with ExitStack() as cleanup:
            return _train_single(cfg, resume, tracker, cleanup)
    except torch.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(
            "Single-GPU RL update ran out of CUDA memory. Start a new run with "
            "runtime.update_backend=fsdp, two update_devices, and dedicated rollout_devices; "
            "a complete prior checkpoint can be supplied with --model as a warm-start.") from exc
    finally:
        import sys
        tracker.finish(failed=sys.exc_info()[0] is not None)


def _train_single(cfg, resume=None, tracker=None, cleanup=None):
    rows, schemas, model, tokenizer, optimizer, scaler, output, data_hash, state = training_setup(cfg, resume)
    tracker.start(cfg, output, state, resume=resume)
    if cfg.sql.evaluator_path is None:
        print("Correctness backend: strict_execution_proxy (not official Spider evaluation).", flush=True)
    policy = HFPolicy(model, tokenizer, cfg.model, cfg.rollout)
    executor = Executor(cfg.sql)
    judge = Judge(executor, cfg.data.database_dir, cfg.data.tables)
    pool = None
    if cfg.runtime.rollout_devices:
        from .runtime.rollout_pool import RolloutPool
        pool = RolloutPool(cfg, cfg.runtime.rollout_devices)
        cleanup.callback(pool.close)
        pool.sync(model, policy_version=state["step"])
    reference = None
    if cfg.train.kl_coefficient:
        reference_cfg = deepcopy(cfg.model)
        reference_cfg.device = cfg.model.reference_device or cfg.model.device
        # Always reconstruct the initial policy, including SFT adapter, not the resumed RL policy.
        # fork_rng keeps initialization from perturbing rollout RNG on restart.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(cfg.train.seed)
            reference, _ = load_model(reference_cfg, trainable=False)
        reference.eval()
    (output / "runtime.json").write_text(json.dumps({
        "torch": torch.__version__, "device": str(next(model.parameters()).device),
        "update_backend": "single", "rollout_devices": cfg.runtime.rollout_devices,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "correctness": "spider_result_eq" if cfg.sql.evaluator_path else "strict_execution_proxy",
    }, indent=2) + "\n")
    from .validation import Validation
    validation = Validation(cfg, output, state, tracker)

    def validate(phase):
        if validation.due(phase) and validation.run(policy=policy if pool is None else None, pool=pool):
            save_checkpoint(validation.record["best_checkpoint"], model, tokenizer, optimizer,
                            scaler, cfg, state, data_hash)
            validation.publish_best()

    validate("start")
    while state["step"] < cfg.train.max_steps and state["groups"] < cfg.train.max_groups:
        batch = []
        while len(batch) < cfg.train.groups_per_update and state["groups"] < cfg.train.max_groups:
            example = random.choice(rows)
            group_id = state["groups"] + 1
            group = (pool.generate(example, schemas[example["db_id"]], group_id, state["step"])
                     if pool else local_group(policy, example, schemas[example["db_id"]], executor,
                                              judge, cfg, group_id, state["step"]))
            state["groups"] += 1
            adv = advantages([t.reward for t in group])
            state["equal_groups"] += int(adv is None)
            for index, trajectory in enumerate(group):
                append_json(output / "trajectories.jsonl", {"group": state["groups"], "sample": index,
                            "step": state["step"], **trajectory.record()})
            metrics = {**state, **summarize(group), **exploration_summary(group),
                       "all_equal_reward_group_rate": state["equal_groups"] / state["groups"],
                       "max_intermediate_reached_rate": sum(len(t.intermediate) == cfg.rollout.max_intermediate for t in group) / len(group)}
            append_json(output / "rollout_metrics.jsonl", metrics)
            tracker.log("rollout", metrics, state)
            print(json.dumps(metrics), flush=True)
            if adv is None:
                continue
            cached = []
            model.eval()
            for trajectory, advantage in zip(group, adv.tolist()):
                turns = []
                for turn in trajectory.turns:
                    with torch.no_grad():
                        old = (torch.tensor(turn.old_log_probs) if turn.old_log_probs is not None else
                               action_log_probs(model, turn, cfg.rollout.temperature, cfg.model).detach().cpu())
                        ref = (torch.tensor(turn.reference_log_probs) if turn.reference_log_probs is not None else
                               action_log_probs(reference, turn, cfg.rollout.temperature, reference_cfg).detach().cpu()
                               if reference is not None else None)
                    turns.append((turn, old, ref))
                cached.append((trajectory, advantage, turns))
            batch.append(cached)
        if not batch:
            break
        batch_changed = False
        for epoch in range(cfg.train.update_epochs):
            update_device = next(model.parameters()).device
            if update_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(update_device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_total, kl_total = 0.0, 0.0
            for group in batch:
                for trajectory, advantage, turns in group:
                    tokens = sum(len(turn.action_ids) for turn, _, _ in turns)
                    if not tokens:
                        continue
                    denominator = len(batch) * cfg.rollout.group_size * tokens
                    # Backprop one turn at a time to avoid retaining all trajectory graphs.
                    for turn, old, ref in turns:
                        current = action_log_probs(model, turn, cfg.rollout.temperature, cfg.model)
                        terms, kl = grpo_terms(current, old.to(current.device),
                                               ref.to(current.device) if ref is not None else None,
                                               advantage, cfg.train.clip_ratio, cfg.train.kl_coefficient)
                        loss = terms.sum() / denominator
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite GRPO loss")
                        scaler.scale(loss).backward()
                        loss_total += loss.item()
                        kl_total += kl.detach().sum().item() / denominator
            update = rl_optimizer_step(model, optimizer, scaler, cfg.train.max_grad_norm)
            applied = update_rl_amp_state(state, update)
            batch_changed |= applied
            metric = {**state, "loss": loss_total, "kl": kl_total, **update,
                      "optimizer_update_applied": applied, "update_epoch": epoch}
            if update_device.type == "cuda":
                metric.update({
                    "update_memory_allocated_mb": torch.cuda.memory_allocated(update_device) / 2**20,
                    "update_memory_reserved_mb": torch.cuda.memory_reserved(update_device) / 2**20,
                    "update_peak_memory_mb": torch.cuda.max_memory_allocated(update_device) / 2**20,
                })
            append_json(output / "train_metrics.jsonl", metric)
            tracker.log("train", metric, state)
            print(json.dumps(metric), flush=True)
            if (update["amp_step_skipped"] and state["consecutive_amp_skips"] >=
                    cfg.train.max_consecutive_amp_skips):
                raise FloatingPointError(
                    f"AMP skipped {state['consecutive_amp_skips']} consecutive RL updates; "
                    "reduce learning rate or inspect loss/log probabilities")
            if state["step"] >= cfg.train.max_steps:
                break
        if pool and batch_changed:
            pool.sync(model, policy_version=state["step"])
        if batch_changed:
            validate("interval")
        # Save only at rollout-batch boundaries; no on-policy cached batch is lost on resume.
        if batch_changed and state["step"] % cfg.train.save_steps == 0:
            save_checkpoint(output / f"checkpoint-{state['step']}", model, tokenizer, optimizer,
                            scaler, cfg, state, data_hash)
    validate("end")
    final = output / f"final-step-{state['step']}-groups-{state['groups']}"
    save_checkpoint(final, model, tokenizer, optimizer, scaler, cfg, state, data_hash)
    status = {**state, "checkpoint": str(final), "max_steps_reached": state["step"] >= cfg.train.max_steps}
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    tracker.summary({"final_step": state["step"], "final_checkpoint": str(final), **state["validation"]})
    tracker.checkpoint(final, "final")
    if not status["max_steps_reached"]:
        print("Group budget exhausted before max_steps; inspect all-equal rate / SFT warm-start.", flush=True)
    if pool:
        pool.close()
    return status


def sft(cfg):
    from .tracking import ExperimentTracker
    tracker = ExperimentTracker()
    try:
        return _sft(cfg, tracker)
    finally:
        import sys
        tracker.finish(failed=sys.exc_info()[0] is not None)


def _sft(cfg, tracker):
    rows, schemas, model, tokenizer, optimizer, scaler, output, data_hash, state = training_setup(cfg)
    tracker.start(cfg, output, state, job_type="sft")
    policy = HFPolicy(model, tokenizer, cfg.model, cfg.rollout)
    from .validation import Validation
    validation = Validation(cfg, output, state, tracker)

    def validate(phase):
        if validation.due(phase) and validation.run(policy=policy):
            save_checkpoint(validation.record["best_checkpoint"], model, tokenizer, optimizer,
                            scaler, cfg, state, data_hash)
            validation.publish_best()

    validate("start")
    skipped = 0
    for epoch in range(cfg.train.sft_epochs):
        indices = list(range(len(rows)))
        random.shuffle(indices)
        for index in indices:
            example = rows[index]
            prompt = policy.encode(initial_messages(example, schemas[example["db_id"]], cfg.rollout.max_intermediate))
            text = f"<answer>\n{example['query']}\n</answer>"
            action = tokenizer.encode(text, add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                action.append(tokenizer.eos_token_id)
            capacity = min(cfg.rollout.max_context_tokens,
                           getattr(model.config, "max_position_embeddings", cfg.rollout.max_context_tokens))
            if len(prompt) + len(action) > capacity:
                skipped += 1
                continue
            model.train()
            loss = -action_log_probs(model, Turn(prompt, action, text), 1.0, cfg.model).mean()
            scaler.scale(loss).backward()
            optimizer_step(model, optimizer, scaler, cfg.train.max_grad_norm)
            state["step"] += 1
            append_json(output / "sft_metrics.jsonl", {"step": state["step"], "epoch": epoch, "loss": loss.item()})
            tracker.log("train", {"loss": loss.item(), "epoch": epoch}, state)
            validate("interval")
            if state["step"] >= cfg.train.max_steps:
                break
        if state["step"] >= cfg.train.max_steps:
            break
    if not state["step"]:
        raise ValueError("No SFT updates; increase context capacity")
    validate("end")
    save_checkpoint(output / "sft-final", model, tokenizer, optimizer, scaler, cfg, state, data_hash)
    tracker.checkpoint(output / "sft-final", "final")
    return {"steps": state["step"], "skipped_overlength": skipped, "checkpoint": str(output / "sft-final")}
