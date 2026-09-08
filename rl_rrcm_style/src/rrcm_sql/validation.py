"""Validation scheduling and checkpoint selection at completed batch boundaries."""
from contextlib import contextmanager
import json
import random
import time
from pathlib import Path

import torch

from .data import split_config


@contextmanager
def preserve_inference_state(model=None):
    rng = random.getstate()
    modes = [(m, m.training) for m in model.modules()] if model is not None else []
    try:
        # CPU workers must not initialize CUDA contexts on the updater GPUs.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            if model is not None:
                model.eval()
            with torch.no_grad():
                yield
    finally:
        random.setstate(rng)
        for module, training in modes:
            module.training = training


class Validation:
    def __init__(self, cfg, output, state, tracker):
        self.cfg, self.output, self.state, self.tracker = cfg, Path(output), state, tracker
        self.record = state.setdefault("validation", {})
        if cfg.evaluation.enabled:
            from dataclasses import asdict
            import hashlib
            from .data import load_split, schema_map
            signature = hashlib.sha256(json.dumps({
                "rows": load_split(cfg.data, "dev"), "schemas": schema_map(cfg.data.tables),
                "evaluation": asdict(cfg.evaluation), "sql": asdict(cfg.sql),
            }, sort_keys=True).encode()).hexdigest()
            previous = self.record.get("signature", signature)
            if previous != signature:
                raise ValueError("Validation data/settings changed; start a new experiment")
            self.record["signature"] = signature
        if self.record.get("best_checkpoint"):
            if not (Path(self.record["best_checkpoint"]) / "complete.json").is_file():
                raise ValueError("Previously selected best checkpoint is missing; restore it before resuming")
            self.write_record()

    def due(self, phase):
        e, step = self.cfg.evaluation, self.state["step"]
        last = self.record.get("last_step", -1)
        if not e.enabled or last == step:
            return False
        if phase == "start":
            return e.at_start and last < 0
        if phase == "end":
            return e.at_end
        return step // e.every_steps > max(last, 0) // e.every_steps

    def run(self, policy=None, pool=None):
        from .evaluate import evaluate_policy
        from .train import append_json
        cfg = split_config(self.cfg, "dev")
        step = self.state["step"]
        directory = self.output / "evaluations" / f"dev-step-{step}"
        # An interrupted/branched run can have an earlier evaluation at this step.
        attempt = 0
        while directory.exists():
            attempt += 1
            directory = self.output / "evaluations" / f"dev-step-{step}-attempt-{attempt}"
        started = time.monotonic()
        with preserve_inference_state(policy.model if policy is not None else None):
            summary = evaluate_policy(cfg, directory, policy=policy, pool=pool,
                                      policy_version=step, limit=cfg.evaluation.limit)
        summary["seconds"] = time.monotonic() - started
        score = summary[cfg.evaluation.selection_metric]
        if score is None:
            raise ValueError("Checkpoint selection metric unavailable")
        old_hash = self.record.get("data_sha256")
        if old_hash and old_hash != summary["data_sha256"]:
            raise ValueError("Validation data changed since checkpoint; start a new experiment")
        self.record.update(last_step=step, data_sha256=summary["data_sha256"])
        improved = score > self.record.get("best_score", float("-inf"))
        if improved:
            self.record.update(best_score=score, best_step=step,
                               metric=cfg.evaluation.selection_metric,
                               best_checkpoint=str((self.output / f"best-step-{step}").resolve()))
        append_json(self.output / "dev_metrics.jsonl", {"step": step, **summary})
        self.tracker.log("dev", summary, self.state)
        self.tracker.evaluation_files(directory, "dev")
        return improved

    def publish_best(self):
        # Publish only after a complete checkpoint has been saved successfully.
        self.write_record()
        self.tracker.summary({"best_dev_score": self.record["best_score"],
                              "best_step": self.record["best_step"],
                              "best_checkpoint": self.record["best_checkpoint"]})
        self.tracker.checkpoint(self.record["best_checkpoint"], "best")

    def write_record(self):
        path = self.output / "best_checkpoint.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.record, indent=2) + "\n")
        temporary.replace(path)
