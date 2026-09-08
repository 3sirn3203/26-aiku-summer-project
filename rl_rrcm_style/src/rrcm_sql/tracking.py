"""Optional experiment tracking; local metrics remain the source of record."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import hashlib
import time
from importlib.metadata import version


class ExperimentTracker:
    def __init__(self):
        self.run = None
        self.cfg = None
        self.started = time.monotonic()

    def start(self, cfg, output, state, resume=None, job_type="train", checkpoint=None):
        from .validation import preserve_inference_state
        with preserve_inference_state():
            self._start(cfg, output, state, resume, job_type, checkpoint)

    def _start(self, cfg, output, state, resume, job_type, checkpoint):
        self.cfg = cfg
        t = cfg.tracking
        if t.backend == "none" or t.mode == "disabled":
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("Install tracking support: pip install -e 'rl_rrcm_style[wandb]'") from exc
        parent = state.get("tracking", {}).get("run_id")
        resume_run = t.resume_run and job_type == "train"
        if resume_run and (not resume or not parent or t.mode != "online"):
            raise ValueError("tracking.resume_run requires an online run and checkpoint with a W&B run ID")
        if resume_run and state["tracking"].get("project") != t.project:
            raise ValueError("Resuming W&B requires the original project")
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        sources = sorted(Path(__file__).parent.rglob("*.py"))
        source_hash = hashlib.sha256(b"".join(str(p.relative_to(Path(__file__).parent)).encode() + p.read_bytes()
                                            for p in sources)).hexdigest()
        self.run = wandb.init(
            project=t.project, entity=t.entity, name=t.run_name, group=t.group,
            tags=t.tags, mode=t.mode, dir=str(output), job_type=job_type,
            id=parent if resume_run else None, resume="must" if resume_run else None,
            config={**asdict(cfg), "git_revision": revision.stdout.strip(),
                    "git_dirty": bool(dirty.stdout.strip()), "source_sha256": source_hash,
                    "torch_version": version("torch"), "transformers_version": version("transformers"),
                    "parent_run_id": parent,
                    "parent_run": dict(state.get("tracking", {})),
                    "source_checkpoint": str(checkpoint or resume) if checkpoint or resume else None})
        self.run.define_metric("train/*", step_metric="optimizer_step")
        self.run.define_metric("dev/*", step_metric="optimizer_step")
        self.run.define_metric("rollout/*", step_metric="group_step")
        if resume_run and state.get("step", 0) < self.run.summary.get("latest_optimizer_step", 0):
            raise ValueError("Checkpoint predates W&B history; disable resume_run to create a linked branch")
        state["tracking"] = {"run_id": self.run.id, "project": t.project, "entity": self.run.entity}
        (Path(output) / "tracking.json").write_text(json.dumps(state["tracking"], indent=2) + "\n")
        manifest = Path(output) / "data_manifest.json"
        if manifest.exists():
            self.run.config.update({"data_manifest": json.loads(manifest.read_text())})

    def log(self, prefix, metrics, state):
        if self.run:
            values = {f"{prefix}/{k}": v for k, v in metrics.items()
                      if isinstance(v, (int, float, bool)) and k not in {"step", "groups"}}
            self.run.log({"optimizer_step": state.get("step", 0),
                          "group_step": state.get("groups", 0),
                          "session_elapsed_seconds": time.monotonic() - self.started, **values})
            self.run.summary["latest_optimizer_step"] = state.get("step", 0)

    def summary(self, values):
        if self.run:
            self.run.summary.update(values)

    def evaluation_files(self, directory, split):
        if not self.run or not self.cfg.tracking.log_trajectories:
            return
        import wandb
        artifact = wandb.Artifact(f"{self.run.id}-{split}-{Path(directory).name}", type="evaluation")
        artifact.add_dir(str(directory))
        self.run.log_artifact(artifact)

    def checkpoint(self, path, alias):
        if not self.run or not self.cfg.tracking.upload_checkpoints:
            return
        import wandb
        artifact = wandb.Artifact(f"{self.run.id}-model", type="model")
        # Optimizer/RNG pickle files are local resume state, not model artifacts.
        for file in Path(path).iterdir():
            if file.is_file() and file.name != "training_state.pt":
                artifact.add_file(str(file))
        self.run.log_artifact(artifact, aliases=[alias])

    def finish(self, failed=False):
        if self.run:
            self.run.finish(exit_code=1 if failed else 0)
            self.run = None
