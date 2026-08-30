from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from rl_finetune.agentic_runtime.models import Trajectory
from rl_finetune.agentic_runtime.rollout import WorkflowRolloutRunner
from rl_finetune.trajectory_grpo import (
    backward_trajectory_grpo_group,
    cache_old_and_reference_log_probs,
    normalize_group_advantages,
)
from rl_finetune.trajectory_reward import TrajectoryRewardRuntime


@dataclass(frozen=True)
class AgenticGRPOConfig:
    learning_rate: float = 5e-6
    max_steps: int = 1
    gradient_accumulation_steps: int = 1
    clip_epsilon: float = 0.2
    kl_beta: float = 0.01
    max_grad_norm: float = 1.0
    credit_mode: str = "all_actions"
    max_rollout_attempts_per_step: int = 20
    old_current_tolerance: float = 1e-5
    policy_temperature: float = 0.9

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_steps < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("step counts must be positive")
        if self.clip_epsilon < 0 or self.kl_beta < 0:
            raise ValueError("clip_epsilon and kl_beta must be non-negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.credit_mode not in {"all_actions", "final_only"}:
            raise ValueError("credit_mode must be all_actions or final_only")
        if self.max_rollout_attempts_per_step < 1:
            raise ValueError("max_rollout_attempts_per_step must be positive")
        if self.old_current_tolerance < 0:
            raise ValueError("old_current_tolerance must be non-negative")
        if self.policy_temperature <= 0:
            raise ValueError("policy_temperature must be positive")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


class AgenticGRPOTrainer:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        rollout_runner: WorkflowRolloutRunner,
        reward_runtime: TrajectoryRewardRuntime,
        records: Sequence[Mapping[str, Any]],
        config: AgenticGRPOConfig,
        run_dir: Path,
        device: Any,
        initial_global_step: int = 0,
        optimizer_state_path: Optional[Path] = None,
        scheduler_state_path: Optional[Path] = None,
        initial_record_index: int = 0,
        checkpoint_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        import torch

        if not records:
            raise ValueError("training records must not be empty")
        if initial_record_index < 0:
            raise ValueError("initial_record_index must be non-negative")
        trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise ValueError("model has no trainable LoRA parameters")
        self.model = model
        self.tokenizer = tokenizer
        self.rollout_runner = rollout_runner
        self.reward_runtime = reward_runtime
        self.records = records
        self.config = config
        self.run_dir = Path(run_dir)
        self.device = device
        self.global_step = initial_global_step
        self.record_index = initial_record_index
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self.optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda _step: 1.0
        )
        if optimizer_state_path is not None:
            self.optimizer.load_state_dict(
                torch.load(optimizer_state_path, map_location=device, weights_only=True)
            )
        if scheduler_state_path is not None:
            self.scheduler.load_state_dict(
                torch.load(scheduler_state_path, map_location="cpu", weights_only=True)
            )
        self.trajectory_trace_path = self.run_dir / "trajectory_trace.jsonl"
        self.reward_trace_path = self.run_dir / "reward_trace.jsonl"

    def train(self, *, write_trajectory_trace: bool = True) -> dict[str, Any]:
        import torch

        if self.global_step >= self.config.max_steps:
            raise ValueError("checkpoint global step already reached --max-steps")
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        accumulated_groups = 0
        attempted_groups = 0
        skipped_gold_groups = 0
        zero_variance_groups = 0
        last_metrics: dict[str, float] = {}
        max_attempts = (
            (self.config.max_steps - self.global_step)
            * self.config.gradient_accumulation_steps
            * self.config.max_rollout_attempts_per_step
        )

        while self.global_step < self.config.max_steps:
            if attempted_groups >= max_attempts:
                raise RuntimeError(
                    "unable to obtain enough non-zero-variance trajectory groups "
                    "within %d attempts" % max_attempts
                )
            record = self.records[self.record_index % len(self.records)]
            self.record_index += 1
            attempted_groups += 1
            trajectories = self.rollout_runner.collect_group(record)
            if not self.reward_runtime.score_group(trajectories, record):
                skipped_gold_groups += 1
                self._write_group_traces(
                    trajectories, write_trajectory_trace=write_trajectory_trace
                )
                continue
            has_signal = normalize_group_advantages(trajectories)
            self._write_group_traces(
                trajectories, write_trajectory_trace=write_trajectory_trace
            )
            if not has_signal:
                zero_variance_groups += 1
                continue

            cache_old_and_reference_log_probs(
                self.model,
                trajectories,
                credit_mode=self.config.credit_mode,
                device=self.device,
                temperature=self.config.policy_temperature,
            )
            metrics = backward_trajectory_grpo_group(
                self.model,
                trajectories,
                credit_mode=self.config.credit_mode,
                clip_epsilon=self.config.clip_epsilon,
                kl_beta=self.config.kl_beta,
                device=self.device,
                loss_scale=1.0 / self.config.gradient_accumulation_steps,
                temperature=self.config.policy_temperature,
            )
            if (
                metrics["max_old_current_log_prob_delta"]
                > self.config.old_current_tolerance
            ):
                raise RuntimeError(
                    "old/current log-probabilities diverged before update: %.8g > %.8g"
                    % (
                        metrics["max_old_current_log_prob_delta"],
                        self.config.old_current_tolerance,
                    )
                )
            gradients_are_finite = all(
                torch.isfinite(parameter.grad).all().item()
                for parameter in self.model.parameters()
                if parameter.grad is not None
            )
            if not gradients_are_finite:
                raise RuntimeError("non-finite gradient detected")
            if not torch.isfinite(torch.tensor(metrics["loss"])):
                raise RuntimeError("non-finite loss detected")
            accumulated_groups += 1
            last_metrics = metrics
            if accumulated_groups < self.config.gradient_accumulation_steps:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm,
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite gradient norm detected")
            base_grad_count = sum(
                1
                for name, parameter in self.model.named_parameters()
                if "lora_" not in name and parameter.grad is not None
            )
            lora_grad_count = sum(
                1
                for name, parameter in self.model.named_parameters()
                if "lora_" in name and parameter.grad is not None
            )
            if base_grad_count:
                raise RuntimeError("base model parameters received gradients")
            if not lora_grad_count:
                raise RuntimeError("LoRA parameters did not receive gradients")
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            accumulated_groups = 0
            last_metrics.update(
                {
                    "gradient_norm": float(grad_norm.item()),
                    "base_gradient_parameter_count": float(base_grad_count),
                    "lora_gradient_parameter_count": float(lora_grad_count),
                }
            )
            self._save_checkpoint(last_metrics)

        return {
            "global_step": self.global_step,
            "record_index": self.record_index,
            "attempted_groups": attempted_groups,
            "skipped_gold_groups": skipped_gold_groups,
            "zero_variance_groups": zero_variance_groups,
            "last_metrics": last_metrics,
        }

    def _write_group_traces(
        self, trajectories: Sequence[Trajectory], *, write_trajectory_trace: bool
    ) -> None:
        for trajectory in trajectories:
            if write_trajectory_trace:
                _append_jsonl(self.trajectory_trace_path, trajectory.to_dict())
            _append_jsonl(
                self.reward_trace_path,
                {
                    "example_id": trajectory.example_id,
                    "generation_index": trajectory.generation_index,
                    "terminal_status": trajectory.terminal_status,
                    "reward": trajectory.reward,
                    "advantage": trajectory.advantage,
                    "reward_result": trajectory.reward_result,
                },
            )

    def _save_checkpoint(self, metrics: Mapping[str, Any]) -> Path:
        import numpy
        import torch

        checkpoint_dir = self.run_dir / "trainer" / (
            "checkpoint-%d" % self.global_step
        )
        if checkpoint_dir.exists():
            raise RuntimeError("checkpoint directory already exists: %s" % checkpoint_dir)
        checkpoint_dir.mkdir(parents=True)
        self.model.save_pretrained(str(checkpoint_dir))
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        rng_state = {
            "python": random.getstate(),
            "numpy": numpy.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        torch.save(rng_state, checkpoint_dir / "rng_state.pt")
        state = {
            "global_step": self.global_step,
            "record_index": self.record_index,
            "config": asdict(self.config),
            "metrics": dict(metrics),
            "checkpoint_metadata": self.checkpoint_metadata,
        }
        (checkpoint_dir / "trainer_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return checkpoint_dir


def restore_rng_state(path: Path) -> None:
    import numpy
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(payload["python"])
    numpy.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and payload.get("cuda"):
        torch.cuda.set_rng_state_all(payload["cuda"])

