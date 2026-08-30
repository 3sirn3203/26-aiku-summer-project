from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from rl_finetune.agentic_runtime.models import RolloutStep, Trajectory


_CREDIT_MODES = {"all_actions", "final_only"}


def selected_steps(trajectory: Trajectory, credit_mode: str) -> list[RolloutStep]:
    if credit_mode not in _CREDIT_MODES:
        raise ValueError("credit_mode must be one of %s" % sorted(_CREDIT_MODES))
    if credit_mode == "all_actions":
        return list(trajectory.steps)
    return [step for step in trajectory.steps if step.role == "final"]


def normalize_group_advantages(
    trajectories: Sequence[Trajectory], *, epsilon: float = 1e-8
) -> bool:
    if not trajectories:
        raise ValueError("trajectory group must not be empty")
    rewards = [trajectory.reward for trajectory in trajectories]
    if any(reward is None or not math.isfinite(reward) for reward in rewards):
        raise ValueError("all trajectories must have finite rewards")
    numeric_rewards = [float(reward) for reward in rewards if reward is not None]
    mean = sum(numeric_rewards) / len(numeric_rewards)
    variance = sum((reward - mean) ** 2 for reward in numeric_rewards) / len(
        numeric_rewards
    )
    standard_deviation = math.sqrt(variance)
    if standard_deviation <= epsilon:
        for trajectory in trajectories:
            trajectory.advantage = 0.0
        return False
    for trajectory, reward in zip(trajectories, numeric_rewards):
        trajectory.advantage = (reward - mean) / standard_deviation
    return True


def action_token_log_probs(
    model: Any,
    context_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    *,
    device: Any,
    temperature: float = 1.0,
) -> Any:
    """Return log-probabilities only for generated tokens with causal shifting."""
    import torch

    if not context_token_ids:
        raise ValueError("context_token_ids must not be empty")
    if not generated_token_ids:
        raise ValueError("generated_token_ids must not be empty")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    all_ids = list(context_token_ids) + list(generated_token_ids)
    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
    logits = model(input_ids=input_ids, use_cache=False).logits[0]
    start = len(context_token_ids) - 1
    action_logits = logits[start : start + len(generated_token_ids)]
    targets = input_ids[0, len(context_token_ids) :]
    return torch.log_softmax(action_logits / temperature, dim=-1).gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)


@contextmanager
def adapters_disabled(model: Any) -> Iterator[None]:
    disable_adapter = getattr(model, "disable_adapter", None)
    if disable_adapter is None:
        raise RuntimeError("PEFT model does not expose disable_adapter()")
    with disable_adapter():
        yield


def cache_old_and_reference_log_probs(
    model: Any,
    trajectories: Sequence[Trajectory],
    *,
    credit_mode: str,
    device: Any,
    temperature: float = 1.0,
) -> None:
    import torch

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for trajectory in trajectories:
                for step in selected_steps(trajectory, credit_mode):
                    step.old_log_probs = action_token_log_probs(
                        model,
                        step.context_token_ids,
                        step.generated_token_ids,
                        device=device,
                        temperature=temperature,
                    ).detach()
            with adapters_disabled(model):
                for trajectory in trajectories:
                    for step in selected_steps(trajectory, credit_mode):
                        step.reference_log_probs = action_token_log_probs(
                            model,
                            step.context_token_ids,
                            step.generated_token_ids,
                            device=device,
                            temperature=temperature,
                        ).detach()
    finally:
        model.train(was_training)


def trajectory_grpo_loss(
    model: Any,
    trajectories: Sequence[Trajectory],
    *,
    credit_mode: str,
    clip_epsilon: float,
    kl_beta: float,
    device: Any,
    temperature: float = 1.0,
) -> tuple[Any, dict[str, float]]:
    import torch

    if clip_epsilon < 0 or kl_beta < 0:
        raise ValueError("clip_epsilon and kl_beta must be non-negative")
    trajectory_losses = []
    max_old_current_delta = 0.0
    token_count = 0
    for trajectory in trajectories:
        if trajectory.advantage is None:
            raise ValueError("trajectory advantage has not been assigned")
        token_losses = []
        advantage = torch.tensor(float(trajectory.advantage), device=device)
        for step in selected_steps(trajectory, credit_mode):
            if step.old_log_probs is None or step.reference_log_probs is None:
                raise ValueError("old/reference log probabilities are missing")
            current = action_token_log_probs(
                model,
                step.context_token_ids,
                step.generated_token_ids,
                device=device,
                temperature=temperature,
            )
            old = step.old_log_probs.to(device)
            reference = step.reference_log_probs.to(device)
            max_old_current_delta = max(
                max_old_current_delta,
                float((current.detach() - old).abs().max().item()),
            )
            ratio = torch.exp(current - old)
            unclipped = ratio * advantage
            clipped = torch.clamp(
                ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
            ) * advantage
            reference_delta = reference - current
            kl = torch.exp(reference_delta) - reference_delta - 1.0
            token_losses.append(-torch.minimum(unclipped, clipped) + kl_beta * kl)
            token_count += int(current.numel())
        if not token_losses:
            raise ValueError("trajectory has no selected generated tokens")
        trajectory_losses.append(torch.cat(token_losses).mean())
    if not trajectory_losses:
        raise ValueError("trajectory group must not be empty")
    loss = torch.stack(trajectory_losses).mean()
    return loss, {
        "max_old_current_log_prob_delta": max_old_current_delta,
        "action_token_count": float(token_count),
    }


def backward_trajectory_grpo_group(
    model: Any,
    trajectories: Sequence[Trajectory],
    *,
    credit_mode: str,
    clip_epsilon: float,
    kl_beta: float,
    device: Any,
    loss_scale: float = 1.0,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Backpropagate one exact group objective one step at a time.

    This is algebraically the same trajectory-then-group mean as
    ``trajectory_grpo_loss`` but releases each large vocabulary-logit graph
    before processing the next action, which is necessary on the target GPU.
    """
    import torch

    if not trajectories:
        raise ValueError("trajectory group must not be empty")
    if clip_epsilon < 0 or kl_beta < 0 or loss_scale <= 0:
        raise ValueError("invalid GRPO objective configuration")
    group_size = len(trajectories)
    total_loss = 0.0
    max_delta = 0.0
    token_count = 0
    for trajectory in trajectories:
        if trajectory.advantage is None:
            raise ValueError("trajectory advantage has not been assigned")
        steps = selected_steps(trajectory, credit_mode)
        trajectory_token_count = sum(len(step.generated_token_ids) for step in steps)
        if trajectory_token_count < 1:
            raise ValueError("trajectory has no selected generated tokens")
        advantage = torch.tensor(float(trajectory.advantage), device=device)
        for step in steps:
            if step.old_log_probs is None or step.reference_log_probs is None:
                raise ValueError("old/reference log probabilities are missing")
            current = action_token_log_probs(
                model,
                step.context_token_ids,
                step.generated_token_ids,
                device=device,
                temperature=temperature,
            )
            old = step.old_log_probs.to(device)
            reference = step.reference_log_probs.to(device)
            max_delta = max(
                max_delta, float((current.detach() - old).abs().max().item())
            )
            ratio = torch.exp(current - old)
            unclipped = ratio * advantage
            clipped = torch.clamp(
                ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
            ) * advantage
            reference_delta = reference - current
            kl = torch.exp(reference_delta) - reference_delta - 1.0
            step_sum = (-torch.minimum(unclipped, clipped) + kl_beta * kl).sum()
            weight = loss_scale / (group_size * trajectory_token_count)
            (step_sum * weight).backward()
            total_loss += float(step_sum.detach().item()) / (
                group_size * trajectory_token_count
            )
            token_count += int(current.numel())
    return {
        "loss": total_loss,
        "max_old_current_log_prob_delta": max_delta,
        "action_token_count": float(token_count),
    }

