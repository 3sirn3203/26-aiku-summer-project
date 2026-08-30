from __future__ import annotations

import json
import unittest
from contextlib import contextmanager
import tempfile
from pathlib import Path

import torch

from rl_finetune.agentic_runtime.models import RolloutStep, Trajectory
from rl_finetune.agentic_grpo_trainer import (
    AgenticGRPOConfig,
    AgenticGRPOTrainer,
)
from rl_finetune.trajectory_grpo import (
    action_token_log_probs,
    backward_trajectory_grpo_group,
    cache_old_and_reference_log_probs,
    normalize_group_advantages,
    selected_steps,
    trajectory_grpo_loss,
)


def _step(role: str, action_tokens=(3,)) -> RolloutStep:
    return RolloutStep(
        role=role,
        context_token_ids=[1, 2],
        generated_token_ids=list(action_tokens),
        raw_output="SQL",
    )


def _trajectory(index: int, reward: float, action_tokens=(3,)) -> Trajectory:
    return Trajectory(
        example_id="mock",
        generation_index=index,
        steps=[_step("draft", action_tokens), _step("final", action_tokens)],
        final_raw_output="SELECT 1",
        terminal_status="ready",
        reward=reward,
    )


class _PositionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, use_cache=False):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 8, device=input_ids.device) * self.scale
        for position in range(length):
            logits[:, position, min(position + 2, 7)] = 10.0 * self.scale
        return type("Output", (), {"logits": logits})()


class _UniformModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(8))

    def forward(self, input_ids, use_cache=False):
        batch, length = input_ids.shape
        logits = self.logits.view(1, 1, -1).expand(batch, length, -1)
        return type("Output", (), {"logits": logits})()


class _AdapterLikeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Parameter(torch.zeros(8), requires_grad=False)
        self.lora_adapter = torch.nn.Parameter(torch.linspace(-0.2, 0.2, 8))
        self._adapter_disabled = False

    def forward(self, input_ids, use_cache=False):
        batch, length = input_ids.shape
        adapter = 0.0 if self._adapter_disabled else self.lora_adapter
        logits = (self.base + adapter).view(1, 1, -1).expand(batch, length, -1)
        return type("Output", (), {"logits": logits})()

    @contextmanager
    def disable_adapter(self):
        previous = self._adapter_disabled
        self._adapter_disabled = True
        try:
            yield
        finally:
            self._adapter_disabled = previous

    def save_pretrained(self, path):
        Path(path, "adapter_config.json").write_text("{}", encoding="utf-8")


class _FakeTokenizer:
    def save_pretrained(self, path):
        Path(path, "tokenizer_config.json").write_text("{}", encoding="utf-8")


class _FakeRunner:
    def collect_group(self, record):
        return [
            _trajectory(0, -1.0, action_tokens=(2, 3)),
            _trajectory(1, 1.0, action_tokens=(6, 7)),
        ]


class _FakeRewardRuntime:
    def score_group(self, trajectories, record):
        trajectories[0].reward = -1.0
        trajectories[1].reward = 1.0
        return True


class TrajectoryGRPOTests(unittest.TestCase):
    def test_causal_shift_selects_generated_token_boundaries(self) -> None:
        log_probs = action_token_log_probs(
            _PositionModel(), [1, 2], [3, 4], device=torch.device("cpu")
        )
        self.assertEqual(tuple(log_probs.shape), (2,))
        self.assertTrue(torch.all(log_probs > -0.01))

    def test_log_probs_use_the_rollout_temperature(self) -> None:
        model = _PositionModel()
        unscaled = action_token_log_probs(
            model, [1, 2], [3, 4], device=torch.device("cpu"), temperature=1.0
        )
        colder = action_token_log_probs(
            model, [1, 2], [3, 4], device=torch.device("cpu"), temperature=0.5
        )
        self.assertTrue(torch.all(colder > unscaled))

    def test_credit_modes_select_expected_actions(self) -> None:
        trajectory = _trajectory(0, 1.0)
        self.assertEqual(
            [step.role for step in selected_steps(trajectory, "all_actions")],
            ["draft", "final"],
        )
        self.assertEqual(
            [step.role for step in selected_steps(trajectory, "final_only")],
            ["final"],
        )

    def test_group_normalization_and_zero_variance(self) -> None:
        group = [_trajectory(0, -1.0), _trajectory(1, 1.0)]
        self.assertTrue(normalize_group_advantages(group))
        self.assertAlmostEqual(group[0].advantage, -1.0)
        self.assertAlmostEqual(group[1].advantage, 1.0)

        equal = [_trajectory(0, 1.0), _trajectory(1, 1.0)]
        self.assertFalse(normalize_group_advantages(equal))
        self.assertEqual([item.advantage for item in equal], [0.0, 0.0])

    def test_trajectory_length_does_not_change_group_weight(self) -> None:
        model = _UniformModel()
        short = _trajectory(0, -1.0, action_tokens=(3,))
        long = _trajectory(1, 1.0, action_tokens=(3, 4, 5))
        self.assertTrue(normalize_group_advantages([short, long]))
        for trajectory in (short, long):
            for step in trajectory.steps:
                values = action_token_log_probs(
                    model,
                    step.context_token_ids,
                    step.generated_token_ids,
                    device=torch.device("cpu"),
                ).detach()
                step.old_log_probs = values
                step.reference_log_probs = values
        loss, _ = trajectory_grpo_loss(
            model,
            [short, long],
            credit_mode="final_only",
            clip_epsilon=0.2,
            kl_beta=0.01,
            device=torch.device("cpu"),
        )
        self.assertAlmostEqual(float(loss.detach().item()), 0.0, places=6)

    def test_old_current_match_and_only_adapter_gets_gradient(self) -> None:
        model = _AdapterLikeModel()
        group = [
            _trajectory(0, -1.0, action_tokens=(2, 3)),
            _trajectory(1, 1.0, action_tokens=(6, 7)),
        ]
        self.assertTrue(normalize_group_advantages(group))
        cache_old_and_reference_log_probs(
            model,
            group,
            credit_mode="all_actions",
            device=torch.device("cpu"),
        )
        metrics = backward_trajectory_grpo_group(
            model,
            group,
            credit_mode="all_actions",
            clip_epsilon=0.2,
            kl_beta=0.01,
            device=torch.device("cpu"),
        )
        self.assertLess(metrics["max_old_current_log_prob_delta"], 1e-7)
        self.assertIsNone(model.base.grad)
        self.assertIsNotNone(model.lora_adapter.grad)
        self.assertTrue(torch.isfinite(model.lora_adapter.grad).all())

    def test_explicit_trainer_performs_step_and_writes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentic_trainer_") as directory:
            trainer = AgenticGRPOTrainer(
                model=_AdapterLikeModel(),
                tokenizer=_FakeTokenizer(),
                rollout_runner=_FakeRunner(),
                reward_runtime=_FakeRewardRuntime(),
                records=[{"example_id": "mock"}],
                config=AgenticGRPOConfig(max_steps=1),
                run_dir=Path(directory),
                device=torch.device("cpu"),
            )
            result = trainer.train()
            checkpoint = Path(directory) / "trainer" / "checkpoint-1"
            self.assertEqual(result["global_step"], 1)
            self.assertTrue((checkpoint / "adapter_config.json").is_file())
            self.assertTrue((checkpoint / "optimizer.pt").is_file())
            self.assertTrue((checkpoint / "scheduler.pt").is_file())
            self.assertTrue((checkpoint / "rng_state.pt").is_file())
            self.assertTrue((checkpoint / "trainer_state.json").is_file())
            saved_state = json.loads(
                (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_state["record_index"], 1)

            resumed_model = _AdapterLikeModel()
            resumed_model.load_state_dict(trainer.model.state_dict())
            resumed_dir = Path(directory) / "resumed"
            resumed = AgenticGRPOTrainer(
                model=resumed_model,
                tokenizer=_FakeTokenizer(),
                rollout_runner=_FakeRunner(),
                reward_runtime=_FakeRewardRuntime(),
                records=[{"example_id": "mock"}],
                config=AgenticGRPOConfig(max_steps=2),
                run_dir=resumed_dir,
                device=torch.device("cpu"),
                initial_global_step=1,
                initial_record_index=saved_state["record_index"],
                optimizer_state_path=checkpoint / "optimizer.pt",
                scheduler_state_path=checkpoint / "scheduler.pt",
            )
            resumed_result = resumed.train()
            self.assertEqual(resumed_result["global_step"], 2)
            self.assertEqual(resumed_result["record_index"], 2)
            self.assertTrue(
                (resumed_dir / "trainer" / "checkpoint-2" / "optimizer.pt").is_file()
            )


if __name__ == "__main__":
    unittest.main()

