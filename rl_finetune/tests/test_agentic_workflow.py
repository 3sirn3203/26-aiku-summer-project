from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from text2sql.config import ExecutionConfig
from text2sql.core.models import ExecutionResult
from rl_finetune.agentic_runtime.environment import (
    SQLWorkflowEnvironment,
    WorkflowInfrastructureError,
)
from rl_finetune.agentic_runtime.models import GeneratedTurn, SQLObservation
from rl_finetune.agentic_runtime.prompt import build_final_messages
from rl_finetune.agentic_runtime.rollout import (
    TransformersWorkflowPolicy,
    WorkflowRolloutRunner,
)
from rl_finetune.rewards import RewardConfig
from rl_finetune.rewards import RewardResult
from rl_finetune.trajectory_reward import TrajectoryRewardRuntime


def _execution_config() -> ExecutionConfig:
    return ExecutionConfig(
        timeout_seconds=1.0,
        max_sql_bytes=10000,
        max_result_rows=100,
        max_result_bytes=100000,
        worker_memory_limit_bytes=1024 * 1024 * 1024,
    )


class _QueuePolicy:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, messages, *, max_new_tokens, temperature):
        self.calls.append([dict(message) for message in messages])
        output = self.outputs.pop(0)
        call_index = len(self.calls)
        return GeneratedTurn(
            context_token_ids=[10, call_index],
            generated_token_ids=[20, call_index],
            raw_output=output,
        )


class _GenerationTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens=True):
        return "SELECT 1"


class _GenerationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.generation_kwargs = None

    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        return torch.tensor([[1, 2, 3]], dtype=torch.long)


class WorkflowObservationTests(unittest.TestCase):
    def test_rollout_disables_inherited_top_k_and_top_p_sampling(self) -> None:
        model = _GenerationModel()
        policy = TransformersWorkflowPolicy(
            model,
            _GenerationTokenizer(),
            device=torch.device("cpu"),
            max_input_tokens=10,
            max_time_seconds=1.0,
        )
        policy.generate(
            [{"role": "user", "content": "question"}],
            max_new_tokens=4,
            temperature=0.9,
        )
        self.assertEqual(model.generation_kwargs["temperature"], 0.9)
        self.assertEqual(model.generation_kwargs["top_p"], 1.0)
        self.assertEqual(model.generation_kwargs["top_k"], 0)

    def test_parse_error_becomes_observation(self) -> None:
        environment = SQLWorkflowEnvironment(_execution_config())
        parse, execution, observation = environment.execute_draft(
            Path("unused.sqlite"), "not sql"
        )
        self.assertEqual(parse.status, "error")
        self.assertEqual(execution.status, "not_run")
        self.assertEqual(observation.error_type, "sql_parse_error")

    def test_execution_statuses_are_preserved_and_bounded(self) -> None:
        cases = (
            ExecutionResult(status="unsafe_sql", error_type="unsafe_sql"),
            ExecutionResult(status="syntax_error", error_type="syntax_error"),
            ExecutionResult(
                status="execution_timeout", error_type="execution_timeout"
            ),
        )
        for execution in cases:
            with self.subTest(status=execution.status), patch(
                "rl_finetune.agentic_runtime.environment.execute_sql",
                return_value=execution,
            ):
                environment = SQLWorkflowEnvironment(_execution_config())
                _, _, observation = environment.execute_draft(
                    Path("unused.sqlite"), "SELECT 1"
                )
                self.assertEqual(observation.status, execution.status)

        oversized = ExecutionResult(
            status="success",
            columns=["value"],
            rows=[[index] for index in range(25)],
            error_message="x" * 700,
        )
        with patch(
            "rl_finetune.agentic_runtime.environment.execute_sql", return_value=oversized
        ):
            observation = SQLWorkflowEnvironment(
                _execution_config(), max_observation_rows=20
            ).execute_draft(Path("unused.sqlite"), "SELECT 1")[2]
        self.assertEqual(len(observation.rows), 20)
        self.assertTrue(observation.truncated)
        self.assertEqual(len(observation.error_message), 500)
        json.dumps(observation.to_dict())

    def test_infrastructure_failure_raises(self) -> None:
        failure = ExecutionResult(
            status="internal_error", error_type="executor_worker_crash"
        )
        with patch(
            "rl_finetune.agentic_runtime.environment.execute_sql", return_value=failure
        ):
            with self.assertRaises(WorkflowInfrastructureError):
                SQLWorkflowEnvironment(_execution_config()).execute_draft(
                    Path("unused.sqlite"), "SELECT 1"
                )

    def test_final_prompt_contains_only_draft_and_observation_feedback(self) -> None:
        initial = [{"role": "user", "content": "question and schema"}]
        observation = SQLObservation(status="success", rows=[[1]])
        messages = build_final_messages(
            initial, draft_raw_output="SELECT 1", observation=observation
        )
        rendered = json.dumps(messages, ensure_ascii=False)
        self.assertIn("SELECT 1", rendered)
        self.assertIn("execution observation", rendered)
        self.assertIn("exactly one final SQLite SQL", rendered)
        self.assertEqual(initial, [{"role": "user", "content": "question and schema"}])


class MockWorkflowEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="agentic_sqlite_")
        self.db_path = Path(self.temporary.name) / "fixture.sqlite"
        connection = sqlite3.connect(str(self.db_path))
        connection.executescript(
            "CREATE TABLE items(id INTEGER, name TEXT);"
            "INSERT INTO items VALUES (1, 'alpha'), (2, 'beta');"
        )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_draft_execute_observe_final_reward(self) -> None:
        policy = _QueuePolicy(
            ["SELECT missing FROM items", "SELECT name FROM items ORDER BY id"]
        )
        runner = WorkflowRolloutRunner(
            policy,
            SQLWorkflowEnvironment(_execution_config()),
            num_generations=1,
        )
        gold_sql = "SELECT name FROM items ORDER BY id"
        record = {
            "prompt": [{"role": "user", "content": "List item names"}],
            "example_id": "mock:0",
            "db_path": str(self.db_path),
            "gold_sql": gold_sql,
            "order_sensitive": True,
        }
        trajectory = runner.collect_group(record)[0]
        self.assertEqual(
            trajectory.steps[0].observation.status, "execution_error"
        )
        final_context = json.dumps(policy.calls[1], ensure_ascii=False)
        self.assertIn("SELECT missing FROM items", final_context)
        self.assertIn("execution_error", final_context)
        self.assertNotIn(gold_sql, final_context)

        runtime = TrajectoryRewardRuntime(
            _execution_config(), reward_config=RewardConfig(latency_weight=0.0)
        )
        self.assertTrue(runtime.score_group([trajectory], record))
        self.assertEqual(trajectory.reward, 1.0)
        self.assertEqual(trajectory.terminal_status, "correct")

    def test_only_final_raw_output_is_sent_to_existing_reward(self) -> None:
        policy = _QueuePolicy(["SELECT 999", "SELECT 1"])
        runner = WorkflowRolloutRunner(
            policy,
            SQLWorkflowEnvironment(_execution_config()),
            num_generations=1,
        )
        record = {
            "prompt": [{"role": "user", "content": "Return one"}],
            "example_id": "mock:reward",
            "db_path": str(self.db_path),
            "gold_sql": "SELECT 1",
            "order_sensitive": False,
        }
        trajectory = runner.collect_group(record)[0]

        class _Cache:
            def get(self, db_path, gold_sql):
                return ExecutionResult(status="success", rows=[[1]], columns=["1"])

        runtime = TrajectoryRewardRuntime(
            _execution_config(), gold_cache=_Cache()
        )
        with patch(
            "rl_finetune.trajectory_reward.score_completion",
            return_value=RewardResult(reward=1.0, status="correct", correct=True),
        ) as scorer:
            self.assertTrue(runtime.score_group([trajectory], record))
        self.assertEqual(scorer.call_args.kwargs["raw_output"], "SELECT 1")
        self.assertNotEqual(scorer.call_args.kwargs["raw_output"], "SELECT 999")

    def test_gold_execution_failure_skips_whole_group(self) -> None:
        policy = _QueuePolicy(["SELECT 1", "SELECT 1"])
        trajectory = WorkflowRolloutRunner(
            policy,
            SQLWorkflowEnvironment(_execution_config()),
            num_generations=1,
        ).collect_group(
            {
                "prompt": [{"role": "user", "content": "Return one"}],
                "example_id": "mock:gold-failure",
                "db_path": str(self.db_path),
            }
        )[0]

        class _FailedCache:
            def get(self, db_path, gold_sql):
                return ExecutionResult(
                    status="execution_error", error_type="fixture_gold_error"
                )

        runtime = TrajectoryRewardRuntime(
            _execution_config(), gold_cache=_FailedCache()
        )
        record = {
            "db_path": str(self.db_path),
            "gold_sql": "SELECT broken",
            "order_sensitive": False,
        }
        self.assertFalse(runtime.score_group([trajectory], record))
        self.assertIsNone(trajectory.reward)
        self.assertEqual(trajectory.terminal_status, "gold_execution_error")

    def test_final_executor_infrastructure_failure_is_not_a_reward(self) -> None:
        policy = _QueuePolicy(["SELECT 1", "SELECT 1"])
        record = {
            "prompt": [{"role": "user", "content": "Return one"}],
            "example_id": "mock:infra",
            "db_path": str(self.db_path),
            "gold_sql": "SELECT 1",
            "order_sensitive": False,
        }
        trajectory = WorkflowRolloutRunner(
            policy,
            SQLWorkflowEnvironment(_execution_config()),
            num_generations=1,
        ).collect_group(record)[0]

        class _Cache:
            def get(self, db_path, gold_sql):
                return ExecutionResult(status="success", rows=[[1]], columns=["1"])

        runtime = TrajectoryRewardRuntime(
            _execution_config(), gold_cache=_Cache()
        )
        failure = RewardResult(
            reward=-0.7,
            status="prediction_execution_error",
            correct=False,
            predicted_execution={"error_type": "executor_worker_crash"},
        )
        with patch(
            "rl_finetune.trajectory_reward.score_completion",
            return_value=failure,
        ), self.assertRaises(WorkflowInfrastructureError):
            runtime.score_group([trajectory], record)


if __name__ == "__main__":
    unittest.main()

