from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from text2sql.config import ExecutionConfig, GenerationConfig, ModelConfig
from text2sql.core.models import GenerationRequest

from rl_finetune.agentic_runtime.models import GeneratedTurn
from rl_finetune.two_turn_backend import TwoTurnWorkflowBackend


class _FakeTorch:
    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def manual_seed_all(_value: int) -> None:
            return None

    @staticmethod
    def manual_seed(_value: int) -> None:
        return None


class _FakeBaseBackend:
    def __init__(self, *, model_config, generation_config, allow_model_download):
        del generation_config, allow_model_download
        self.model_config = model_config
        self._torch = _FakeTorch()
        self._model = SimpleNamespace()
        self._tokenizer = SimpleNamespace()
        self._device = "cpu"
        self._resolved_revision = model_config.revision

    def metadata(self):
        return {"resolved_revision": self._resolved_revision}

    def close(self):
        return None


class _FakePolicy:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.calls = 0

    def generate(self, messages, *, max_new_tokens, temperature):
        del messages, max_new_tokens, temperature
        self.calls += 1
        sql = "SELECT value FROM items"
        return GeneratedTurn(
            context_token_ids=(1, 2),
            generated_token_ids=(3, 4),
            raw_output=sql,
            input_tokens=2,
            output_tokens=2,
        )


class TwoTurnBackendTests(unittest.TestCase):
    def test_draft_execution_trace_and_final_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "fixture.sqlite"
            connection = sqlite3.connect(str(db_path))
            connection.execute("CREATE TABLE items(value INTEGER)")
            connection.execute("INSERT INTO items VALUES (7)")
            connection.commit()
            connection.close()

            with patch(
                "rl_finetune.two_turn_backend.HuggingFaceBackend",
                _FakeBaseBackend,
            ), patch(
                "rl_finetune.two_turn_backend.TransformersWorkflowPolicy",
                _FakePolicy,
            ):
                backend = TwoTurnWorkflowBackend(
                    model_config=ModelConfig(
                        model_id="Qwen/Coder",
                        revision="a" * 40,
                        dtype="float32",
                        device="cuda:0",
                        attention_implementation="eager",
                        trust_remote_code=False,
                        cache_dir=None,
                    ),
                    generation_config=GenerationConfig(
                        do_sample=False,
                        num_beams=1,
                        repetition_penalty=1.0,
                        max_time_seconds=10.0,
                        max_input_tokens=512,
                        max_new_tokens=32,
                        batch_size=1,
                    ),
                    execution_config=ExecutionConfig(
                        timeout_seconds=1.0,
                        max_sql_bytes=4096,
                        max_result_rows=20,
                        max_result_bytes=4096,
                        worker_memory_limit_bytes=128 * 1024 * 1024,
                    ),
                    workflow_contract={
                        "max_draft_tokens": 32,
                        "max_final_tokens": 32,
                        "max_observation_rows": 5,
                        "temperature": 0.9,
                        "seed": 42,
                    },
                    database_paths={"dev:3": str(db_path)},
                    adapter_dir=None,
                    allow_model_download=False,
                    expected_adapter_identity=None,
                )
                result = backend.generate(
                    GenerationRequest(
                        example_id="dev:3",
                        messages=({"role": "user", "content": "query"},),
                    )
                )
                trace = backend.pop_trace("dev:3")

        self.assertEqual(result.status, "success")
        self.assertEqual(result.raw_output, "SELECT value FROM items")
        self.assertEqual(trace["draft"]["execution"]["status"], "success")
        self.assertEqual(trace["draft"]["observation"]["rows"], ((7,),))
        self.assertEqual(trace["seed"], 45)


if __name__ == "__main__":
    unittest.main()
