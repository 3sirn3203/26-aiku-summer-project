from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from text2sql.config import ExecutionConfig, GenerationConfig, ModelConfig
from text2sql.core.backends.base import GenerationBackend
from text2sql.core.backends.hf import HuggingFaceBackend
from text2sql.core.models import GenerationRequest, GenerationResult
from text2sql.core.sql_output import extract_sql

from rl_finetune.agentic_runtime.environment import SQLWorkflowEnvironment
from rl_finetune.agentic_runtime.prompt import build_final_messages
from rl_finetune.agentic_runtime.rollout import TransformersWorkflowPolicy


class TwoTurnWorkflowBackend(GenerationBackend):
    """Two model actions with one bounded SQLite observation between them."""

    name = "two_turn_workflow"

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        generation_config: GenerationConfig,
        execution_config: ExecutionConfig,
        workflow_contract: Mapping[str, Any],
        database_paths: Mapping[str, Any],
        adapter_dir: Optional[Path],
        allow_model_download: bool,
        expected_adapter_identity: Optional[str],
    ) -> None:
        self.workflow_contract = dict(workflow_contract)
        self.database_paths = {
            str(key): Path(str(value)).expanduser().resolve()
            for key, value in database_paths.items()
        }
        required_positive = (
            "max_draft_tokens",
            "max_final_tokens",
            "max_observation_rows",
        )
        for key in required_positive:
            value = self.workflow_contract.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("two-turn workflow %s must be a positive integer" % key)
        temperature = self.workflow_contract.get("temperature")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("two-turn workflow temperature must be numeric")
        if float(temperature) <= 0:
            raise ValueError("two-turn workflow temperature must be positive")
        seed = self.workflow_contract.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("two-turn workflow seed must be a non-negative integer")

        if adapter_dir is None:
            self._base_backend: GenerationBackend = HuggingFaceBackend(
                model_config=model_config,
                generation_config=generation_config,
                allow_model_download=allow_model_download,
            )
        else:
            from text2sql.core.backends.peft import PeftAdapterBackend

            self._base_backend = PeftAdapterBackend(
                model_config=model_config,
                generation_config=generation_config,
                adapter_dir=adapter_dir,
                allow_model_download=allow_model_download,
                expected_adapter_identity=expected_adapter_identity,
            )
        backend = self._base_backend
        self._torch = getattr(backend, "_torch")
        self._model = getattr(backend, "_model")
        self._tokenizer = getattr(backend, "_tokenizer")
        self._device = getattr(backend, "_device")
        self._resolved_revision = getattr(backend, "_resolved_revision")
        self.policy = TransformersWorkflowPolicy(
            self._model,
            self._tokenizer,
            device=self._device,
            max_input_tokens=generation_config.max_input_tokens,
            max_time_seconds=generation_config.max_time_seconds,
        )
        self.environment = SQLWorkflowEnvironment(
            execution_config,
            max_observation_rows=int(
                self.workflow_contract["max_observation_rows"]
            ),
        )
        self._traces: Dict[str, Dict[str, Any]] = {}

    def _seed_for_example(self, example_id: str) -> None:
        try:
            index = int(example_id.rsplit(":", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError("invalid Spider example ID: %s" % example_id) from exc
        value = int(self.workflow_contract["seed"]) + index
        self._torch.manual_seed(value)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(value)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        started = time.monotonic()
        db_path = self.database_paths.get(request.example_id)
        if db_path is None:
            raise RuntimeError("two-turn worker has no DB path for %s" % request.example_id)
        self._seed_for_example(request.example_id)
        draft = self.policy.generate(
            request.messages,
            max_new_tokens=int(self.workflow_contract["max_draft_tokens"]),
            temperature=float(self.workflow_contract["temperature"]),
        )
        draft_parse, draft_execution, observation = self.environment.execute_draft(
            db_path,
            draft.raw_output,
        )
        final_messages = build_final_messages(
            request.messages,
            draft_raw_output=draft.raw_output,
            observation=observation,
        )
        final = self.policy.generate(
            final_messages,
            max_new_tokens=int(self.workflow_contract["max_final_tokens"]),
            temperature=float(self.workflow_contract["temperature"]),
        )
        final_parse = extract_sql(final.raw_output)
        self._traces[request.example_id] = {
            "draft": {
                "raw_output": draft.raw_output,
                "sql_parsing": draft_parse.to_dict(),
                "execution": draft_execution.to_dict(),
                "observation": observation.to_dict(),
            },
            "final": {
                "raw_output": final.raw_output,
                "sql_parsing": final_parse.to_dict(),
            },
            "seed": int(self.workflow_contract["seed"])
            + int(request.example_id.rsplit(":", 1)[1]),
        }
        return GenerationResult(
            status="success",
            raw_output=final.raw_output,
            elapsed_seconds=time.monotonic() - started,
            input_tokens=final.input_tokens,
            output_tokens=final.output_tokens,
            model_id=getattr(self._base_backend, "model_config").model_id,
            requested_revision=getattr(
                self._base_backend, "model_config"
            ).revision,
            resolved_revision=self._resolved_revision,
        )

    def pop_trace(self, example_id: str) -> Dict[str, Any]:
        try:
            return self._traces.pop(example_id)
        except KeyError as exc:
            raise RuntimeError("two-turn backend did not produce a trace") from exc

    def metadata(self) -> Dict[str, Any]:
        payload = self._base_backend.metadata()
        payload["backend"] = self.name
        payload["workflow"] = dict(self.workflow_contract)
        return payload

    def close(self) -> None:
        self._traces.clear()
        self._base_backend.close()
