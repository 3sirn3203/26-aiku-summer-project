from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from text2sql.core.sql_output import extract_sql
from rl_finetune.agentic_runtime.environment import SQLWorkflowEnvironment
from rl_finetune.agentic_runtime.models import GeneratedTurn, RolloutStep, Trajectory
from rl_finetune.agentic_runtime.prompt import build_final_messages


class WorkflowGenerationPolicy(Protocol):
    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> GeneratedTurn: ...


class TransformersWorkflowPolicy:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: Any,
        max_input_tokens: int,
        max_time_seconds: float,
    ) -> None:
        if max_input_tokens < 1:
            raise ValueError("max_input_tokens must be positive")
        if max_time_seconds <= 0:
            raise ValueError("max_time_seconds must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.max_time_seconds = max_time_seconds

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> GeneratedTurn:
        import torch

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        encoded = self.tokenizer.apply_chat_template(
            list(messages),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        context_ids = encoded["input_ids"][0]
        if int(context_ids.shape[0]) > self.max_input_tokens:
            raise RuntimeError(
                "workflow prompt has %d tokens, exceeding max_input_tokens=%d"
                % (int(context_ids.shape[0]), self.max_input_tokens)
            )
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=temperature,
                    top_p=1.0,
                    top_k=0,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    max_time=self.max_time_seconds,
                    repetition_penalty=1.0,
                    pad_token_id=(
                        self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None
                        else self.tokenizer.eos_token_id
                    ),
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            self.model.train(was_training)
        generated_ids = output[0, context_ids.shape[0] :]
        return GeneratedTurn(
            context_token_ids=context_ids.detach().cpu().tolist(),
            generated_token_ids=generated_ids.detach().cpu().tolist(),
            raw_output=self.tokenizer.decode(generated_ids, skip_special_tokens=True),
            input_tokens=int(context_ids.shape[0]),
            output_tokens=int(generated_ids.shape[0]),
        )


class WorkflowRolloutRunner:
    def __init__(
        self,
        policy: WorkflowGenerationPolicy,
        environment: SQLWorkflowEnvironment,
        *,
        num_generations: int = 4,
        temperature: float = 0.9,
        max_draft_tokens: int = 256,
        max_final_tokens: int = 256,
    ) -> None:
        if num_generations < 1:
            raise ValueError("num_generations must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if max_draft_tokens < 1 or max_final_tokens < 1:
            raise ValueError("generation token limits must be positive")
        self.policy = policy
        self.environment = environment
        self.num_generations = num_generations
        self.temperature = temperature
        self.max_draft_tokens = max_draft_tokens
        self.max_final_tokens = max_final_tokens

    def collect_group(self, record: Mapping[str, Any]) -> list[Trajectory]:
        messages = record.get("prompt")
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise ValueError("rollout record prompt must be a message sequence")
        example_id = str(record.get("example_id", ""))
        db_path = Path(str(record.get("db_path", "")))
        trajectories = []
        for generation_index in range(self.num_generations):
            draft = self.policy.generate(
                messages,
                max_new_tokens=self.max_draft_tokens,
                temperature=self.temperature,
            )
            draft_parse, _, observation = self.environment.execute_draft(
                db_path, draft.raw_output
            )
            final_messages = build_final_messages(
                messages,
                draft_raw_output=draft.raw_output,
                observation=observation,
            )
            final = self.policy.generate(
                final_messages,
                max_new_tokens=self.max_final_tokens,
                temperature=self.temperature,
            )
            final_parse = extract_sql(final.raw_output)
            trajectories.append(
                Trajectory(
                    example_id=example_id,
                    generation_index=generation_index,
                    steps=[
                        RolloutStep(
                            role="draft",
                            context_token_ids=draft.context_token_ids,
                            generated_token_ids=draft.generated_token_ids,
                            raw_output=draft.raw_output,
                            sql=draft_parse.sql,
                            observation=observation,
                        ),
                        RolloutStep(
                            role="final",
                            context_token_ids=final.context_token_ids,
                            generated_token_ids=final.generated_token_ids,
                            raw_output=final.raw_output,
                            sql=final_parse.sql,
                        ),
                    ],
                    final_raw_output=final.raw_output,
                    terminal_status=(
                        "ready" if final_parse.status == "success" else "parse_error"
                    ),
                )
            )
        return trajectories

