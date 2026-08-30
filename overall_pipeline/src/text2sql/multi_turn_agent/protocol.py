from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


PROTOCOL_SCHEMA_VERSION = 1
LOGICAL_CUDA_DEVICE = "cuda:0"
GIB = 1024 * 1024 * 1024
ROLE_MINIMUM_FREE_VRAM_BYTES = {
    "planner": 8 * GIB,
    "coder": 4 * GIB,
    "verifier": 8 * GIB,
}

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")


class AgentWorkerProtocolError(ValueError):
    """Raised when a worker specification or JSONL message is malformed."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def messages_sha256(messages: Sequence[Mapping[str, str]]) -> str:
    return hashlib.sha256(_canonical_json_bytes(list(messages))).hexdigest()


def normalize_messages(
    messages: Sequence[Mapping[str, str]],
) -> Tuple[Mapping[str, str], ...]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise AgentWorkerProtocolError("messages must be a non-empty sequence")
    normalized = []
    for position, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise AgentWorkerProtocolError(
                "messages[%d] must be an object" % position
            )
        if set(message) != {"role", "content"}:
            raise AgentWorkerProtocolError(
                "messages[%d] must contain only role and content" % position
            )
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise AgentWorkerProtocolError(
                "messages[%d].role is invalid" % position
            )
        if not isinstance(content, str) or not content:
            raise AgentWorkerProtocolError(
                "messages[%d].content must be a non-empty string" % position
            )
        normalized.append({"role": role, "content": content})
    if not normalized:
        raise AgentWorkerProtocolError("messages must not be empty")
    return tuple(normalized)


def _validate_identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SAFE_IDENTIFIER.fullmatch(value):
        raise AgentWorkerProtocolError(
            "%s may contain only letters, digits, dot, underscore, and dash" % label
        )


def _json_safe_mapping(value: Mapping[str, Any], label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AgentWorkerProtocolError("%s must be an object" % label)
    copied = dict(value)
    try:
        _canonical_json_bytes(copied)
    except (TypeError, ValueError) as exc:
        raise AgentWorkerProtocolError("%s must be JSON-serializable" % label) from exc
    return copied


@dataclass(frozen=True)
class RoleWorkerSpec:
    """Run-level specification for one long-lived role process.

    The physical GPU is exposed to the child through ``CUDA_VISIBLE_DEVICES``;
    every model inside the child is pinned to logical ``cuda:0``.
    """

    role: str
    worker_id: str
    physical_gpu: int
    backend: str
    model: Mapping[str, Any] = field(default_factory=dict)
    generation: Mapping[str, Any] = field(default_factory=dict)
    allow_model_download: bool = False
    mock_responses: Mapping[str, Any] = field(default_factory=dict)
    minimum_free_vram_bytes: Optional[int] = None
    startup_timeout_seconds: float = 600.0
    response_timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        _validate_identifier(self.role, "role")
        _validate_identifier(self.worker_id, "worker_id")
        if isinstance(self.physical_gpu, bool) or not isinstance(
            self.physical_gpu, int
        ) or self.physical_gpu < 0:
            raise AgentWorkerProtocolError("physical_gpu must be a non-negative integer")
        if self.backend not in {"hf", "mock"}:
            raise AgentWorkerProtocolError("backend must be hf or mock")
        if not isinstance(self.allow_model_download, bool):
            raise AgentWorkerProtocolError("allow_model_download must be boolean")
        model = _json_safe_mapping(self.model, "model")
        generation = _json_safe_mapping(self.generation, "generation")
        mock_responses = _json_safe_mapping(self.mock_responses, "mock_responses")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "mock_responses", mock_responses)
        if self.backend == "hf" and (not model or not generation):
            raise AgentWorkerProtocolError(
                "HF workers require model and generation objects"
            )
        minimum = self.minimum_free_vram_bytes
        if minimum is None:
            minimum = ROLE_MINIMUM_FREE_VRAM_BYTES.get(self.role, 4 * GIB)
            object.__setattr__(self, "minimum_free_vram_bytes", minimum)
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise AgentWorkerProtocolError(
                "minimum_free_vram_bytes must be a positive integer"
            )
        for label, value in (
            ("startup_timeout_seconds", self.startup_timeout_seconds),
            ("response_timeout_seconds", self.response_timeout_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise AgentWorkerProtocolError("%s must be positive" % label)

    def to_payload(self, launch_attempt: int) -> Dict[str, Any]:
        if isinstance(launch_attempt, bool) or not isinstance(
            launch_attempt, int
        ) or launch_attempt < 0:
            raise AgentWorkerProtocolError("launch_attempt must be non-negative")
        return {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "role": self.role,
            "worker_id": self.worker_id,
            "physical_gpu": self.physical_gpu,
            "logical_device": LOGICAL_CUDA_DEVICE,
            "backend": self.backend,
            "model": dict(self.model),
            "generation": dict(self.generation),
            "allow_model_download": self.allow_model_download,
            "mock_responses": dict(self.mock_responses),
            "minimum_free_vram_bytes": self.minimum_free_vram_bytes,
            "launch_attempt": launch_attempt,
        }


@dataclass(frozen=True)
class RoleTask:
    """One role generation call with an intentionally narrow, gold-free payload."""

    task_id: str
    example_id: str
    iteration: int
    messages: Sequence[Mapping[str, str]]
    prompt_sha256: Optional[str] = None
    mock_output: Optional[str] = None
    infrastructure_retry_limit: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise AgentWorkerProtocolError("task_id must be a non-empty string")
        if "\n" in self.task_id or "\r" in self.task_id:
            raise AgentWorkerProtocolError("task_id must not contain newlines")
        if not isinstance(self.example_id, str) or not self.example_id:
            raise AgentWorkerProtocolError("example_id must be a non-empty string")
        if isinstance(self.iteration, bool) or not isinstance(
            self.iteration, int
        ) or self.iteration < 1:
            raise AgentWorkerProtocolError("iteration must be a positive integer")
        normalized = normalize_messages(self.messages)
        object.__setattr__(self, "messages", normalized)
        digest = messages_sha256(normalized)
        if self.prompt_sha256 is not None and self.prompt_sha256 != digest:
            raise AgentWorkerProtocolError(
                "prompt_sha256 does not match the canonical messages"
            )
        object.__setattr__(self, "prompt_sha256", digest)
        if self.mock_output is not None and not isinstance(self.mock_output, str):
            raise AgentWorkerProtocolError("mock_output must be a string or null")
        if isinstance(self.infrastructure_retry_limit, bool) or self.infrastructure_retry_limit not in {
            0,
            1,
        }:
            raise AgentWorkerProtocolError(
                "infrastructure_retry_limit must be 0 or 1"
            )

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "type": "generate",
            "task_id": self.task_id,
            "example_id": self.example_id,
            "iteration": self.iteration,
            "messages": [dict(message) for message in self.messages],
            "prompt_sha256": self.prompt_sha256,
        }
        if self.mock_output is not None:
            payload["mock_output"] = self.mock_output
        # infrastructure_retry_limit is parent-local resume state and is never
        # exposed to the model worker protocol.
        return payload


@dataclass(frozen=True)
class RoleTaskResult:
    role: str
    task_id: str
    example_id: str
    iteration: int
    prompt_sha256: str
    generation: Mapping[str, Any]
    infrastructure_retries: int = 0

    @property
    def succeeded(self) -> bool:
        return self.generation.get("status") == "success"
