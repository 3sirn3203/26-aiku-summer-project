from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from text2sql.config import GenerationConfig, ModelConfig
from text2sql.core.models import GenerationRequest, GenerationResult
from text2sql.core.model_source import LOCAL_IDENTITY_PREFIX
from text2sql.multi_turn_agent.protocol import (
    LOGICAL_CUDA_DEVICE,
    PROTOCOL_SCHEMA_VERSION,
    AgentWorkerProtocolError,
    messages_sha256,
    normalize_messages,
)


def _configure_linux_parent_death_signal() -> None:
    """Ask Linux to terminate this GPU worker if its coordinator disappears."""

    if not sys.platform.startswith("linux"):
        return
    import ctypes

    parent_pid = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    pr_set_pdeathsig = 1
    if libc.prctl(pr_set_pdeathsig, signal.SIGTERM, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "prctl(PR_SET_PDEATHSIG) failed")
    # Close the small fork/exec-to-prctl race: if the coordinator already died,
    # Linux could not deliver the newly configured signal retroactively.
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def _emit(payload: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise AgentWorkerProtocolError("worker spec field %s must be a string" % key)
    return value


def _required_integer(payload: Mapping[str, Any], key: str, minimum: int = 0) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AgentWorkerProtocolError(
            "worker spec field %s must be an integer >= %d" % (key, minimum)
        )
    return value


def _required_number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise AgentWorkerProtocolError(
            "worker spec field %s must be positive" % key
        )
    return float(value)


def _load_spec(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentWorkerProtocolError("could not read worker spec: %s" % exc) from exc
    if not isinstance(payload, dict):
        raise AgentWorkerProtocolError("worker spec must be a JSON object")
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise AgentWorkerProtocolError("unsupported worker protocol schema")
    if payload.get("logical_device") != LOGICAL_CUDA_DEVICE:
        raise AgentWorkerProtocolError("worker logical device must be cuda:0")
    physical_gpu = _required_integer(payload, "physical_gpu")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise AgentWorkerProtocolError(
            "CUDA_VISIBLE_DEVICES must expose exactly physical GPU %d" % physical_gpu
        )
    _required_string(payload, "role")
    _required_string(payload, "worker_id")
    backend = _required_string(payload, "backend")
    if backend not in {"hf", "mock"}:
        raise AgentWorkerProtocolError("worker backend must be hf or mock")
    _required_integer(payload, "minimum_free_vram_bytes", minimum=1)
    return payload


def _model_config(payload: Mapping[str, Any]) -> ModelConfig:
    cache_dir = payload.get("cache_dir")
    if cache_dir is not None and not isinstance(cache_dir, str):
        raise AgentWorkerProtocolError("model.cache_dir must be a string or null")
    trust_remote_code = payload.get("trust_remote_code")
    if not isinstance(trust_remote_code, bool):
        raise AgentWorkerProtocolError("model.trust_remote_code must be boolean")
    source = payload.get("source", "hub")
    if source not in {"hub", "local"}:
        raise AgentWorkerProtocolError("model.source must be hub or local")
    checkpoint_identity = payload.get("checkpoint_identity")
    if checkpoint_identity is not None and not isinstance(checkpoint_identity, str):
        raise AgentWorkerProtocolError(
            "model.checkpoint_identity must be a string or null"
        )
    if source == "local" and (
        not isinstance(checkpoint_identity, str)
        or not checkpoint_identity.startswith(LOCAL_IDENTITY_PREFIX)
    ):
        raise AgentWorkerProtocolError(
            "local worker model requires a prepared checkpoint identity"
        )
    return ModelConfig(
        model_id=_required_string(payload, "id"),
        revision=_required_string(payload, "revision"),
        dtype=_required_string(payload, "dtype"),
        device=LOGICAL_CUDA_DEVICE,
        attention_implementation=_required_string(
            payload, "attention_implementation"
        ),
        trust_remote_code=trust_remote_code,
        cache_dir=Path(cache_dir).expanduser().resolve() if cache_dir else None,
        source=source,
        checkpoint_identity=checkpoint_identity,
    )


def _generation_config(payload: Mapping[str, Any]) -> GenerationConfig:
    do_sample = payload.get("do_sample")
    if not isinstance(do_sample, bool):
        raise AgentWorkerProtocolError("generation.do_sample must be boolean")
    batch_size = _required_integer(payload, "batch_size", minimum=1)
    num_beams = _required_integer(payload, "num_beams", minimum=1)
    max_input_tokens = _required_integer(payload, "max_input_tokens", minimum=1)
    max_new_tokens = _required_integer(payload, "max_new_tokens", minimum=1)
    repetition_penalty = _required_number(payload, "repetition_penalty")
    max_time_seconds = _required_number(payload, "max_time_seconds")
    return GenerationConfig(
        do_sample=do_sample,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        max_time_seconds=max_time_seconds,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )


def _check_role_vram_before_load(minimum_free_vram_bytes: int) -> Dict[str, int]:
    # This import is intentionally confined to the child worker process.
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("HF worker requires torch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    device = torch.device(LOGICAL_CUDA_DEVICE)
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    if int(free_memory) < minimum_free_vram_bytes:
        raise RuntimeError(
            "role requires at least %d free GPU bytes before model load; found %d"
            % (minimum_free_vram_bytes, int(free_memory))
        )
    return {
        "free_before_backend_bytes": int(free_memory),
        "total_bytes": int(total_memory),
    }


class _ScriptedMockBackend:
    def __init__(self, responses: Mapping[str, Any]):
        self.responses = dict(responses)

    def generate(
        self,
        request: GenerationRequest,
        *,
        task_id: str,
        explicit_output: Optional[str],
    ) -> GenerationResult:
        started = time.monotonic()
        scripted = explicit_output
        delay_seconds = 0.0
        if scripted is None:
            value = self.responses.get(task_id, self.responses.get(request.example_id))
            if isinstance(value, str):
                scripted = value
            elif isinstance(value, Mapping):
                output = value.get("output")
                delay = value.get("delay_seconds", 0.0)
                if output is not None and not isinstance(output, str):
                    return GenerationResult(
                        status="error",
                        error_type="invalid_mock_response",
                        error_message="scripted mock output must be a string",
                        model_id="mock_role",
                        requested_revision="local",
                        resolved_revision="local",
                    )
                if isinstance(delay, bool) or not isinstance(delay, (int, float)) or delay < 0:
                    return GenerationResult(
                        status="error",
                        error_type="invalid_mock_response",
                        error_message="scripted mock delay must be non-negative",
                        model_id="mock_role",
                        requested_revision="local",
                        resolved_revision="local",
                    )
                scripted = output
                delay_seconds = float(delay)
        if delay_seconds:
            time.sleep(delay_seconds)
        if scripted is None:
            return GenerationResult(
                status="error",
                elapsed_seconds=time.monotonic() - started,
                error_type="mock_response_missing",
                error_message="No scripted response for task %s" % task_id,
                model_id="mock_role",
                requested_revision="local",
                resolved_revision="local",
            )
        return GenerationResult(
            status="success",
            raw_output=scripted,
            elapsed_seconds=time.monotonic() - started,
            model_id="mock_role",
            requested_revision="local",
            resolved_revision="local",
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "backend": "mock_scripted",
            "response_count": len(self.responses),
            "accuracy_measurement": False,
        }

    def close(self) -> None:
        return None


def _create_backend(spec: Mapping[str, Any]) -> Any:
    backend_name = spec["backend"]
    if backend_name == "mock":
        responses = spec.get("mock_responses")
        if not isinstance(responses, Mapping):
            raise AgentWorkerProtocolError("mock_responses must be an object")
        return _ScriptedMockBackend(responses)
    if not isinstance(spec.get("model"), Mapping) or not isinstance(
        spec.get("generation"), Mapping
    ):
        raise AgentWorkerProtocolError("HF worker config is incomplete")
    minimum = _required_integer(spec, "minimum_free_vram_bytes", minimum=1)
    vram = _check_role_vram_before_load(minimum)
    from text2sql.core.backends.hf import HuggingFaceBackend

    backend = HuggingFaceBackend(
        model_config=_model_config(spec["model"]),
        generation_config=_generation_config(spec["generation"]),
        allow_model_download=spec.get("allow_model_download") is True,
    )
    metadata = backend.metadata()
    environment = metadata.get("environment")
    if not isinstance(environment, Mapping):
        backend.close()
        raise RuntimeError("HF backend did not report environment metadata")
    reported_free = environment.get("gpu_memory_free_before_load_bytes")
    reported_total = environment.get("gpu_memory_total_bytes")
    if not isinstance(reported_free, int) or reported_free < minimum:
        backend.close()
        raise RuntimeError(
            "HF backend metadata does not satisfy role VRAM requirement"
        )
    if not isinstance(reported_total, int) or reported_total < minimum:
        backend.close()
        raise RuntimeError("visible GPU total memory is below role requirement")
    if vram["total_bytes"] != reported_total:
        backend.close()
        raise RuntimeError("GPU memory metadata changed during backend initialization")
    return backend


def _validate_generate(payload: Mapping[str, Any]) -> Dict[str, Any]:
    allowed = {
        "schema_version",
        "type",
        "task_id",
        "example_id",
        "iteration",
        "messages",
        "prompt_sha256",
        "mock_output",
    }
    unexpected = set(payload) - allowed
    if unexpected:
        raise AgentWorkerProtocolError(
            "generate request contains unexpected fields: %s"
            % ", ".join(sorted(unexpected))
        )
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise AgentWorkerProtocolError("unsupported request schema")
    task_id = _required_string(payload, "task_id")
    example_id = _required_string(payload, "example_id")
    iteration = _required_integer(payload, "iteration", minimum=1)
    messages_value = payload.get("messages")
    messages = normalize_messages(messages_value)  # type: ignore[arg-type]
    digest = messages_sha256(messages)
    if payload.get("prompt_sha256") != digest:
        raise AgentWorkerProtocolError("request prompt_sha256 mismatch")
    mock_output = payload.get("mock_output")
    if mock_output is not None and not isinstance(mock_output, str):
        raise AgentWorkerProtocolError("mock_output must be a string or null")
    return {
        "task_id": task_id,
        "example_id": example_id,
        "iteration": iteration,
        "messages": messages,
        "prompt_sha256": digest,
        "mock_output": mock_output,
    }


def run_worker(spec_path: Path) -> int:
    backend = None
    try:
        _configure_linux_parent_death_signal()
        spec = _load_spec(spec_path)
        backend = _create_backend(spec)
        backend_metadata = backend.metadata()
        _emit(
            {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "type": "ready",
                "role": spec["role"],
                "worker_id": spec["worker_id"],
                "physical_gpu": spec["physical_gpu"],
                "logical_device": LOGICAL_CUDA_DEVICE,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "launch_attempt": spec.get("launch_attempt"),
                "backend": backend_metadata,
                "ready_at_ns": time.time_ns(),
            }
        )
        print(
            "%s ready for role %s on logical %s"
            % (spec["worker_id"], spec["role"], LOGICAL_CUDA_DEVICE),
            file=sys.stderr,
            flush=True,
        )
        for raw_line in sys.stdin:
            if not raw_line.strip():
                continue
            request_payload = None
            try:
                request_payload = json.loads(raw_line)
                if not isinstance(request_payload, dict):
                    raise AgentWorkerProtocolError("request must be a JSON object")
                request_type = request_payload.get("type")
                if request_type == "shutdown":
                    if set(request_payload) != {"schema_version", "type"} or request_payload.get(
                        "schema_version"
                    ) != PROTOCOL_SCHEMA_VERSION:
                        raise AgentWorkerProtocolError("invalid shutdown request")
                    _emit(
                        {
                            "schema_version": PROTOCOL_SCHEMA_VERSION,
                            "type": "shutdown_ack",
                        }
                    )
                    return 0
                if request_type != "generate":
                    raise AgentWorkerProtocolError("unknown request type")
                task = _validate_generate(request_payload)
                generation_request = GenerationRequest(
                    example_id=task["example_id"],
                    messages=task["messages"],
                )
                if isinstance(backend, _ScriptedMockBackend):
                    generation = backend.generate(
                        generation_request,
                        task_id=task["task_id"],
                        explicit_output=task["mock_output"],
                    )
                else:
                    generation = backend.generate(generation_request)
                _emit(
                    {
                        "schema_version": PROTOCOL_SCHEMA_VERSION,
                        "type": "generation_result",
                        "task_id": task["task_id"],
                        "example_id": task["example_id"],
                        "iteration": task["iteration"],
                        "prompt_sha256": task["prompt_sha256"],
                        "generation": generation.to_dict(),
                    }
                )
            except Exception as exc:
                task_id = None
                if isinstance(locals().get("request_payload"), Mapping):
                    value = request_payload.get("task_id")
                    if isinstance(value, str):
                        task_id = value
                _emit(
                    {
                        "schema_version": PROTOCOL_SCHEMA_VERSION,
                        "type": "protocol_error",
                        "task_id": task_id,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc)[:2000],
                        },
                    }
                )
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        try:
            _emit(
                {
                    "schema_version": PROTOCOL_SCHEMA_VERSION,
                    "type": "startup_error",
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc)[:2000],
                    },
                }
            )
        except Exception:
            pass
        return 2
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                traceback.print_exc(file=sys.stderr)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Long-lived agent role worker")
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args(argv)
    return run_worker(args.spec.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
