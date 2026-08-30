from __future__ import annotations

import copy
import time
from typing import Any, Dict, Optional

from text2sql.core.backends.base import GenerationBackend
from text2sql.config import GenerationConfig, ModelConfig
from text2sql.core.doctor import (
    EXPECTED_TORCH_CUDA_VERSION,
    EXPECTED_TORCH_VERSION,
    EXPECTED_TRANSFORMERS_VERSION,
    MINIMUM_FREE_VRAM_BYTES,
)
from text2sql.core.models import GenerationRequest, GenerationResult
from text2sql.core.model_source import LOCAL_IDENTITY_PREFIX, prepare_model_config


class HuggingFaceBackend(GenerationBackend):
    """Server-only Qwen backend with lazy torch/transformers imports."""

    name = "huggingface"

    def __init__(
        self,
        model_config: ModelConfig,
        generation_config: GenerationConfig,
        allow_model_download: bool,
    ):
        source = getattr(model_config, "source", "hub")
        checkpoint_identity = getattr(model_config, "checkpoint_identity", None)
        if source == "local" and checkpoint_identity is None:
            model_config, _ = prepare_model_config(model_config)
        self.model_config = model_config
        self.generation_config = generation_config
        self.allow_model_download = allow_model_download
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None
        self._effective_generation_config: Any = None
        self._resolved_revision: Optional[str] = None
        self._environment: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The Hugging Face backend requires a server-compatible torch build "
                "and `pip install -e '.[inference]'`."
            ) from exc

        if torch.__version__.split("+", 1)[0] != EXPECTED_TORCH_VERSION:
            raise RuntimeError(
                "Expected torch %s, found %s"
                % (EXPECTED_TORCH_VERSION, torch.__version__)
            )
        if torch.version.cuda != EXPECTED_TORCH_CUDA_VERSION:
            raise RuntimeError(
                "Expected torch CUDA runtime %s, found %s"
                % (EXPECTED_TORCH_CUDA_VERSION, torch.version.cuda)
            )
        if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
            raise RuntimeError(
                "Expected transformers %s, found %s"
                % (EXPECTED_TRANSFORMERS_VERSION, transformers.__version__)
            )

        if self.model_config.dtype != "float32":
            raise RuntimeError("Initial TITAN Xp inference must use float32")
        if self.model_config.attention_implementation != "eager":
            raise RuntimeError("TITAN Xp inference must use eager attention")
        if self.model_config.trust_remote_code:
            raise RuntimeError("trust_remote_code must remain disabled")
        model_source = getattr(self.model_config, "source", "hub")
        if model_source not in {"hub", "local"}:
            raise RuntimeError("unsupported model source: %s" % model_source)
        if model_source == "local" and self.allow_model_download:
            raise RuntimeError(
                "model downloads cannot be enabled for a local checkpoint"
            )
        if model_source == "local" and not str(
            getattr(self.model_config, "checkpoint_identity", "")
        ).startswith(LOCAL_IDENTITY_PREFIX):
            raise RuntimeError("local checkpoint identity is missing or invalid")
        if not self.model_config.device.startswith("cuda"):
            raise RuntimeError("The server Hugging Face smoke test requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")

        device = torch.device(self.model_config.device)
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        capability = tuple(torch.cuda.get_device_capability(device_index))
        architecture_list = list(torch.cuda.get_arch_list())
        if capability != (6, 1):
            raise RuntimeError(
                "Expected the contracted TITAN Xp capability (6, 1), got %s" % (capability,)
            )
        free_memory, total_memory = torch.cuda.mem_get_info(device_index)
        if int(free_memory) < MINIMUM_FREE_VRAM_BYTES:
            raise RuntimeError(
                "At least %d free GPU bytes are required; found %d"
                % (MINIMUM_FREE_VRAM_BYTES, free_memory)
            )
        try:
            probe = torch.ones((2, 2), dtype=torch.float32, device=device)
            probe_result = probe.matmul(probe)
            torch.cuda.synchronize(device)
            if float(probe_result[0, 0].item()) != 2.0:
                raise RuntimeError("unexpected FP32 CUDA probe result")
            del probe_result, probe
        except Exception as exc:
            raise RuntimeError(
                "The installed PyTorch/CUDA build cannot execute FP32 work on %s: %s"
                % (self.model_config.device, exc)
            ) from exc

        cache_dir = str(self.model_config.cache_dir) if self.model_config.cache_dir else None
        common_kwargs = {"trust_remote_code": False}
        if model_source == "hub":
            common_kwargs.update(
                {
                    "revision": self.model_config.revision,
                    "cache_dir": cache_dir,
                    "local_files_only": not self.allow_model_download,
                }
            )
        else:
            common_kwargs["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_id, **common_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_id,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            **common_kwargs,
        )
        model.to(device)
        model.eval()

        tokenizer_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        model_revision = (
            getattr(model.config, "_commit_hash", None)
            if model_source == "hub"
            else getattr(self.model_config, "checkpoint_identity", None)
        )
        if not model_revision:
            raise RuntimeError(
                "the loaded model did not expose a reproducible checkpoint identity"
            )
        if (
            model_source == "hub"
            and tokenizer_revision
            and model_revision
            and tokenizer_revision != model_revision
        ):
            raise RuntimeError(
                "Tokenizer and model resolved to different revisions: %s != %s"
                % (tokenizer_revision, model_revision)
            )

        effective_generation_config = copy.deepcopy(model.generation_config)
        effective_generation_config.do_sample = False
        effective_generation_config.num_beams = self.generation_config.num_beams
        effective_generation_config.repetition_penalty = (
            self.generation_config.repetition_penalty
        )
        effective_generation_config.temperature = None
        effective_generation_config.top_p = None
        effective_generation_config.top_k = None
        effective_generation_config.max_time = self.generation_config.max_time_seconds
        effective_generation_config.max_new_tokens = self.generation_config.max_new_tokens
        if effective_generation_config.pad_token_id is None:
            effective_generation_config.pad_token_id = tokenizer.pad_token_id
        if effective_generation_config.pad_token_id is None:
            effective_generation_config.pad_token_id = tokenizer.eos_token_id

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._effective_generation_config = effective_generation_config
        self._resolved_revision = model_revision
        self._environment = {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "torch_cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device_index),
            "gpu_capability": list(capability),
            "torch_arch_list": architecture_list,
            "cuda_fp32_probe": "passed",
            "gpu_memory_free_before_load_bytes": int(free_memory),
            "gpu_memory_total_bytes": int(total_memory),
            "tokenizer_resolved_revision": tokenizer_revision,
            "model_source": model_source,
            "checkpoint_identity": model_revision,
        }

    def generate(self, request: GenerationRequest) -> GenerationResult:
        started = time.monotonic()
        try:
            encoded = self._tokenizer.apply_chat_template(
                list(request.messages),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_tokens = int(encoded["input_ids"].shape[-1])
            if input_tokens > self.generation_config.max_input_tokens:
                return GenerationResult(
                    status="error",
                    elapsed_seconds=time.monotonic() - started,
                    error_type="input_too_long",
                    error_message="Prompt has %d tokens; limit is %d"
                    % (input_tokens, self.generation_config.max_input_tokens),
                    input_tokens=input_tokens,
                    model_id=self.model_config.model_id,
                    requested_revision=self.model_config.revision,
                    resolved_revision=self._resolved_revision,
                )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                output = self._model.generate(
                    **encoded,
                    generation_config=self._effective_generation_config,
                )
            generated_ids = output[0, input_tokens:]
            raw_output = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            return GenerationResult(
                status="success",
                raw_output=raw_output,
                elapsed_seconds=time.monotonic() - started,
                input_tokens=input_tokens,
                output_tokens=int(generated_ids.shape[-1]),
                model_id=self.model_config.model_id,
                requested_revision=self.model_config.revision,
                resolved_revision=self._resolved_revision,
            )
        except Exception as exc:
            oom_type = (
                getattr(self._torch.cuda, "OutOfMemoryError", None)
                if self._torch is not None
                else None
            )
            if oom_type is not None and isinstance(exc, oom_type):
                error_type = "cuda_out_of_memory"
                self._torch.cuda.empty_cache()
            else:
                error_type = "generation_error"
            return GenerationResult(
                status="error",
                elapsed_seconds=time.monotonic() - started,
                error_type=error_type,
                error_message="%s: %s" % (type(exc).__name__, exc),
                model_id=self.model_config.model_id,
                requested_revision=self.model_config.revision,
                resolved_revision=self._resolved_revision,
            )

    def metadata(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "model_source": getattr(self.model_config, "source", "hub"),
            "model_id": self.model_config.model_id,
            "requested_revision": self.model_config.revision,
            "resolved_revision": self._resolved_revision,
            "dtype": self.model_config.dtype,
            "device": self.model_config.device,
            "attention_implementation": self.model_config.attention_implementation,
            "generation": {
                "do_sample": False,
                "num_beams": self.generation_config.num_beams,
                "repetition_penalty": self.generation_config.repetition_penalty,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "max_time_seconds": self.generation_config.max_time_seconds,
                "max_new_tokens": self.generation_config.max_new_tokens,
            },
            "allow_model_download": self.allow_model_download,
            "environment": self._environment,
        }

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
