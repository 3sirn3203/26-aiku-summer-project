from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from text2sql.core.backends.hf import HuggingFaceBackend
from text2sql.core.models import GenerationRequest


class _FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    @property
    def shape(self):
        return (1, len(self.values))

    def to(self, _device):
        return self

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) == 2 and isinstance(key[1], slice):
                return _FakeTensor(self.values[key[1]])
            return _FakeScalar(2.0)
        if isinstance(key, slice):
            return _FakeTensor(self.values[key])
        return self.values[key]

    def matmul(self, _other):
        return self


class _FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _InferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, _exc_type, _exc, _traceback):
        return False


class _FakeOOM(Exception):
    pass


class _FakeCuda:
    OutOfMemoryError = _FakeOOM

    def __init__(self):
        self.cache_cleared = False

    def empty_cache(self):
        self.cache_cleared = True


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, input_ids=(10, 11)):
        self.input_ids = input_ids
        self.calls = []
        self.decoded = None

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {
            "input_ids": _FakeTensor(self.input_ids),
            "attention_mask": _FakeTensor([1] * len(self.input_ids)),
        }

    def decode(self, generated_ids, skip_special_tokens):
        self.decoded = (generated_ids.values, skip_special_tokens)
        return "SELECT 1"


class _FakeModel:
    def __init__(self, error=None):
        self.error = error
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return _FakeTensor([10, 11, 90, 91])


def _backend(input_ids=(10, 11), model_error=None):
    backend = object.__new__(HuggingFaceBackend)
    backend.model_config = SimpleNamespace(
        model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        revision="commit",
        dtype="float32",
        device="cuda:0",
        attention_implementation="eager",
    )
    backend.generation_config = SimpleNamespace(
        max_input_tokens=8,
        max_new_tokens=4,
        max_time_seconds=120.0,
        num_beams=1,
        repetition_penalty=1.0,
    )
    backend.allow_model_download = False
    backend._torch = SimpleNamespace(
        inference_mode=lambda: _InferenceMode(),
        cuda=_FakeCuda(),
    )
    backend._tokenizer = _FakeTokenizer(input_ids)
    backend._model = _FakeModel(model_error)
    backend._device = "cuda:0"
    backend._effective_generation_config = object()
    backend._resolved_revision = "commit"
    backend._environment = {}
    return backend


class HuggingFaceBackendTests(unittest.TestCase):
    def test_generate_uses_messages_and_decodes_only_new_tokens(self):
        backend = _backend()
        request = GenerationRequest(
            example_id="dev:1",
            messages=({"role": "user", "content": "question"},),
        )

        result = backend.generate(request)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.raw_output, "SELECT 1")
        self.assertEqual(result.input_tokens, 2)
        self.assertEqual(result.output_tokens, 2)
        self.assertEqual(backend._tokenizer.decoded, ([90, 91], True))
        self.assertEqual(
            backend._model.kwargs["generation_config"],
            backend._effective_generation_config,
        )
        messages, kwargs = backend._tokenizer.calls[0]
        self.assertEqual(messages, list(request.messages))
        self.assertTrue(kwargs["add_generation_prompt"])
        self.assertTrue(kwargs["tokenize"])

    def test_input_too_long_does_not_call_model(self):
        backend = _backend(input_ids=range(9))
        result = backend.generate(
            GenerationRequest(
                example_id="dev:1",
                messages=({"role": "user", "content": "question"},),
            )
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "input_too_long")
        self.assertIsNone(backend._model.kwargs)

    def test_cuda_oom_is_structured_and_cache_is_cleared(self):
        backend = _backend(model_error=_FakeOOM("out of memory"))
        result = backend.generate(
            GenerationRequest(
                example_id="dev:1",
                messages=({"role": "user", "content": "question"},),
            )
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "cuda_out_of_memory")
        self.assertTrue(backend._torch.cuda.cache_cleared)

    def test_load_applies_titan_xp_and_greedy_contract(self):
        calls = {"tokenizer": [], "model": []}
        tokenizer = SimpleNamespace(
            init_kwargs={"_commit_hash": "resolved-sha"},
            pad_token_id=0,
            eos_token_id=1,
            chat_template="embedded-template",
        )
        model = SimpleNamespace(
            config=SimpleNamespace(_commit_hash="resolved-sha"),
            generation_config=SimpleNamespace(
                do_sample=True,
                num_beams=1,
                repetition_penalty=1.1,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                max_time=None,
                max_new_tokens=2048,
                pad_token_id=0,
            ),
            to=lambda _device: None,
            eval=lambda: None,
        )

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                calls["tokenizer"].append((model_id, kwargs))
                return tokenizer

        class AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                calls["model"].append((model_id, kwargs))
                return model

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.__version__ = "4.46.3"
        fake_transformers.AutoTokenizer = AutoTokenizer
        fake_transformers.AutoModelForCausalLM = AutoModelForCausalLM

        fake_cuda = SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_capability=lambda _index: (6, 1),
            get_arch_list=lambda: ["sm_60"],
            mem_get_info=lambda _index: (10 * 1024**3, 12 * 1024**3),
            synchronize=lambda _device: None,
            get_device_name=lambda _index: "NVIDIA TITAN Xp",
        )
        fake_torch = types.ModuleType("torch")
        fake_torch.__version__ = "2.5.1+cu121"
        fake_torch.version = SimpleNamespace(cuda="12.1")
        fake_torch.cuda = fake_cuda
        fake_torch.float32 = "float32"
        fake_torch.device = lambda value: SimpleNamespace(index=0, value=value)
        fake_torch.ones = lambda *_args, **_kwargs: _FakeTensor([1, 1, 1, 1])

        model_config = SimpleNamespace(
            model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
            revision="main",
            dtype="float32",
            device="cuda:0",
            attention_implementation="eager",
            trust_remote_code=False,
            cache_dir=None,
        )
        generation_config = SimpleNamespace(
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
            max_time_seconds=120.0,
            max_input_tokens=8192,
            max_new_tokens=512,
            batch_size=1,
        )

        with mock.patch.dict(
            sys.modules,
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            backend = HuggingFaceBackend(
                model_config,
                generation_config,
                allow_model_download=False,
            )
            tokenizer.init_kwargs = {}
            tokenizer.chat_template = None
            model.config._commit_hash = None
            with tempfile.TemporaryDirectory() as temp_dir:
                local_model_path = Path(temp_dir) / "coder-sft"
                local_model_path.mkdir()
                expected_chat_template = "{{ messages }}"
                (local_model_path / "chat_template.jinja").write_text(
                    expected_chat_template,
                    encoding="utf-8",
                )
                local_config = SimpleNamespace(
                    model_id=str(local_model_path),
                    revision="local",
                    dtype="float32",
                    device="cuda:0",
                    attention_implementation="eager",
                    trust_remote_code=False,
                    cache_dir=None,
                    source="local",
                    checkpoint_identity="local-sha256:" + "a" * 64,
                )
                local_backend = HuggingFaceBackend(
                    local_config,
                    generation_config,
                    allow_model_download=False,
                )

                self.assertEqual(local_backend._tokenizer.chat_template, expected_chat_template)
                self.assertEqual(
                    local_backend._environment["tokenizer_chat_template_source"],
                    "chat_template.jinja",
                )

        self.assertEqual(calls["tokenizer"][0][1]["revision"], "main")
        self.assertTrue(calls["tokenizer"][0][1]["local_files_only"])
        self.assertEqual(calls["model"][0][1]["torch_dtype"], "float32")
        effective = backend._effective_generation_config
        self.assertFalse(effective.do_sample)
        self.assertEqual(effective.repetition_penalty, 1.0)
        self.assertIsNone(effective.temperature)
        self.assertIsNone(effective.top_p)
        self.assertIsNone(effective.top_k)
        self.assertEqual(effective.max_time, 120.0)
        self.assertEqual(effective.max_new_tokens, 512)
        self.assertEqual(backend._resolved_revision, "resolved-sha")
        self.assertEqual(backend._environment["cuda_fp32_probe"], "passed")
        self.assertEqual(
            backend._environment["tokenizer_chat_template_source"],
            "tokenizer_config",
        )
        local_tokenizer_kwargs = calls["tokenizer"][1][1]
        local_model_kwargs = calls["model"][1][1]
        self.assertEqual(calls["tokenizer"][1][0], str(local_model_path))
        self.assertNotIn("revision", local_tokenizer_kwargs)
        self.assertNotIn("cache_dir", local_tokenizer_kwargs)
        self.assertTrue(local_tokenizer_kwargs["local_files_only"])
        self.assertNotIn("revision", local_model_kwargs)
        self.assertEqual(
            local_backend._resolved_revision, "local-sha256:" + "a" * 64
        )


if __name__ == "__main__":
    unittest.main()
