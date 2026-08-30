from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from text2sql.config import GenerationConfig, ModelConfig
from text2sql.core.backends.hf import HuggingFaceBackend
from text2sql.core.backends.peft import PeftAdapterBackend
from text2sql.core.model_source import inspect_peft_adapter


def _write_adapter(path: Path) -> None:
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Coder"}),
        encoding="utf-8",
    )
    (path / "text2sql_adapter_provenance.json").write_text(
        json.dumps(
            {
                "base_model_id": "Qwen/Coder",
                "base_model_revision": "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def _model_config() -> ModelConfig:
    return ModelConfig(
        model_id="Qwen/Coder",
        revision="a" * 40,
        dtype="float32",
        device="cuda:0",
        attention_implementation="eager",
        trust_remote_code=False,
        cache_dir=None,
        source="hub",
    )


def _generation_config() -> GenerationConfig:
    return GenerationConfig(
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
        max_time_seconds=10.0,
        max_input_tokens=512,
        max_new_tokens=32,
        batch_size=1,
    )


class _WrappedModel:
    def to(self, _device):
        return self

    def eval(self):
        return self


class _FakePeftModel:
    calls = []

    @classmethod
    def from_pretrained(cls, base, path, is_trainable):
        cls.calls.append((base, path, is_trainable))
        return _WrappedModel()


class PeftBackendTests(unittest.TestCase):
    def test_loads_only_the_contracted_local_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            _write_adapter(adapter)
            identity = inspect_peft_adapter(adapter).identity

            def fake_hf_init(instance, model_config, generation_config, allow_model_download):
                instance.model_config = model_config
                instance.generation_config = generation_config
                instance.allow_model_download = allow_model_download
                instance._model = object()
                instance._device = "cuda:0"
                instance._resolved_revision = model_config.revision
                instance._environment = {}

            fake_peft = types.ModuleType("peft")
            fake_peft.PeftModel = _FakePeftModel
            _FakePeftModel.calls.clear()
            with mock.patch.object(HuggingFaceBackend, "__init__", fake_hf_init), mock.patch(
                "importlib.metadata.version", return_value="0.14.0"
            ), mock.patch.dict(sys.modules, {"peft": fake_peft}):
                backend = PeftAdapterBackend(
                    model_config=_model_config(),
                    generation_config=_generation_config(),
                    adapter_dir=adapter,
                    allow_model_download=False,
                    expected_adapter_identity=identity,
                )

            self.assertEqual(len(_FakePeftModel.calls), 1)
            self.assertEqual(_FakePeftModel.calls[0][1], str(adapter.resolve()))
            self.assertFalse(_FakePeftModel.calls[0][2])
            self.assertEqual(backend.metadata()["adapter"]["identity"], identity)

    def test_rejects_changed_adapter_before_base_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            _write_adapter(adapter)
            identity = inspect_peft_adapter(adapter).identity
            (adapter / "adapter_model.safetensors").write_bytes(b"changed")
            with mock.patch.object(HuggingFaceBackend, "__init__") as base_init:
                with self.assertRaisesRegex(RuntimeError, "content changed"):
                    PeftAdapterBackend(
                        model_config=_model_config(),
                        generation_config=_generation_config(),
                        adapter_dir=adapter,
                        allow_model_download=False,
                        expected_adapter_identity=identity,
                    )
            base_init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
