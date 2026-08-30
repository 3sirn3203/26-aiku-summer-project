from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from text2sql.config import ConfigError, ModelConfig
from text2sql.core.model_source import (
    LOCAL_IDENTITY_PREFIX,
    inspect_local_checkpoint,
    prepare_model_config,
    validate_download_policy,
)


def _write_full_checkpoint(path: Path, weight: bytes = b"weights") -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps({"model_type": "qwen2"}), encoding="utf-8"
    )
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(weight)


def _model(path: Path) -> ModelConfig:
    return ModelConfig(
        model_id=str(path),
        revision="local",
        dtype="float32",
        device="cuda:0",
        attention_implementation="eager",
        trust_remote_code=False,
        cache_dir=None,
        source="local",
    )


class LocalModelSourceTests(unittest.TestCase):
    def test_full_checkpoint_gets_content_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            _write_full_checkpoint(path)
            inspection = inspect_local_checkpoint(path)
            self.assertTrue(inspection.ok)
            self.assertTrue(inspection.identity.startswith(LOCAL_IDENTITY_PREFIX))
            prepared, payload = prepare_model_config(_model(path))
            self.assertEqual(prepared.checkpoint_identity, inspection.identity)
            self.assertEqual(payload["identity"], inspection.identity)

    def test_weight_change_changes_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            _write_full_checkpoint(path, b"first")
            first = inspect_local_checkpoint(path).identity
            (path / "model.safetensors").write_bytes(b"second")
            second = inspect_local_checkpoint(path).identity
            self.assertNotEqual(first, second)

    def test_adapter_only_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter"
            path.mkdir()
            (path / "config.json").write_text("{}", encoding="utf-8")
            (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")
            inspection = inspect_local_checkpoint(path)
            self.assertFalse(inspection.ok)
            self.assertTrue(any("adapter-only" in item for item in inspection.errors))
            with self.assertRaises(ConfigError):
                prepare_model_config(_model(path))

    def test_local_checkpoint_disallows_download_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ConfigError):
                validate_download_policy((_model(Path(directory)),), True)


if __name__ == "__main__":
    unittest.main()
