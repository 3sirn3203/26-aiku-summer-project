from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from text2sql.config import ConfigError, ModelConfig


LOCAL_IDENTITY_PREFIX = "local-sha256:"

_TOKENIZER_PAYLOAD_NAMES = {
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "vocab.txt",
}
_INFERENCE_METADATA_NAMES = {
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "merges.txt",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
}


@dataclass(frozen=True)
class LocalCheckpointInspection:
    path: Path
    ok: bool
    identity: Optional[str]
    files: Sequence[str]
    total_bytes: int
    errors: Sequence[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": "local",
            "path": str(self.path),
            "ok": self.ok,
            "identity": self.identity,
            "files": list(self.files),
            "file_count": len(self.files),
            "total_bytes": self.total_bytes,
            "errors": list(self.errors),
        }


def _is_weight_file(path: Path) -> bool:
    name = path.name
    return (
        name == "model.safetensors"
        or (name.startswith("model-") and name.endswith(".safetensors"))
        or name == "pytorch_model.bin"
        or (name.startswith("pytorch_model-") and name.endswith(".bin"))
    )


def _is_inference_file(path: Path) -> bool:
    name = path.name
    return (
        name in _INFERENCE_METADATA_NAMES
        or name in _TOKENIZER_PAYLOAD_NAMES
        or name.startswith("vocab.")
        or _is_weight_file(path)
    )


def _iter_file_bytes(path: Path) -> Iterable[bytes]:
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                return
            yield chunk


def inspect_local_checkpoint(model_path: Path) -> LocalCheckpointInspection:
    path = Path(model_path).expanduser().resolve()
    errors = []
    if not path.exists():
        errors.append("local checkpoint directory does not exist")
    elif not path.is_dir():
        errors.append("local checkpoint path is not a directory")
    if errors:
        return LocalCheckpointInspection(path, False, None, (), 0, tuple(errors))

    top_level_files = tuple(sorted(item for item in path.iterdir() if item.is_file()))
    names = {item.name for item in top_level_files}
    if "config.json" not in names:
        errors.append("local checkpoint is missing config.json")
    weights = tuple(item for item in top_level_files if _is_weight_file(item))
    if not weights:
        if "adapter_config.json" in names or any(
            name.startswith("adapter_model") for name in names
        ):
            errors.append(
                "adapter-only checkpoints are not supported; save or merge a full model"
            )
        else:
            errors.append("local checkpoint is missing full model weights")
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = path / index_name
        if not index_path.is_file():
            continue
        try:
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index_payload.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("weight_map is missing or empty")
            shard_names = list(weight_map.values())
            if any(not isinstance(value, str) for value in shard_names):
                raise ValueError("weight_map contains a non-string shard name")
            if any(Path(value).name != value for value in shard_names):
                raise ValueError("weight_map shard names must be top-level files")
            referenced = set(shard_names)
            missing = sorted(name for name in referenced if not (path / name).is_file())
            if missing:
                errors.append(
                    "%s references missing shard(s): %s"
                    % (index_name, ", ".join(missing[:5]))
                )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append("invalid %s: %s" % (index_name, exc))
    if "tokenizer_config.json" not in names:
        errors.append("local checkpoint is missing tokenizer_config.json")
    if not names.intersection(_TOKENIZER_PAYLOAD_NAMES) and not any(
        name.startswith("vocab.") for name in names
    ):
        errors.append("local checkpoint is missing tokenizer vocabulary files")

    inference_files = tuple(item for item in top_level_files if _is_inference_file(item))
    if errors:
        return LocalCheckpointInspection(
            path,
            False,
            None,
            tuple(item.name for item in inference_files),
            sum(item.stat().st_size for item in inference_files),
            tuple(errors),
        )

    digest = hashlib.sha256()
    digest.update(b"text2sql-local-checkpoint-v1\0")
    total_bytes = 0
    for item in inference_files:
        size = item.stat().st_size
        total_bytes += size
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        for chunk in _iter_file_bytes(item):
            digest.update(chunk)
        digest.update(b"\0")
    identity = LOCAL_IDENTITY_PREFIX + digest.hexdigest()
    return LocalCheckpointInspection(
        path,
        True,
        identity,
        tuple(item.name for item in inference_files),
        total_bytes,
        (),
    )


def prepare_model_config(model: ModelConfig) -> Tuple[ModelConfig, Optional[Dict[str, Any]]]:
    if model.source == "hub":
        return replace(model, checkpoint_identity=None), None
    if model.source != "local":
        raise ConfigError("unsupported model source: %s" % model.source)
    if model.revision != "local":
        raise ConfigError("local model revision must be exactly 'local'")
    inspection = inspect_local_checkpoint(Path(model.model_id))
    if not inspection.ok or inspection.identity is None:
        raise ConfigError(
            "invalid local checkpoint %s: %s"
            % (inspection.path, "; ".join(inspection.errors))
        )
    return (
        replace(
            model,
            model_id=str(inspection.path),
            checkpoint_identity=inspection.identity,
        ),
        inspection.to_dict(),
    )


def expected_model_identity(model: ModelConfig) -> str:
    if model.source == "hub":
        return model.revision
    if (
        not isinstance(model.checkpoint_identity, str)
        or not model.checkpoint_identity.startswith(LOCAL_IDENTITY_PREFIX)
    ):
        raise ConfigError("local checkpoint identity has not been prepared")
    return model.checkpoint_identity


def validate_download_policy(models: Sequence[ModelConfig], allow_model_download: bool) -> None:
    if allow_model_download and any(model.source == "local" for model in models):
        raise ConfigError(
            "--allow-model-download cannot be used with a local checkpoint"
        )
