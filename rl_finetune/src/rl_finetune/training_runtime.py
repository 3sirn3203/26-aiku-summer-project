from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any, Dict

from text2sql.config import AppConfig, ConfigError


TRAINING_DEPENDENCIES = {
    "datasets": "datasets",
    "trl": "trl",
    "peft": "peft",
    "accelerate": "accelerate",
    "transformers": "transformers",
    "torch": "torch",
}
PINNED_TRAINING_VERSIONS = {
    "transformers": "4.46.3",
    "trl": "0.14.0",
    "peft": "0.14.0",
}
TORCH_VERSION_PREFIX = "2.5.1"


def check_training_dependencies() -> Dict[str, str]:
    if sys.version_info[:2] != (3, 11):
        raise ConfigError("RL fine-tuning requires Python 3.11")
    versions: Dict[str, str] = {}
    missing = []
    for distribution, import_name in TRAINING_DEPENDENCIES.items():
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(import_name)
    if missing:
        raise ConfigError(
            "RL fine-tuning dependencies are missing: %s. "
            "Install the current pipeline and RL package with: "
            "python -m pip install -e './overall_pipeline[official-eval]' && "
            "python -m pip install -e ./rl_finetune. "
            "Install the server-compatible PyTorch CUDA build first if torch is missing."
            % ", ".join(sorted(missing))
        )
    mismatched = []
    for distribution, expected in PINNED_TRAINING_VERSIONS.items():
        if versions.get(distribution) != expected:
            mismatched.append(
                "%s==%s required, found %s"
                % (distribution, expected, versions.get(distribution))
            )
    torch_version = versions.get("torch", "")
    if not torch_version.startswith(TORCH_VERSION_PREFIX):
        mismatched.append(
            "torch %s.x required by the TITAN Xp training contract, found %s"
            % (TORCH_VERSION_PREFIX, torch_version)
        )
    if mismatched:
        raise ConfigError(
            "RL fine-tuning dependency versions are incompatible: %s. "
            "Use a fresh conda env, or uninstall the mismatched packages and run: "
            "python -m pip install -e ./rl_finetune"
            % "; ".join(mismatched)
        )
    return versions


def load_model_and_tokenizer(config: AppConfig, *, training_dtype: Any = None) -> Any:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = str(config.model.cache_dir) if config.model.cache_dir else None
    common_kwargs = {
        "revision": config.model.revision,
        "cache_dir": cache_dir,
        "trust_remote_code": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id, **common_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        torch_dtype=training_dtype or torch.float32,
        attn_implementation=config.model.attention_implementation,
        **common_kwargs,
    )
    return model, tokenizer


def create_lora_config(
    *, r: int, alpha: int, dropout: float, target_modules: str
) -> Any:
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )


def write_adapter_provenance(
    adapter_dir: Path,
    config: AppConfig,
    *,
    training_run_type: str,
) -> None:
    """Write the base checkpoint contract beside adapter weights."""

    payload = {
        "schema_version": 1,
        "training_run_type": training_run_type,
        "base_model_id": config.model.model_id,
        "base_model_revision": config.model.revision,
        "base_model_source": config.model.source,
    }
    path = Path(adapter_dir) / "text2sql_adapter_provenance.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
