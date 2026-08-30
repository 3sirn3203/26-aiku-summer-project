"""Compatibility import for the current pipeline PEFT generation backend."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

from text2sql.config import GenerationConfig, ModelConfig
from text2sql.core.backends.peft import PeftAdapterBackend


class AdapterBackend(PeftAdapterBackend):
    """Backwards-compatible constructor used by the migrated RL commands."""

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        generation_config: GenerationConfig,
        adapter_dir: Path,
        device: Optional[str] = None,
        allow_model_download: bool = False,
    ) -> None:
        if device is not None:
            model_config = replace(model_config, device=device)
        super().__init__(
            model_config=model_config,
            generation_config=generation_config,
            adapter_dir=adapter_dir,
            allow_model_download=allow_model_download,
        )
