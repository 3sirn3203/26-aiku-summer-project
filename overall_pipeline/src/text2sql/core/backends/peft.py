from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any, Dict, Optional

from text2sql.config import GenerationConfig, ModelConfig
from text2sql.core.backends.hf import HuggingFaceBackend
from text2sql.core.model_source import (
    ADAPTER_IDENTITY_PREFIX,
    AdapterInspection,
    inspect_peft_adapter,
)


EXPECTED_PEFT_VERSION = "0.14.0"


class PeftAdapterBackend(HuggingFaceBackend):
    """Pinned Hugging Face base model with one local PEFT LoRA adapter."""

    name = "peft_adapter"

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        generation_config: GenerationConfig,
        adapter_dir: Path,
        allow_model_download: bool,
        expected_adapter_identity: Optional[str] = None,
    ) -> None:
        if model_config.source != "hub":
            raise RuntimeError("PEFT adapter evaluation requires a Hub base model")
        inspection = inspect_peft_adapter(
            adapter_dir,
            expected_base_model_id=model_config.model_id,
            expected_base_model_revision=model_config.revision,
        )
        if not inspection.ok or inspection.identity is None:
            raise RuntimeError(
                "invalid PEFT adapter %s: %s"
                % (inspection.path, "; ".join(inspection.errors))
            )
        if expected_adapter_identity is not None:
            if not expected_adapter_identity.startswith(ADAPTER_IDENTITY_PREFIX):
                raise RuntimeError("expected adapter identity is invalid")
            if inspection.identity != expected_adapter_identity:
                raise RuntimeError("PEFT adapter content changed after run creation")
        try:
            peft_version = importlib.metadata.version("peft")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "PEFT adapter evaluation requires peft==%s" % EXPECTED_PEFT_VERSION
            ) from exc
        if peft_version != EXPECTED_PEFT_VERSION:
            raise RuntimeError(
                "Expected peft %s, found %s"
                % (EXPECTED_PEFT_VERSION, peft_version)
            )

        self.adapter_inspection: AdapterInspection = inspection
        super().__init__(
            model_config=model_config,
            generation_config=generation_config,
            allow_model_download=allow_model_download,
        )
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "PEFT adapter evaluation requires peft==%s" % EXPECTED_PEFT_VERSION
            ) from exc
        self._model = PeftModel.from_pretrained(
            self._model,
            str(inspection.path),
            is_trainable=False,
        )
        self._model.to(self._device)
        self._model.eval()
        self._environment["peft_version"] = peft_version
        self._environment["adapter_identity"] = inspection.identity

    def metadata(self) -> Dict[str, Any]:
        payload = super().metadata()
        payload["adapter"] = self.adapter_inspection.to_dict()
        return payload
