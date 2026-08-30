from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from text2sql.core.models import GenerationRequest, GenerationResult


class GenerationBackend(ABC):
    name = "unknown"

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        return {"backend": self.name}

    def close(self) -> None:
        return None

