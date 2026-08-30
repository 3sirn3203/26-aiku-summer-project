from __future__ import annotations

import time
from typing import Any, Dict, Mapping

from text2sql.core.backends.base import GenerationBackend
from text2sql.core.models import GenerationRequest, GenerationResult


class GoldMockBackend(GenerationBackend):
    """Deterministic local-only backend for pipeline validation.

    Gold SQL is held in this isolated response mapping and is never included in
    the generation request or prompt. Results from this backend are not model
    accuracy measurements.
    """

    name = "mock_gold"

    def __init__(self, responses: Mapping[str, str]):
        self._responses = dict(responses)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        started = time.monotonic()
        response = self._responses.get(request.example_id)
        if response is None:
            return GenerationResult(
                status="error",
                elapsed_seconds=time.monotonic() - started,
                error_type="mock_response_missing",
                error_message="No deterministic response for %s" % request.example_id,
                model_id="mock_gold",
                requested_revision="local",
                resolved_revision="local",
            )
        return GenerationResult(
            status="success",
            raw_output=response,
            elapsed_seconds=time.monotonic() - started,
            model_id="mock_gold",
            requested_revision="local",
            resolved_revision="local",
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "response_count": len(self._responses),
            "accuracy_measurement": False,
            "notice": "Gold-backed mock validates plumbing only; it is not an LLM result.",
        }

