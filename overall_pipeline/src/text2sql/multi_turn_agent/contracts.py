from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence


MAX_ITERATIONS = 3

_JSON_FENCE = re.compile(
    r"\A```json\s*(?P<payload>\{.*\})\s*```\Z",
    re.IGNORECASE | re.DOTALL,
)


class ContractError(ValueError):
    """Raised when a planner or verifier response violates its JSON contract."""


@dataclass(frozen=True)
class PlannerOutput:
    iteration: int
    approach: str
    plan: Sequence[str]
    coder_instruction: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["plan"] = list(self.plan)
        return payload


@dataclass(frozen=True)
class VerifierOutput:
    iteration: int
    decision: str
    reason: str
    feedback: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _reject_constant(value: str) -> None:
    raise ContractError("Non-finite JSON value is not allowed: %s" % value)


def _unique_object(pairs: Sequence[Sequence[Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("Duplicate JSON field: %s" % key)
        result[key] = value
    return result


def _extract_json_object(raw_output: str) -> Mapping[str, Any]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ContractError("Model output is empty")
    candidate = raw_output.strip()
    fence = _JSON_FENCE.fullmatch(candidate)
    if candidate.startswith("```"):
        if fence is None:
            raise ContractError("Expected exactly one complete ```json fenced object")
        candidate = fence.group("payload")
    elif "```" in candidate:
        raise ContractError("JSON output cannot contain a partial or additional fence")
    try:
        payload = json.loads(
            candidate,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except ContractError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError("Invalid JSON object: %s" % exc) from exc
    if not isinstance(payload, Mapping):
        raise ContractError("JSON output must be an object")
    return payload


def _require_exact_fields(payload: Mapping[str, Any], fields: Sequence[str]) -> None:
    expected = set(fields)
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details: List[str] = []
        if missing:
            details.append("missing=%s" % missing)
        if extra:
            details.append("extra=%s" % extra)
        raise ContractError("JSON fields do not match the contract (%s)" % ", ".join(details))


def _require_iteration(value: Any, expected_iteration: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError("iteration must be an integer")
    if value != expected_iteration:
        raise ContractError(
            "iteration must equal the requested iteration %d" % expected_iteration
        )
    if value < 1 or value > MAX_ITERATIONS:
        raise ContractError("iteration must be between 1 and %d" % MAX_ITERATIONS)
    return value


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError("%s must be a non-empty string" % field_name)
    return value.strip()


def parse_planner_output(raw_output: str, expected_iteration: int) -> PlannerOutput:
    payload = _extract_json_object(raw_output)
    _require_exact_fields(
        payload,
        ("iteration", "approach", "plan", "coder_instruction"),
    )
    iteration = _require_iteration(payload["iteration"], expected_iteration)
    approach = payload["approach"]
    if approach not in {"direct", "iterative"}:
        raise ContractError("approach must be either 'direct' or 'iterative'")
    raw_plan = payload["plan"]
    if not isinstance(raw_plan, list) or not raw_plan:
        raise ContractError("plan must be a non-empty JSON array")
    plan = tuple(
        _non_empty_string(step, "plan[%d]" % index)
        for index, step in enumerate(raw_plan)
    )
    coder_instruction = _non_empty_string(
        payload["coder_instruction"], "coder_instruction"
    )
    return PlannerOutput(
        iteration=iteration,
        approach=approach,
        plan=plan,
        coder_instruction=coder_instruction,
    )


def parse_verifier_output(raw_output: str, expected_iteration: int) -> VerifierOutput:
    payload = _extract_json_object(raw_output)
    _require_exact_fields(payload, ("iteration", "decision", "reason", "feedback"))
    iteration = _require_iteration(payload["iteration"], expected_iteration)
    decision = payload["decision"]
    if decision not in {"stop", "continue"}:
        raise ContractError("decision must be either 'stop' or 'continue'")
    reason = _non_empty_string(payload["reason"], "reason")
    feedback = payload["feedback"]
    if not isinstance(feedback, str):
        raise ContractError("feedback must be a string")
    feedback = feedback.strip()
    if decision == "continue" and not feedback:
        raise ContractError("feedback must be non-empty when decision is 'continue'")
    return VerifierOutput(
        iteration=iteration,
        decision=decision,
        reason=reason,
        feedback=feedback,
    )
