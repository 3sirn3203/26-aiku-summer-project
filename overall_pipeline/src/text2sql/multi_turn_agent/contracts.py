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
    tables: Sequence[str]
    columns: Sequence[str]
    plan: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["tables"] = list(self.tables)
        payload["columns"] = list(self.columns)
        return payload


@dataclass(frozen=True)
class VerifierOutput:
    decision: str
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


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError("%s must be a non-empty string" % field_name)
    return value.strip()


_PLACEHOLDERS = {"step", "instruction", "table", "table_name", "column", "column_name"}


def _non_empty_string_list(value: Any, field_name: str) -> Sequence[str]:
    if not isinstance(value, list) or not value:
        raise ContractError("%s must be a non-empty JSON array" % field_name)
    result = tuple(
        _non_empty_string(item, "%s[%d]" % (field_name, index))
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ContractError("%s must not contain duplicates" % field_name)
    for item in result:
        if item.lower() in _PLACEHOLDERS:
            raise ContractError("%s contains placeholder value: %s" % (field_name, item))
    return result


def _schema_identifiers(serialized_schema: str) -> tuple[set[str], set[str]]:
    tables: set[str] = set()
    columns: set[str] = set()
    current_table: str | None = None
    for line in serialized_schema.splitlines():
        table_match = re.fullmatch(r'Table "((?:[^"]|"")+)"', line)
        if table_match:
            current_table = table_match.group(1).replace('""', '"')
            tables.add(current_table)
            continue
        column_match = re.match(r'  - "((?:[^"]|"")+)"\s+', line)
        if column_match and current_table is not None:
            column = column_match.group(1).replace('""', '"')
            columns.add("%s.%s" % (current_table, column))
    if not tables or not columns:
        raise ContractError("Supplied schema contains no parseable table/column identifiers")
    return tables, columns


def parse_planner_output(raw_output: str, serialized_schema: str) -> PlannerOutput:
    payload = _extract_json_object(raw_output)
    _require_exact_fields(payload, ("tables", "columns", "plan"))
    tables = _non_empty_string_list(payload["tables"], "tables")
    columns = _non_empty_string_list(payload["columns"], "columns")
    plan = _non_empty_string(payload["plan"], "plan")
    if plan.lower() in _PLACEHOLDERS:
        raise ContractError("plan contains a placeholder value")
    schema_tables, schema_columns = _schema_identifiers(serialized_schema)
    unknown_tables = sorted(set(tables) - schema_tables)
    unknown_columns = sorted(set(columns) - schema_columns)
    if unknown_tables:
        raise ContractError("Unknown schema tables: %s" % unknown_tables)
    if unknown_columns:
        raise ContractError("Unknown schema columns: %s" % unknown_columns)
    missing_tables = sorted({column.rsplit(".", 1)[0] for column in columns} - set(tables))
    if missing_tables:
        raise ContractError("Column parent tables missing from tables: %s" % missing_tables)
    return PlannerOutput(tables=tables, columns=columns, plan=plan)


def parse_verifier_output(raw_output: str) -> VerifierOutput:
    payload = _extract_json_object(raw_output)
    _require_exact_fields(payload, ("decision", "feedback"))
    decision = payload["decision"]
    if decision not in ("stop", "continue"):
        raise ContractError("decision must be either 'stop' or 'continue'")
    feedback = _non_empty_string(payload["feedback"], "feedback")
    return VerifierOutput(
        decision=decision,
        feedback=feedback,
    )
