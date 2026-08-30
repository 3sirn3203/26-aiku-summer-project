from __future__ import annotations

import re

from text2sql.core.models import ParseResult
from text2sql.core.sql_text import first_keyword, split_sql_statements


_FENCED_BLOCK = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SQL_PREFIX = re.compile(r"^\s*SQL\s*:\s*", re.IGNORECASE)
_SQL_KEYWORDS = {
    "SELECT",
    "WITH",
    "INSERT",
    "UPDATE",
    "DELETE",
    "REPLACE",
    "CREATE",
    "DROP",
    "ALTER",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "VACUUM",
    "REINDEX",
    "ANALYZE",
}


def extract_sql(raw_output: str) -> ParseResult:
    if not isinstance(raw_output, str) or not raw_output.strip():
        return ParseResult(
            status="error",
            error_type="sql_parse_error",
            error_message="Model output is empty",
        )
    fenced = _FENCED_BLOCK.findall(raw_output)
    if len(fenced) > 1:
        return ParseResult(
            status="error",
            error_type="sql_parse_error",
            error_message="Model output contains multiple fenced blocks",
        )
    candidate = fenced[0] if fenced else raw_output
    candidate = _SQL_PREFIX.sub("", candidate.strip())
    statements = split_sql_statements(candidate)
    if not statements:
        return ParseResult(
            status="error",
            error_type="sql_parse_error",
            error_message="No SQL statement was found",
        )
    if len(statements) != 1:
        return ParseResult(
            status="error",
            error_type="multiple_statements",
            error_message="Exactly one SQL statement is required",
        )
    statement = statements[0].strip()
    keyword = first_keyword(statement)
    if keyword not in _SQL_KEYWORDS:
        return ParseResult(
            status="error",
            error_type="sql_parse_error",
            error_message="Output does not begin with a recognized SQL statement",
        )
    return ParseResult(status="success", sql=statement)
