from __future__ import annotations

from typing import Dict, List


SYSTEM_PROMPT = (
    "You translate natural-language questions into SQLite queries. "
    "Return exactly one read-only SQLite query and nothing else. "
    "Do not use Markdown, explanations, or multiple statements."
)


def build_messages(question: str, serialized_schema: str) -> List[Dict[str, str]]:
    user_prompt = (
        "Database schema:\n"
        + serialized_schema
        + "\n\nQuestion:\n"
        + question.strip()
        + "\n\nSQL:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

