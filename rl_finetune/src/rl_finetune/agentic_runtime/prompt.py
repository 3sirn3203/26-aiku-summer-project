from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from rl_finetune.agentic_runtime.models import SQLObservation


_FINAL_INSTRUCTION = (
    "Use the execution observation to correct the draft when needed. "
    "Return exactly one final SQLite SQL statement and no explanation."
)


def build_final_messages(
    initial_messages: Sequence[Mapping[str, str]],
    *,
    draft_raw_output: str,
    observation: SQLObservation,
) -> List[Dict[str, str]]:
    messages = [dict(message) for message in initial_messages]
    messages.append({"role": "assistant", "content": draft_raw_output})
    messages.append(
        {
            "role": "user",
            "content": "%s\n\n%s" % (observation.to_prompt_text(), _FINAL_INSTRUCTION),
        }
    )
    return messages


