from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from text2sql.config import AppConfig
from rl_finetune.dataset import build_prompt_records
from rl_vm_step.reference import build_gold_reference


def build_vm_step_records(
    config: AppConfig,
    *,
    limit: Optional[int] = None,
    offset: int = 0,
    indices: Optional[Sequence[int]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build prompts and attach gold VM metadata without prompt leakage."""
    base_records = build_prompt_records(
        config, limit=limit, offset=offset, indices=indices
    )
    eligible: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    total = len(base_records)
    for position, base in enumerate(base_records, start=1):
        reference = build_gold_reference(base, config.execution)
        if reference["status"] != "ready":
            rejected.append(reference)
        else:
            record = dict(base)
            record["gold_reference"] = reference
            eligible.append(record)
        if progress_callback is not None:
            progress_callback(position, total)
    return eligible, rejected


def gold_reference_rows(records: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [record["gold_reference"] for record in records]
