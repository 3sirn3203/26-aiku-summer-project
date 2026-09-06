from __future__ import annotations

from pathlib import Path
from typing import Any


def load_tests(loader: Any, _tests: Any, pattern: str | None) -> Any:
    suite_dir = Path(__file__).with_name("rl_vm_step")
    return loader.discover(
        str(suite_dir),
        pattern=pattern or "test*.py",
        top_level_dir=str(suite_dir),
    )
