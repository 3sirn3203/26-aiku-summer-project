from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rl_vm_step.distributed import (
    DistributedContext,
    merge_reward_traces,
    reward_trace_path,
)


class DistributedTests(unittest.TestCase):
    def test_context_reads_accelerate_environment(self) -> None:
        with patch.dict(
            "os.environ", {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2"}
        ):
            context = DistributedContext.from_environment()
        self.assertEqual(context.rank, 1)
        self.assertEqual(context.local_rank, 1)
        self.assertEqual(context.world_size, 2)
        self.assertFalse(context.is_main_process)

    def test_rank_traces_are_separate_and_merge_in_rank_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for rank, value in ((0, "zero\n"), (1, "one\n")):
                path = reward_trace_path(
                    run_dir, DistributedContext(rank, rank, 2)
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value, encoding="utf-8")
            merged = merge_reward_traces(run_dir, 2)
            content = merged.read_text(encoding="utf-8")
        self.assertEqual(content, "zero\none\n")


if __name__ == "__main__":
    unittest.main()
