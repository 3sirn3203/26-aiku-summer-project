from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @classmethod
    def from_environment(cls) -> "DistributedContext":
        return cls(
            rank=int(os.environ.get("RANK", "0")),
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
        )


def reward_trace_path(run_dir: Path, context: DistributedContext) -> Path:
    if context.world_size == 1:
        return run_dir / "reward_trace.jsonl"
    return run_dir / "reward_trace" / ("rank-%05d.jsonl" % context.rank)


def merge_reward_traces(run_dir: Path, world_size: int) -> Path:
    destination = run_dir / "reward_trace.jsonl"
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as output:
        for rank in range(world_size):
            source = run_dir / "reward_trace" / ("rank-%05d.jsonl" % rank)
            if not source.is_file():
                raise RuntimeError("missing reward trace for rank %d" % rank)
            with source.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    output.write(chunk)
    temporary.replace(destination)
    return destination
