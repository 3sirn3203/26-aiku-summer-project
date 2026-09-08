"""External action exploration; evaluation defaults to the free policy."""
from copy import deepcopy
import hashlib
import random


def task_seed(seed, group, index):
    return int.from_bytes(hashlib.sha256(f"{seed}:{group}:{index}".encode()).digest()[:8], "big") % (2**31)


def group_modes(cfg):
    return ["free"] * (cfg.group_size - cfg.prompted_trajectories) + ["prompted_random"] * cfg.prompted_trajectories


def requested_action(mode, count, cfg, rng):
    if count >= cfg.max_intermediate:
        return "answer"
    if mode == "free":
        return None
    return "answer" if rng.random() < cfg.answer_probability else "intermediate"


def action_messages(history, action):
    messages = deepcopy(history)
    if action:
        messages[-1]["content"] += f"\nFor this turn, output exactly one <{action}>...</{action}> action."
    return messages
