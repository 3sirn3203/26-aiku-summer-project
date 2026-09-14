"""Helpers for constraining an action at a protocol boundary."""
from copy import deepcopy


def action_messages(history, action):
    messages = deepcopy(history)
    if action:
        messages[-1]["content"] += f"\nFor this turn, output exactly one <{action}>...</{action}> action."
    return messages
