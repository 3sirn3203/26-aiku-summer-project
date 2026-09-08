from dataclasses import asdict, dataclass, field
import re
import time
import random

from .data import database_path
from .sql import reward
from .exploration import requested_action, action_messages


INSTRUCTION = """
Translate the question into SQLite SQL using the full database schema below.
At each step, choose exactly ONE of the following two actions:

1. <answer>SQL</answer>
    - Use this action when you are confident enough to construct the final SQL query.
    - The SQL inside <answer> must directly answer the user's question.
    - This action submits the final SQL and terminates the task.

2. <intermediate>SQL</intermediate>
    - Use this action when the final SQL is difficult to construct reliably in a single step and executing a partial query would help resolve or verify part of the problem.
    - The intermediate SQL is executed against the database, and its result will be returned to you as additional evidence for the next step.
    - Intermediate queries may be useful for obtaining execution feedback, such as:
        - validating a candidate multi-table join by inspecting the joined rows,
        - checking how relevant values are represented in the database,
        - testing a candidate filter, subquery, or aggregation before incorporating it into the final SQL,
        - verifying an intermediate relation needed for a more complex query.
    - Each intermediate query should address a concrete uncertainty or subproblem whose result will help construct the final answer.
    - Do not use <intermediate> if the final SQL can already be constructed confidently.
"""


def initial_messages(example, schema, max_intermediate):
    # Single user message accommodates templates without a system role.
    content = (INSTRUCTION + f"\nAt most {max_intermediate} intermediate calls are allowed.\n"
               + ("You must output <answer> now.\n" if max_intermediate == 0 else "")
               + f"\nSchema:\n{schema}\n\nQuestion:\n{example['question']}")
    return [{"role": "user", "content": content}]


def parse_action(text):
    match = re.fullmatch(r"\s*<(answer|intermediate)>\s*(.*?)\s*</\1>\s*", text, re.DOTALL)
    if not match or not match[2].strip() or re.search(r"</?(?:answer|intermediate|response)>", match[2]):
        raise ValueError("Expected exactly one answer or intermediate action without extra text")
    return match[1], match[2].strip()


@dataclass
class Trajectory:
    example_id: str
    db_id: str
    question: str
    schema: str
    turns: list = field(default_factory=list)
    intermediate: list = field(default_factory=list)
    final_sql: str | None = None
    termination: str = ""
    outcome: str = "non_executable"
    reward: float = 0.0
    scores: dict = field(default_factory=dict)
    elapsed: float = 0.0
    mode: str = "free"
    action_trace: list = field(default_factory=list)
    seed: int | None = None
    policy_version: int = 0

    def record(self):
        result = asdict(self)
        for turn in result["turns"]:
            turn.pop("old_log_probs", None)
            turn.pop("reference_log_probs", None)
        result["input_tokens"] = sum(len(t.prompt_ids) for t in self.turns)
        result["output_tokens"] = sum(len(t.action_ids) for t in self.turns)
        result["intermediate_count"] = len(self.intermediate)
        return result


def rollout(policy, example, schema, executor, judge, cfg, sample=True, evaluate_suite=False,
            mode="free", seed=None, policy_version=0):
    started = time.monotonic()
    result = Trajectory(str(example.get("example_id", "")), example["db_id"], example["question"], schema)
    if mode not in {"free", "prompted_random"}:
        raise ValueError("Unknown exploration mode")
    result.mode, result.seed, result.policy_version = mode, seed, policy_version
    rng = random.Random(seed)
    messages = initial_messages(example, schema, cfg.max_intermediate)
    for _ in range(cfg.max_intermediate + 1):
        requested = requested_action(mode, len(result.intermediate), cfg, rng)
        trace = {"requested": requested, "actual": None,
                 "cap_forced": len(result.intermediate) >= cfg.max_intermediate}
        result.action_trace.append(trace)
        turn = policy.generate(action_messages(messages, requested), sample=sample)
        if turn is None or not turn.action_ids:
            result.termination = "context_limit"
            break
        result.turns.append(turn)
        try:
            action, sql = parse_action(turn.text)
        except ValueError:
            result.termination = "invalid_format"
            break
        trace["actual"] = action
        if requested and action != requested:
            result.termination = ("intermediate_limit" if trace["cap_forced"]
                                  else "action_instruction_violation")
            break
        if action == "answer":
            result.final_sql = sql
            result.scores = judge.score(example, sql, evaluate_suite=evaluate_suite)
            result.outcome = result.scores["outcome"]
            result.termination = "answer"
            break
        if len(result.intermediate) >= cfg.max_intermediate:
            result.termination = "intermediate_limit"
            break
        observation = executor.execute(database_path(judge.database_dir, example["db_id"]), sql)
        result.intermediate.append({"sql": sql, "result": observation})
        remaining = cfg.max_intermediate - len(result.intermediate)
        suffix = (f"\n{remaining} intermediate calls remain." if remaining
                  else "\nNo intermediate calls remain. You must output <answer> now.")
        messages.extend([{"role": "assistant", "content": turn.text},
                         {"role": "user", "content": executor.observation(observation) + suffix}])
    result.reward = reward(result.outcome, len(result.intermediate), cfg.max_intermediate,
                           cfg.efficiency_beta, cfg.non_executable_penalty)
    result.elapsed = time.monotonic() - started
    return result
