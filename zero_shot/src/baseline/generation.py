import time

from rrcm_sql.data import database_path
from rrcm_sql.exploration import action_messages, requested_action
from rrcm_sql.rollout import Trajectory, initial_messages, parse_action


def generate(policy, example, schema, executor, database_dir, rollout_cfg):
    """Generate with the exact free-action prompt/turn protocol used by RRCM evaluation."""
    started = time.monotonic()
    result = Trajectory(example["example_id"], example["db_id"], example["question"], schema)
    result.mode = "free"
    messages = initial_messages(example, schema, rollout_cfg.max_intermediate)
    for _ in range(rollout_cfg.max_intermediate + 1):
        requested = requested_action("free", len(result.intermediate), rollout_cfg, None)
        trace = {"requested": requested, "actual": None,
                 "cap_forced": len(result.intermediate) >= rollout_cfg.max_intermediate}
        result.action_trace.append(trace)
        turn = policy.generate(action_messages(messages, requested), sample=False)
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
            result.termination = "answer"
            break
        if len(result.intermediate) >= rollout_cfg.max_intermediate:
            result.termination = "intermediate_limit"
            break
        observation = executor.execute(database_path(database_dir, example["db_id"]), sql)
        result.intermediate.append({"sql": sql, "result": observation})
        remaining = rollout_cfg.max_intermediate - len(result.intermediate)
        suffix = (f"\n{remaining} intermediate calls remain." if remaining else
                  "\nNo intermediate calls remain. You must output <answer> now.")
        messages.extend([{"role": "assistant", "content": turn.text},
                         {"role": "user", "content": executor.observation(observation) + suffix}])
    result.elapsed = time.monotonic() - started
    return result.record()
