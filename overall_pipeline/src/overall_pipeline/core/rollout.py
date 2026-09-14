from dataclasses import asdict, dataclass, field
import re


INSTRUCTION = """
Translate the question into SQLite SQL using the full database schema below.
Write one complete read-only SQLite SQL query using tables and columns from the provided schema.
Enclose the SQL query in exactly one matching tag pair:

<answer>...</answer>
Use this tag when you are confident that you can construct the final SQL query.
Submit the final SQL that directly answers the question and end the task.

<intermediate>...</intermediate>
Use this tag when the final SQL is difficult to construct directly and a partial query can help resolve or verify part of the problem.
The SQL will be executed, and its result or error will be returned for the next step.
Intermediate queries may be useful for obtaining execution feedback, such as inspecting how relevant values are stored, checking a candidate join for unexpected duplicates, or testing a filter, subquery or aggregation.
Choose a query whose result can inform the final SQL, and use the returned evidence to revise or confirm your assumptions.

Examples using a toy schema inventory(item_id, category):
To inspect the stored categories:
<intermediate>SELECT DISTINCT category FROM inventory LIMIT 10;</intermediate>
To answer how many items exist:
<answer>SELECT COUNT(*) FROM inventory;</answer>

The content inside the tags must be executable SQL, not the word "SQL", ellipsis, pseudocode, or a description of a query.
Never output bare SQL, Markdown code fences, explanations, or any text outside the tags.
"""

ANSWER_ONLY_INSTRUCTION = """
Translate the question into SQLite SQL using the full database schema below.
Write one complete read-only SQLite SQL query using tables and columns from the provided schema.
Enclose the SQL query in exactly one matching tag pair:

<answer>...</answer>

The content inside the tags must be executable SQL, not the word "SQL", ellipsis, pseudocode, or a description of a query.
Never output bare SQL, Markdown code fences, explanations, or any text outside the tags.
"""


def answer_only_messages(example, schema):
    """Canonical one-turn prompt shared by SFT and single-turn evaluation."""
    content = (ANSWER_ONLY_INSTRUCTION
               + f"\nSchema:\n{schema}\n\nQuestion:\n{example['question']}")
    return [{"role": "user", "content": content}]

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
    reward_components: dict = field(default_factory=dict)
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
