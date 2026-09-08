"""SQLite is executed in bounded, fresh CPU-only subprocesses, never in the trainer."""
from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import sqlparse


class InfrastructureError(RuntimeError):
    """A dataset/evaluator/resource failure must not become a negative policy reward."""


class PredictionResultLimitError(RuntimeError):
    def __init__(self, result):
        super().__init__(result["error"])
        self.result = result


def validate_sql(sql, max_chars=20000):
    if len(sql) > max_chars:
        raise ValueError("SQL exceeds length limit")
    try:
        statements = [s for s in sqlparse.parse(sql) if sqlparse.format(str(s), strip_comments=True).strip()]
    except (sqlparse.exceptions.SQLParseError, RecursionError) as exc:
        raise ValueError("SQL parsing complexity limit exceeded") from exc
    if len(statements) != 1 or statements[0].get_type() != "SELECT":
        raise ValueError("Expected one read-only SELECT or WITH ... SELECT statement")


def worker_call(payload, timeout):
    try:
        result = subprocess.run(
            [sys.executable, "-m", "rrcm_sql.sql_worker"], input=json.dumps(payload),
            text=True, capture_output=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "SQL execution timeout", "kind": "timeout"}
    if result.returncode:
        raise InfrastructureError(f"SQL worker failed: {result.stderr[-2000:]}")
    try:
        response = json.loads(result.stdout)
    except ValueError as exc:
        raise InfrastructureError(f"Invalid SQL worker output: {result.stdout[-1000:]}") from exc
    if response.get("infrastructure_error"):
        raise InfrastructureError(response["infrastructure_error"])
    return response


class Executor:
    def __init__(self, config):
        self.config = config

    def execute(self, path, sql, preview=True):
        try:
            validate_sql(sql, self.config.max_sql_chars)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "kind": "invalid"}
        if not Path(path).is_file():
            raise InfrastructureError(f"Missing database: {path}")
        return worker_call({"op": "execute", "path": str(path), "sql": sql,
                            "preview": preview, "config": asdict(self.config)},
                           self.config.timeout_seconds + 2)

    def observation(self, result):
        # Escape markup even inside JSON strings: DB content cannot inject action tags.
        def encode(value):
            return json.dumps(value, ensure_ascii=False, default=str).replace("<", "\\u003c").replace(">", "\\u003e")
        result = deepcopy(result)
        result.pop("elapsed", None)
        while len(encode(result)) > self.config.max_response_chars and result.get("rows"):
            result["rows"].pop()
            result["truncated"] = True
        if len(encode(result)) > self.config.max_response_chars:
            result = {"ok": result["ok"], "truncated": True,
                      "row_count": result.get("row_count"),
                      "error": "Response metadata exceeded character limit"}
        return "<response>\n" + encode(result) + "\n</response>"


class Judge:
    def __init__(self, executor, database_dir, tables=None):
        self.executor = executor
        self.database_dir = database_dir
        self.exact_match_scorer = None
        if executor.config.evaluator_path and tables:
            from .spider_metrics import ExactMatchScorer
            self.exact_match_scorer = ExactMatchScorer(
                executor.config.evaluator_path, tables, database_dir, executor.config.nltk_data)

    def compare(self, path, predicted, gold):
        cfg = self.executor.config
        result = worker_call({"op": "compare", "path": str(path), "sql": predicted,
                              "gold": gold, "config": asdict(cfg)},
                             2 * cfg.timeout_seconds + 5)
        if not result.get("ok"):
            raise InfrastructureError("Correctness evaluation timed out or failed")
        if result.get("prediction_failure"):
            raise PredictionResultLimitError(result["prediction_failure"])
        return result["match"]

    def score(self, example, final_sql, evaluate_suite=False):
        try:
            return self._score(example, final_sql, evaluate_suite)
        except PredictionResultLimitError as exc:
            return {"outcome": "non_executable", "execution_correct": False,
                    "exact_match": False, "final_execution": exc.result,
                    "test_suite_correct": None}

    def _score(self, example, final_sql, evaluate_suite=False):
        from .data import database_path
        path = database_path(self.database_dir, example["db_id"])
        executed = self.executor.execute(path, final_sql)
        if not executed["ok"]:
            return {"outcome": "non_executable", "execution_correct": False,
                    "exact_match": False, "final_execution": executed,
                    "test_suite_correct": None}
        correct = self.compare(path, final_sql, example["query"])
        exact = (self.exact_match_scorer.score(example, final_sql)["exact_match"]
                 if self.exact_match_scorer else False)
        suite = None
        cfg = self.executor.config
        if evaluate_suite or cfg.reward_metric == "test_suite":
            if not cfg.suite_database_dir:
                raise ValueError("Test-suite evaluation requires suite_database_dir")
            root = Path(cfg.suite_database_dir).resolve()
            directory = (root / example["db_id"]).resolve()
            if root not in directory.parents:
                raise InfrastructureError("Invalid suite db_id")
            variants = sorted(directory.glob("*.sqlite"))
            if len(variants) < 2:
                raise InfrastructureError(f"Need generated test-suite databases, not just original DB: {directory}")
            suite = correct
            for variant in variants:
                # Check gold even if prediction fails: corrupt suites must stop the run.
                gold = self.executor.execute(variant, example["query"])
                if not gold["ok"]:
                    raise InfrastructureError(f"Gold SQL failed on suite database {variant}")
                pred = self.executor.execute(variant, final_sql)
                suite = (pred["ok"] and self.compare(variant, final_sql, example["query"])) and suite
        selected = suite if cfg.reward_metric == "test_suite" else correct
        return {"outcome": "correct" if selected else "executable_incorrect",
                "execution_correct": correct, "exact_match": exact,
                "test_suite_correct": suite,
                "final_execution": executed}


def reward_components(outcome, exact_match, intermediate_count, max_intermediate,
                      alpha=1.5, beta=0.2, non_executable_penalty=0.25):
    if intermediate_count < 0 or max_intermediate < 0:
        raise ValueError("Negative intermediate count")
    if intermediate_count > max_intermediate:
        return {"execution_reward": 0.0, "exact_match_bonus": 0.0,
                "intermediate_penalty": 0.0,
                "failure_penalty": -non_executable_penalty}
    if outcome == "correct":
        return {"execution_reward": 1.0,
                "exact_match_bonus": alpha if exact_match else 0.0,
                "intermediate_penalty": (-beta * intermediate_count / max_intermediate
                                         if max_intermediate else 0.0),
                "failure_penalty": 0.0}
    if outcome == "executable_incorrect":
        return {"execution_reward": 0.0, "exact_match_bonus": 0.0,
                "intermediate_penalty": 0.0, "failure_penalty": 0.0}
    if outcome == "non_executable":
        return {"execution_reward": 0.0, "exact_match_bonus": 0.0,
                "intermediate_penalty": 0.0,
                "failure_penalty": -non_executable_penalty}
    raise ValueError(f"Unknown outcome: {outcome}")


def reward(outcome, intermediate_count, max_intermediate, alpha=1.5, beta=0.2,
           non_executable_penalty=0.25, exact_match=False):
    components = reward_components(outcome, exact_match, intermediate_count, max_intermediate,
                                   alpha, beta, non_executable_penalty)
    return sum(components.values())
