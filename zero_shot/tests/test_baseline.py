import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from baseline.config import (Config, DataConfig, EvaluationConfig, GenerationConfig,
                             ModelConfig, load_config)
from baseline.evaluator import _summarize, _validate_suite, evaluate, rescore
from baseline.generation import generate
from baseline.model import load_zero_shot_policy
from rrcm_sql.exploration import action_messages
from rrcm_sql.model import Turn
from rrcm_sql.rollout import answer_only_messages, initial_messages
from rrcm_sql.sql import Executor
from rrcm_sql.config import SQLConfig


class ScriptedPolicy:
    def __init__(self, texts):
        self.texts = iter(texts)
        self.messages = []

    def generate(self, messages, sample=False):
        self.messages.append(messages)
        text = next(self.texts)
        return Turn([1, 2], [3], text)


class BaselineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        db_dir = self.root / "database" / "toy"
        db_dir.mkdir(parents=True)
        with sqlite3.connect(db_dir / "toy.sqlite") as connection:
            connection.execute("CREATE TABLE t(id INTEGER)")
            connection.executemany("INSERT INTO t VALUES (?)", [(1,), (2,)])
        self.database_dir = self.root / "database"
        self.example = {"example_id": "test:0", "db_id": "toy", "question": "List IDs"}
        self.schema = 'CREATE TABLE "t" (\n  "id" number\n);'
        self.executor = Executor(SQLConfig(timeout_seconds=1))

    def tearDown(self):
        self.temp.cleanup()

    def test_checked_in_modes(self):
        repository = Path(__file__).resolve().parents[1]
        single = load_config(repository / "configs/qwen3_1_7b_single_turn_test.json")
        multi = load_config(repository / "configs/qwen3_1_7b_multi_turn_test.json")
        self.assertEqual((single.generation.mode, single.generation.max_intermediate),
                         ("single_turn", 0))
        self.assertEqual((multi.generation.mode, multi.generation.max_intermediate),
                         ("multi_turn", 3))

    def test_adapter_path_is_accepted_by_config(self):
        cfg = ModelConfig(adapter_name_or_path="runs/final_adapter")
        self.assertEqual(cfg.adapter_name_or_path, "runs/final_adapter")

    def test_adapter_is_loaded_on_configured_base(self):
        cfg = Config(
            model=ModelConfig(name_or_path="Qwen/Qwen3-1.7B",
                              adapter_name_or_path="runs/final_adapter",
                              local_files_only=True, device="cpu", dtype="float32"),
            data=DataConfig(str(self.root), str(self.root), str(self.root)),
            generation=GenerationConfig(mode="single_turn", max_intermediate=0),
            sql=SQLConfig(timeout_seconds=1),
            evaluation=EvaluationConfig(evaluator_path=None, nltk_data=None),
        )
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.chat_template = "template"
        base = MagicMock()
        adapter_model = MagicMock()
        peft_config = MagicMock(base_model_name_or_path="Qwen/Qwen3-1.7B")
        with patch("baseline.model.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("baseline.model.AutoModelForCausalLM.from_pretrained", return_value=base), \
             patch("peft.PeftConfig.from_pretrained", return_value=peft_config), \
             patch("peft.PeftModel.from_pretrained", return_value=adapter_model) as load_adapter:
            policy = load_zero_shot_policy(cfg, "cpu")
        load_adapter.assert_called_once()
        self.assertIs(policy.model, adapter_model)

    def test_single_turn_uses_exact_rrcm_prompt_and_one_call(self):
        policy = ScriptedPolicy(["<answer>SELECT id FROM t</answer>"])
        rollout_cfg = GenerationConfig(mode="single_turn", max_intermediate=0).rollout_config()
        record = generate(policy, self.example, self.schema, self.executor,
                          self.database_dir, rollout_cfg)
        expected = action_messages(answer_only_messages(self.example, self.schema), "answer")
        self.assertEqual(policy.messages, [expected])
        self.assertIn("For this turn, output exactly one <answer>",
                      policy.messages[0][0]["content"])
        self.assertEqual(len(record["turns"]), 1)
        self.assertEqual(record["intermediate_count"], 0)
        self.assertNotIn("<intermediate>", policy.messages[0][0]["content"])

    def test_multi_turn_uses_rrcm_free_prompt_and_observation(self):
        policy = ScriptedPolicy([
            "<intermediate>SELECT id FROM t</intermediate>",
            "<answer>SELECT id FROM t</answer>",
        ])
        rollout_cfg = GenerationConfig(mode="multi_turn", max_intermediate=3).rollout_config()
        record = generate(policy, self.example, self.schema, self.executor,
                          self.database_dir, rollout_cfg)
        self.assertEqual(policy.messages[0], initial_messages(self.example, self.schema, 3))
        self.assertIn("2 intermediate calls remain", policy.messages[1][-1]["content"])
        self.assertEqual(len(record["turns"]), 2)
        self.assertEqual(record["intermediate_count"], 1)

    def test_incomplete_suite_is_rejected(self):
        suite = self.root / "suite"
        (suite / "toy").mkdir(parents=True)
        (suite / "toy" / "one.sqlite").touch()
        with self.assertRaisesRegex(ValueError, "Incomplete test-suite"):
            _validate_suite(suite, [self.example])

    def test_summary_keeps_call_and_intermediate_costs(self):
        records = [{
            "outcome": "correct", "scores": {"execution_correct": True,
            "final_execution": {"elapsed": 0.2}}, "intermediate": [],
            "intermediate_count": 0, "termination": "answer", "turns": [{}],
            "input_tokens": 10, "output_tokens": 3, "elapsed": 0.5,
        }]
        summary = _summarize(records)
        self.assertEqual(summary["llm_calls_mean"], 1)
        self.assertEqual(summary["intermediate_mean"], 0)
        self.assertEqual(summary["final_sql_seconds_mean"], 0.2)

    def test_test_only_generation_scoring_and_reports(self):
        test_json = self.root / "test.json"
        tables = self.root / "test_tables.json"
        test_json.write_text(json.dumps([{
            "db_id": "toy", "question": "List IDs", "query": "SELECT id FROM t",
        }]))
        tables.write_text(json.dumps([{
            "db_id": "toy", "table_names_original": ["t"],
            "column_names_original": [[-1, "*"], [0, "id"]],
            "column_types": ["text", "number"], "primary_keys": [], "foreign_keys": [],
        }]))
        cfg = Config(
            model=ModelConfig(device="cpu", dtype="float32"),
            data=DataConfig(str(test_json), str(tables), str(self.database_dir)),
            generation=GenerationConfig(mode="single_turn", max_intermediate=0),
            sql=SQLConfig(timeout_seconds=1),
            evaluation=EvaluationConfig(evaluator_path=None, nltk_data=None),
        )
        cfg.validate()

        class FakePool:
            def __init__(self, _cfg, devices):
                self.devices = devices
            def run(self, jobs, save):
                for index, example, _schema in jobs:
                    self.assert_no_gold(example)
                    save(index, {
                        "example_id": example["example_id"], "db_id": "toy",
                        "question": example["question"], "schema": "schema",
                        "turns": [{}], "intermediate": [], "final_sql": "SELECT id FROM t",
                        "termination": "answer", "outcome": "non_executable", "reward": 0,
                        "scores": {}, "elapsed": 0.1, "mode": "free", "action_trace": [],
                        "seed": None, "policy_version": 0, "input_tokens": 10,
                        "output_tokens": 3, "intermediate_count": 0,
                    })
            @staticmethod
            def assert_no_gold(example):
                if "query" in example:
                    raise AssertionError("Gold SQL leaked into generation worker")
            def close(self):
                pass

        output = self.root / "result"
        with patch("baseline.evaluator.GenerationPool", FakePool):
            summary = evaluate(cfg, output, ["cpu"])
        self.assertEqual(summary["split"], "test")
        self.assertEqual(summary["execution_accuracy"], 1)
        self.assertEqual(summary["llm_calls_mean"], 1)
        self.assertEqual(summary["test_suite_status"], "not_configured")
        self.assertTrue((output / "generation_shards" / "000000.json").is_file())
        self.assertTrue((output / "summary.json").is_file())

        suite = self.root / "suite" / "toy"
        suite.mkdir(parents=True)
        source_db = self.database_dir / "toy" / "toy.sqlite"
        (suite / "toy.sqlite").write_bytes(source_db.read_bytes())
        (suite / "toy_variant.sqlite").write_bytes(source_db.read_bytes())
        rescored = rescore(cfg, output, self.root / "suite")
        self.assertEqual(rescored["test_suite_accuracy"], 1)
        self.assertEqual(rescored["test_suite_status"], "computed")
        self.assertTrue((output / "rescore_config.json").is_file())


if __name__ == "__main__":
    unittest.main()
