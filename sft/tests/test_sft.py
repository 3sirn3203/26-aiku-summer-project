import json
from pathlib import Path
import tempfile
import unittest

from sft.collator import AnswerOnlyCollator
from sft.config import load_config
from sft.data import build_examples


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert tokenize and add_generation_prompt
        assert kwargs == {"enable_thinking": False}
        return [1, 2, 3]

    def encode(self, text, add_special_tokens=False):
        assert text.startswith("<answer>") and "SELECT 1" in text
        return [4, 5]


class SFTTest(unittest.TestCase):
    def test_collator_masks_prompt_and_padding(self):
        collator = AnswerOnlyCollator(FakeTokenizer(), 16, {"enable_thinking": False})
        batch = collator([{"messages": [{"role": "user", "content": "q"}],
                           "response": "<answer>SELECT 1</answer>"}])
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 3, 4, 5, 9]])
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, -100, 4, 5, 9]])

    def test_answer_only_dataset_uses_rrcm_schema_and_no_intermediate(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            data = root / "train.json"
            tables = root / "tables.json"
            data.write_text(json.dumps([{"db_id": "toy", "question": "Count", "query": "SELECT 1"}]))
            tables.write_text(json.dumps([{
                "db_id": "toy", "table_names_original": ["t"],
                "column_names_original": [[-1, "*"], [0, "id"]],
                "column_types": ["text", "number"], "primary_keys": [1],
                "foreign_keys": []}]))
            item = build_examples(data, tables)[0]
            prompt = item["messages"][0]["content"]
            self.assertIn("CREATE TABLE", prompt)
            self.assertIn("<answer>", prompt)
            self.assertNotIn("<intermediate>", prompt)
            self.assertEqual(item["response"], "<answer>\nSELECT 1\n</answer>")

    def test_config_rejects_thinking(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "config.json"
            path.write_text(json.dumps({"model": {"chat_template_kwargs": {"enable_thinking": True}}}))
            with self.assertRaisesRegex(ValueError, "enable_thinking"):
                load_config(path, check_paths=False)

    def test_wandb_requires_project(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "config.json"
            path.write_text(json.dumps({"runtime": {"report_to": "wandb"}}))
            with self.assertRaisesRegex(ValueError, "wandb_project"):
                load_config(path, check_paths=False)

    def test_generation_devices_must_be_unique(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "config.json"
            path.write_text(json.dumps({
                "runtime": {"generation_devices": ["cuda:0", "cuda:0"]}}))
            with self.assertRaisesRegex(ValueError, "must be unique"):
                load_config(path, check_paths=False)

    def test_effective_batch_size_includes_world_size(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "config.json"
            path.write_text(json.dumps({
                "training": {"per_device_batch_size": 1,
                             "gradient_accumulation_steps": 2,
                             "effective_batch_size": 16},
                "runtime": {"expected_world_size": 8}}))
            cfg = load_config(path, check_paths=False)
            self.assertEqual(cfg.training.effective_batch_size, 16)

    def test_effective_batch_size_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "config.json"
            path.write_text(json.dumps({
                "training": {"per_device_batch_size": 1,
                             "gradient_accumulation_steps": 16,
                             "effective_batch_size": 16},
                "runtime": {"expected_world_size": 8}}))
            with self.assertRaisesRegex(ValueError, "per-device batch"):
                load_config(path, check_paths=False)


if __name__ == "__main__":
    unittest.main()
