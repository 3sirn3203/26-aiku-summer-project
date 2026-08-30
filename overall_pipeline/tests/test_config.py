from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from text2sql.config import ConfigError, load_config


CODE_ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_contract_config_loads(self) -> None:
        config = load_config(CODE_ROOT / "configs" / "smoke.json")
        self.assertEqual(config.model.model_id, "Qwen/Qwen2.5-Coder-0.5B-Instruct")
        self.assertEqual(
            config.model.revision,
            "ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
        )
        self.assertEqual(config.model.dtype, "float32")
        self.assertEqual(config.model.attention_implementation, "eager")
        self.assertFalse(config.generation.do_sample)
        self.assertEqual(config.generation.num_beams, 1)
        self.assertEqual(config.generation.repetition_penalty, 1.0)
        self.assertEqual(config.generation.max_time_seconds, 120.0)
        self.assertEqual(len(config.smoke.samples), 8)
        self.assertEqual(config.spider.split, "dev")
        self.assertTrue(config.official_evaluation.enabled)
        self.assertEqual(
            config.official_evaluation.evaluator_root,
            (CODE_ROOT / "vendor" / "spider_test_suite_eval").resolve(),
        )
        self.assertEqual(
            config.official_evaluation.test_suite_database_root,
            (
                CODE_ROOT.parent / "data" / "spider_test_suite" / "database"
            ).resolve(),
        )
        self.assertEqual(
            config.official_evaluation.upstream_commit,
            "e97acc546ecbee8fa27fa8dbf025ef61493a876c",
        )
        self.assertFalse(config.official_evaluation.plug_value)
        self.assertFalse(config.official_evaluation.keep_distinct)
        self.assertEqual(config.official_evaluation.timeout_seconds, 60.0)
        self.assertEqual(
            config.official_evaluation.nltk_data_dir,
            (CODE_ROOT.parent / "data" / "nltk_data").resolve(),
        )

    def test_full_dev_config_pins_model_revision_and_output_root(self) -> None:
        config = load_config(CODE_ROOT / "configs" / "evaluate_dev.json")
        self.assertEqual(config.spider.split, "dev")
        self.assertEqual(
            config.model.revision,
            "ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
        )
        self.assertEqual(
            config.output.directory,
            (CODE_ROOT / "outputs" / "evaluation").resolve(),
        )

    def test_local_model_path_is_resolved_relative_to_config(self) -> None:
        payload = self._contract_payload()
        payload["model"].update(
            {"source": "local", "id": "checkpoints/coder-sft", "revision": "local"}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            config = load_config(path)
            self.assertEqual(config.model.source, "local")
            self.assertEqual(
                config.model.model_id,
                str((Path(directory) / "checkpoints" / "coder-sft").resolve()),
            )
            self.assertEqual(config.model.revision, "local")

    def test_local_model_requires_local_revision_sentinel(self) -> None:
        payload = self._contract_payload()
        payload["model"].update(
            {"source": "local", "id": "checkpoint", "revision": "main"}
        )
        self._assert_rejected(payload)

    def test_contract_rejects_remote_code(self) -> None:
        source = CODE_ROOT / "configs" / "smoke.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["model"]["trust_remote_code"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_contract_rejects_non_finite_time_limit(self) -> None:
        source = CODE_ROOT / "configs" / "smoke.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["generation"]["max_time_seconds"] = float("nan")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_official_evaluation_requires_boolean_enabled(self) -> None:
        payload = self._contract_payload()
        payload["official_evaluation"]["enabled"] = 1
        self._assert_rejected(payload)

    def test_official_evaluation_requires_lowercase_full_commit(self) -> None:
        for commit in (
            "e97acc5",
            "E97ACC546ECBEE8FA27FA8DBF025EF61493A876C",
            "z97acc546ecbee8fa27fa8dbf025ef61493a876c",
        ):
            with self.subTest(commit=commit):
                payload = self._contract_payload()
                payload["official_evaluation"]["upstream_commit"] = commit
                self._assert_rejected(payload)

    def test_official_evaluation_rejects_non_contract_flags(self) -> None:
        for key in ("plug_value", "keep_distinct"):
            with self.subTest(key=key):
                payload = self._contract_payload()
                payload["official_evaluation"][key] = True
                self._assert_rejected(payload)

    def test_official_evaluation_requires_positive_finite_timeout(self) -> None:
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout):
                payload = self._contract_payload()
                payload["official_evaluation"]["timeout_seconds"] = timeout
                self._assert_rejected(payload)

    def test_official_evaluation_rejects_invalid_nltk_data_path(self) -> None:
        payload = self._contract_payload()
        payload["official_evaluation"]["nltk_data_dir"] = 123
        self._assert_rejected(payload)

    @staticmethod
    def _contract_payload() -> dict:
        source = CODE_ROOT / "configs" / "smoke.json"
        return json.loads(source.read_text(encoding="utf-8"))

    def _assert_rejected(self, payload: dict) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
