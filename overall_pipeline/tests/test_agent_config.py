from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from text2sql.config import ConfigError
from text2sql.multi_turn_agent.config import load_agent_config


CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = CODE_ROOT / "configs" / "multi_turn_multi_agent_zero_shot.json"


class AgentConfigTests(unittest.TestCase):
    def _load_mutated(self, mutate) -> None:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        mutate(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            load_agent_config(path)

    def test_agent_contract_is_pinned_and_role_separated(self) -> None:
        config = load_agent_config(CONFIG_PATH)
        self.assertNotIn("smoke", config.raw)
        self.assertEqual(config.workflow.max_iterations, 3)
        self.assertEqual(config.workflow.observation_max_rows, 5)
        self.assertEqual(config.workflow.observation_max_bytes, 4096)
        self.assertEqual(config.workflow.infrastructure_retry_limit, 1)
        self.assertEqual(config.workflow.execution_concurrency, 2)
        self.assertEqual(config.gpu_pools.planner, (0, 1, 2))
        self.assertEqual(config.gpu_pools.coder, (3, 4))
        self.assertEqual(config.gpu_pools.verifier, (5, 6, 7))
        self.assertEqual(
            config.roles["planner"].model.model_id,
            "Qwen/Qwen2.5-1.5B-Instruct",
        )
        self.assertEqual(
            config.roles["planner"].model.revision,
            "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        )
        self.assertEqual(
            config.roles["coder"].model.model_id,
            "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        )
        self.assertEqual(
            config.roles["coder"].model.revision,
            "ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
        )
        self.assertEqual(
            config.roles["verifier"].model,
            config.roles["planner"].model,
        )
        self.assertTrue(config.roles["planner"].trainable)
        self.assertTrue(config.roles["coder"].trainable)
        self.assertFalse(config.roles["verifier"].trainable)
        self.assertEqual(config.roles["planner"].generation.max_new_tokens, 512)
        self.assertEqual(config.roles["coder"].generation.max_new_tokens, 512)
        self.assertEqual(config.roles["verifier"].generation.max_new_tokens, 384)
        for role in ("planner", "coder", "verifier"):
            generation = config.roles[role].generation
            self.assertEqual(generation.max_time_seconds, 120.0)
            self.assertEqual(generation.max_input_tokens, 8192)
            self.assertFalse(generation.do_sample)
            self.assertEqual(generation.num_beams, 1)
            self.assertEqual(generation.batch_size, 1)
        self.assertEqual(
            config.roles["planner"].minimum_free_vram_bytes, 8 * 1024**3
        )
        self.assertEqual(
            config.roles["coder"].minimum_free_vram_bytes, 4 * 1024**3
        )
        self.assertEqual(
            config.roles["verifier"].minimum_free_vram_bytes, 8 * 1024**3
        )

    def test_gpu_pools_must_not_overlap(self) -> None:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        payload["gpu_pools"]["coder"] = [2, 3]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_agent_config(path)

    def test_workflow_limits_are_fixed(self) -> None:
        mutations = (
            lambda payload: payload["workflow"].__setitem__("max_iterations", 4),
            lambda payload: payload["workflow"].__setitem__(
                "observation_max_rows", 6
            ),
            lambda payload: payload["workflow"].__setitem__(
                "observation_max_bytes", 8192
            ),
            lambda payload: payload["workflow"].__setitem__(
                "infrastructure_retry_limit", 2
            ),
            lambda payload: payload["workflow"].__setitem__(
                "execution_concurrency", 3
            ),
        )
        for mutate in mutations:
            with self.assertRaises(ConfigError):
                self._load_mutated(mutate)

    def test_role_model_and_resource_contracts_are_fixed(self) -> None:
        mutations = (
            lambda payload: payload["roles"]["planner"]["model"].__setitem__(
                "id", "Qwen/another-model"
            ),
            lambda payload: payload["roles"]["coder"]["model"].__setitem__(
                "revision", "0" * 40
            ),
            lambda payload: payload["roles"]["verifier"][
                "generation"
            ].__setitem__("max_new_tokens", 512),
            lambda payload: payload["roles"]["planner"][
                "generation"
            ].__setitem__("max_input_tokens", 4096),
            lambda payload: payload["roles"]["coder"].__setitem__(
                "minimum_free_vram_bytes", 8 * 1024**3
            ),
        )
        for mutate in mutations:
            with self.assertRaises(ConfigError):
                self._load_mutated(mutate)

    def test_local_sft_model_can_replace_one_role_checkpoint(self) -> None:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        payload["roles"]["coder"]["model"].update(
            {
                "source": "local",
                "id": "checkpoints/coder-sft",
                "revision": "local",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            config = load_agent_config(path)
            coder = config.roles["coder"].model
            self.assertEqual(coder.source, "local")
            self.assertEqual(
                coder.model_id,
                str((Path(directory) / "checkpoints" / "coder-sft").resolve()),
            )
            self.assertEqual(coder.revision, "local")
            self.assertEqual(config.roles["planner"].model.source, "hub")

    def test_official_evaluation_cannot_be_disabled(self) -> None:
        with self.assertRaises(ConfigError):
            self._load_mutated(
                lambda payload: payload["official_evaluation"].__setitem__(
                    "enabled", False
                )
            )

    def test_unknown_fields_are_rejected_before_raw_config_is_recorded(self) -> None:
        with self.assertRaises(ConfigError):
            self._load_mutated(
                lambda payload: payload.__setitem__("api_token", "must-not-be-saved")
            )


if __name__ == "__main__":
    unittest.main()
