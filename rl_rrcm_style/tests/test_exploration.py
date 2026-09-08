from dataclasses import replace
from types import SimpleNamespace
import unittest

from rrcm_sql.config import RolloutConfig
from rrcm_sql.exploration import action_messages, group_modes, requested_action, task_seed
from rrcm_sql.runtime.rollout_pool import CombinedEvaluationPool


class ExplorationTest(unittest.TestCase):
    def test_combined_validation_pool_uses_both_pools_and_restores_order(self):
        class Pool:
            def __init__(self, name, devices):
                self.name, self.devices, self.seen = name, devices, []
            def evaluate(self, rows, schemas, policy_version, cfg):
                self.seen.extend(row["index"] for row in rows)
                return [(self.name, row["index"]) for row in rows]
        first = Pool("first", ["cuda:0", "cuda:1"])
        second = Pool("second", ["cuda:2", "cuda:3"])
        combined = CombinedEvaluationPool([first, second])
        rows = [{"index": index} for index in range(9)]
        result = combined.evaluate(rows, {}, 3, object())
        self.assertEqual([index for _, index in result], list(range(9)))
        self.assertEqual(set(first.seen + second.seen), set(range(9)))
        self.assertTrue(first.seen and second.seen)

    def test_group_composition_and_stable_seeds(self):
        cfg = RolloutConfig(group_size=6, prompted_trajectories=3)
        self.assertEqual(group_modes(cfg), ["free"] * 3 + ["prompted_random"] * 3)
        self.assertEqual(task_seed(42, 1, 2), task_seed(42, 1, 2))
        self.assertNotEqual(task_seed(42, 1, 2), task_seed(42, 1, 3))

    def test_action_sampling_and_cap(self):
        answer = SimpleNamespace(random=lambda: 0.1)
        intermediate = SimpleNamespace(random=lambda: 0.9)
        cfg = RolloutConfig(max_intermediate=3, answer_probability=0.5)
        self.assertIsNone(requested_action("free", 0, cfg, answer))
        self.assertEqual(requested_action("prompted_random", 0, cfg, answer), "answer")
        self.assertEqual(requested_action("prompted_random", 0, cfg, intermediate), "intermediate")
        self.assertEqual(requested_action("free", 3, cfg, intermediate), "answer")
        self.assertEqual(requested_action("prompted_random", 3, cfg, intermediate), "answer")

    def test_instruction_is_ephemeral(self):
        history = [{"role": "user", "content": "question"}]
        prompted = action_messages(history, "intermediate")
        self.assertIn("<intermediate>", prompted[-1]["content"])
        self.assertEqual(history[-1]["content"], "question")
        self.assertNotIn("<intermediate>", action_messages(history, None)[-1]["content"])


if __name__ == "__main__":
    unittest.main()
