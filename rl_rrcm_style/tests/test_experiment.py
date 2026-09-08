from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from rrcm_sql.config import Config
from rrcm_sql.data import load_split, official_manifest, split_config
from rrcm_sql.tracking import ExperimentTracker
from rrcm_sql.validation import Validation


class ExperimentTest(unittest.TestCase):
    def test_official_resources_and_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Config()
            cfg.data.split_mode = "official"
            for split in ("train", "dev", "test"):
                source = root / f"{split}.json"
                source.write_text(json.dumps([{"db_id": split, "query": "SELECT 1", "question": "one"}]))
                setattr(cfg.data, f"{split}_json", str(source))
                db = root / ("test_database" if split == "test" else "database") / split
                db.mkdir(parents=True)
                with sqlite3.connect(db / f"{split}.sqlite") as conn:
                    conn.execute("CREATE TABLE t(id INTEGER)")
            schema = lambda db: {"db_id": db, "table_names_original": ["t"],
                "column_names_original": [[-1, "*"], [0, "id"]],
                "column_types": ["text", "number"], "primary_keys": [], "foreign_keys": []}
            for name, dbs in (("tables", ["train", "dev"]), ("test_tables", ["test"])):
                path = root / f"{name}.json"
                path.write_text(json.dumps([schema(db) for db in dbs]))
                setattr(cfg.data, name, str(path))
            cfg.data.database_dir = str(root / "database")
            cfg.data.test_database_dir = str(root / "test_database")
            manifest = official_manifest(cfg)
            self.assertEqual(manifest["splits"]["test"]["count"], 1)
            self.assertEqual(load_split(cfg.data, "validation")[0]["db_id"], "dev")
            resolved = split_config(cfg, "test")
            self.assertEqual(resolved.data.database_dir, cfg.data.test_database_dir)
            self.assertEqual(resolved.data.tables, cfg.data.test_tables)
            self.assertNotEqual(cfg.data.tables, resolved.data.tables)
            (root / "dev.json").write_text((root / "train.json").read_text())
            with self.assertRaisesRegex(ValueError, "overlap"):
                official_manifest(cfg)

    def test_crossed_interval_and_duplicate_suppression(self):
        cfg = Config()
        state = {"step": 26, "validation": {"last_step": 24}}
        validation = Validation(cfg, "/tmp", state, ExperimentTracker())
        cfg.evaluation.enabled = True
        self.assertTrue(validation.due("interval"))
        state["validation"]["last_step"] = 26
        self.assertFalse(validation.due("interval"))
        self.assertFalse(validation.due("end"))

    def test_wandb_axes_and_checkpoint_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config()
            cfg.tracking.backend = "wandb"
            mock = MagicMock()
            mock.init.return_value = SimpleNamespace(id="new-id", entity="team", summary={},
                define_metric=MagicMock(), log=MagicMock(), finish=MagicMock(), config=MagicMock())
            state = {"step": 12, "groups": 40, "tracking": {"run_id": "parent"}}
            tracker = ExperimentTracker()
            with patch.dict("sys.modules", {"wandb": mock}):
                tracker.start(cfg, tmp, state, resume="checkpoint-12")
                tracker.log("rollout", {"reward_mean": 0.5}, state)
                tracker.finish()
            self.assertEqual(mock.init.call_args.kwargs["config"]["parent_run_id"], "parent")
            self.assertIsNone(mock.init.call_args.kwargs["id"])
            self.assertEqual(state["tracking"]["run_id"], "new-id")
            event = mock.init.return_value.log.call_args.args[0]
            self.assertEqual(event["group_step"], 40)
            self.assertEqual(event["optimizer_step"], 12)
            mock.init.return_value.finish.assert_called_once_with(exit_code=0)

    def test_offline_wandb(self):
        try:
            import wandb
        except ImportError:
            self.skipTest("Optional wandb SDK is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config()
            cfg.tracking.backend = "wandb"
            cfg.tracking.mode = "offline"
            tracker = ExperimentTracker()
            try:
                tracker.start(cfg, tmp, {"step": 0})
                tracker.log("dev", {"execution_accuracy": 0.5}, {"step": 1})
            finally:
                tracker.finish()
                wandb.teardown()
            self.assertTrue(list(Path(tmp).glob("wandb/offline-run-*")))


if __name__ == "__main__":
    unittest.main()
