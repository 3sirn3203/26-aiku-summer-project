from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from rrcm_sql.config import Config, ModelConfig, RolloutConfig
from rrcm_sql.data import prepare
from rrcm_sql.model import HFPolicy, Turn, action_log_probs, decode_action, load_model, normalize_token_ids
from rrcm_sql.train import advantages, grpo_terms, rl_optimizer_step, sft, train, update_rl_amp_state
from rrcm_sql.evaluate import evaluate


def _fsdp_update_smoke(rank, port, result_queue):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from rrcm_sql.runtime.fsdp import _full_state, _optimizer_step
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        torch.manual_seed(9)
        module = torch.nn.Linear(8, 4)
        before = module.weight.detach().clone()
        model = FSDP(module, device_id=rank)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        cuda = torch.device(f"cuda:{rank}")
        model(torch.ones(2, 8, device=cuda)).square().mean().backward()
        update = _optimizer_step(model, optimizer, scaler, 1.0, cuda)
        state = _full_state(model)
        if rank == 0:
            result_queue.put((not update["amp_step_skipped"],
                              not torch.equal(before, state["weight"])))
    finally:
        dist.destroy_process_group()


def _scripted_rollout_worker(device, cfg, tasks, results):
    from rrcm_sql.runtime.rollout_pool import _worker
    counter = [0]
    def generate(policy, messages, sample=True):
        if sample:
            counter[0] += 1
        sql = "2" if sample and counter[0] % 2 == 0 else "1"
        text = f"<answer> SELECT {sql} </answer>"
        return Turn(policy.encode(messages), policy.tokenizer.encode(text, add_special_tokens=False), text)
    with patch.object(HFPolicy, "generate", generate):
        _worker(device, cfg, tasks, results)


def _scripted_fsdp_rank(rank, cfg, resume, port, result_queue):
    from rrcm_sql.runtime.fsdp import _rank_main
    with patch("rrcm_sql.runtime.rollout_pool._worker", _scripted_rollout_worker):
        _rank_main(rank, cfg, resume, port, result_queue)


class TrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        vocab = {token: index for index, token in enumerate(
            ["[UNK]", "[PAD]", "[EOS]", "<answer>", "</answer>", "<intermediate>", "</intermediate>",
             "SELECT", "1", "2", "id", "FROM", "t", "USER:", "ASSISTANT:"])}
        tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
        tokenizer.save_pretrained(self.base)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=1024, n_embd=16,
                                          n_layer=1, n_head=2, eos_token_id=2, bos_token_id=2, pad_token_id=1))
        model.save_pretrained(self.base)
        self.cfg = Config(model=ModelConfig(name_or_path=str(self.base), mode="full", device="cpu",
                                            dtype="float32", chat_format="plain", gradient_checkpointing=True))
        schemas, rows = [], []
        for db in ("a", "b"):
            directory = self.root / "db" / db
            directory.mkdir(parents=True)
            with sqlite3.connect(directory / f"{db}.sqlite") as c:
                c.execute("CREATE TABLE t(id INTEGER)")
            schemas.append({"db_id": db, "table_names_original": ["t"],
                            "column_names_original": [[-1,"*"], [0,"id"]], "column_types": ["text","number"],
                            "primary_keys": [], "foreign_keys": []})
            rows.append({"db_id": db, "question": "Return one", "query": "SELECT 1"})
        (self.root / "tables.json").write_text(json.dumps(schemas))
        (self.root / "source.json").write_text(json.dumps(rows))
        prepare(self.root / "source.json", self.root / "prepared", seed=42)
        self.cfg.data.prepared_dir = str(self.root / "prepared")
        self.cfg.data.database_dir = str(self.root / "db")
        self.cfg.data.tables = str(self.root / "tables.json")
        self.cfg.train.output_dir = str(self.root / "run")
        self.cfg.train.max_steps = 1
        self.cfg.train.max_groups = 1
        self.cfg.train.learning_rate = 1e-3
        self.cfg.train.save_steps = 1
        self.cfg.rollout.group_size = 2
        self.cfg.rollout.max_new_tokens = 4

    def tearDown(self):
        self.temp.cleanup()

    def test_advantages_and_clipping(self):
        self.assertIsNone(advantages([0,0,0]))
        self.assertTrue(torch.allclose(advantages([1,0]), torch.tensor([1.0,-1.0])))
        current = torch.tensor([0.0, 0.0], requires_grad=True)
        terms, kl = grpo_terms(current, torch.tensor([-1.0, 1.0]), current.detach(), 1.0, 0.2, 0.1)
        self.assertAlmostEqual(terms[0].item(), -1.2, places=5)
        self.assertEqual(kl.sum().item(), 0)
        terms.sum().backward()
        self.assertEqual(current.grad[0].item(), 0)
        self.assertLess(current.grad[1].item(), 0)

    def test_tokenizer_output_normalization(self):
        from transformers import BatchEncoding
        from types import SimpleNamespace
        expected = [1, 2, 3]
        self.assertEqual(normalize_token_ids(expected), expected)
        self.assertEqual(normalize_token_ids(torch.tensor([expected])), expected)
        self.assertEqual(normalize_token_ids(BatchEncoding({"input_ids": expected})), expected)
        self.assertEqual(normalize_token_ids(SimpleNamespace(ids=expected)), expected)
        with self.assertRaises(TypeError):
            normalize_token_ids([[1], [2]])
        with self.assertRaises(TypeError):
            normalize_token_ids({"attention_mask": expected})

    def test_rl_amp_overflow_skip_and_state(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        next(model.parameters()).grad = torch.full_like(next(model.parameters()), float("inf"))

        class FakeScaler:
            def __init__(self): self.scale = 1024.0
            def is_enabled(self): return True
            def get_scale(self): return self.scale
            def unscale_(self, optimizer): pass
            def step(self, optimizer): pass
            def update(self): self.scale /= 2

        result = rl_optimizer_step(model, optimizer, FakeScaler(), 1.0)
        self.assertTrue(result["amp_step_skipped"])
        self.assertEqual(result["loss_scale_before"], 1024.0)
        self.assertEqual(result["loss_scale_after"], 512.0)
        state = {"step": 3, "amp_skipped_updates": 0, "consecutive_amp_skips": 0}
        self.assertFalse(update_rl_amp_state(state, result))
        self.assertEqual(state["step"], 3)
        self.assertEqual(state["amp_skipped_updates"], 1)
        self.assertEqual(state["consecutive_amp_skips"], 1)

    def test_rl_success_resets_amp_skip_streak(self):
        state = {"step": 3, "amp_skipped_updates": 2, "consecutive_amp_skips": 2}
        self.assertTrue(update_rl_amp_state(state, {"amp_step_skipped": False}))
        self.assertEqual(state["step"], 4)
        self.assertEqual(state["amp_skipped_updates"], 2)
        self.assertEqual(state["consecutive_amp_skips"], 0)

    def test_generation_and_action_mask(self):
        model, tokenizer = load_model(self.cfg.model)
        policy = HFPolicy(model, tokenizer, self.cfg.model, self.cfg.rollout)
        turn = policy.generate([{"role": "user", "content": "SELECT 1"}], sample=True)
        self.assertGreater(len(turn.action_ids), 0)
        logp = action_log_probs(model, turn, 1.0, self.cfg.model)
        self.assertEqual(logp.numel(), len(turn.action_ids))
        ids = torch.tensor([turn.prompt_ids + turn.action_ids])
        full = model(input_ids=ids).logits.float().log_softmax(-1)
        for offset, token in enumerate(turn.action_ids):
            self.assertAlmostEqual(logp[offset].item(), full[0, len(turn.prompt_ids)-1+offset, token].item(), places=5)
        model.train()
        (-logp.mean()).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
        policy.config = replace(policy.config, max_context_tokens=1)
        self.assertIsNone(policy.generate([{"role": "user", "content": "SELECT 1"}]))

    def scripted(self):
        counter = [0]
        def generate(policy, messages, sample=True):
            counter[0] += 1
            sql = "1" if counter[0] % 2 else "2"
            text = f"<answer> SELECT {sql} </answer>"
            return Turn(policy.encode(messages), policy.tokenizer.encode(text, add_special_tokens=False), text)
        return generate

    def test_actual_grpo_update_resume_and_evaluate(self):
        before = GPT2LMHeadModel.from_pretrained(self.base).state_dict()
        with patch.object(HFPolicy, "generate", self.scripted()):
            result = train(self.cfg)
        self.assertEqual(result["step"], 1)
        after = GPT2LMHeadModel.from_pretrained(result["checkpoint"]).state_dict()
        self.assertTrue(any(not torch.equal(before[key], after[key]) for key in before))
        self.cfg.train.max_steps = 2
        self.cfg.train.max_groups = 2
        self.cfg.train.output_dir = str(self.root / "resumed")
        with patch.object(HFPolicy, "generate", self.scripted()):
            resumed = train(self.cfg, result["checkpoint"])
        self.assertEqual(resumed["step"], 2)
        uninterrupted = deepcopy(self.cfg)
        uninterrupted.train.output_dir = str(self.root / "uninterrupted")
        with patch.object(HFPolicy, "generate", self.scripted()):
            full = train(uninterrupted)
        resumed_state = GPT2LMHeadModel.from_pretrained(resumed["checkpoint"]).state_dict()
        full_state = GPT2LMHeadModel.from_pretrained(full["checkpoint"]).state_dict()
        for key in full_state:
            self.assertTrue(torch.equal(full_state[key], resumed_state[key]), key)
        with patch.object(HFPolicy, "generate", self.scripted()):
            metrics = evaluate(self.cfg, self.root / "evaluation", resumed["checkpoint"])
        self.assertEqual(metrics["execution_accuracy"], 1)
        self.assertTrue((self.root / "evaluation" / "summary.csv").is_file())

    def test_equal_groups_skip_and_save(self):
        def invalid(policy, messages, sample=True):
            return Turn(policy.encode(messages), [7], "SELECT")
        with patch.object(HFPolicy, "generate", invalid):
            result = train(self.cfg)
        self.assertEqual(result["step"], 0)
        self.assertEqual(result["equal_groups"], 1)
        self.assertFalse(result["max_steps_reached"])

    def test_validation_best_resume_and_rng_isolation(self):
        self.cfg.evaluation.enabled = True
        self.cfg.evaluation.every_steps = 1
        # Validation intentionally consumes RNG; training must still match a run without validation.
        def script():
            training = self.scripted()
            def generate(policy, messages, sample=True):
                if sample:
                    return training(policy, messages, sample)
                __import__("random").random()
                torch.rand(3)
                text = "<answer> SELECT 1 </answer>"
                return Turn(policy.encode(messages), policy.tokenizer.encode(text, add_special_tokens=False), text)
            return generate
        with patch.object(HFPolicy, "generate", script()):
            result = train(self.cfg)
        metrics = [json.loads(line) for line in (self.root / "run" / "dev_metrics.jsonl").read_text().splitlines()]
        self.assertEqual([r["step"] for r in metrics], [0, 1])
        self.assertEqual(result["validation"]["best_step"], 0)
        self.assertTrue((Path(result["validation"]["best_checkpoint"]) / "complete.json").is_file())
        without = deepcopy(self.cfg)
        without.evaluation.enabled = False
        without.train.output_dir = str(self.root / "without-eval")
        with patch.object(HFPolicy, "generate", script()):
            control = train(without)
        actual = GPT2LMHeadModel.from_pretrained(result["checkpoint"]).state_dict()
        expected = GPT2LMHeadModel.from_pretrained(control["checkpoint"]).state_dict()
        self.assertTrue(all(torch.equal(actual[k], expected[k]) for k in actual))
        saved = torch.load(Path(result["checkpoint"]) / "training_state.pt", weights_only=False)
        control_saved = torch.load(Path(control["checkpoint"]) / "training_state.pt", weights_only=False)
        self.assertTrue(torch.equal(saved["torch_rng"], control_saved["torch_rng"]))
        self.assertEqual(saved["python_rng"], control_saved["python_rng"])
        self.cfg.train.max_steps = self.cfg.train.max_groups = 2
        self.cfg.train.output_dir = str(self.root / "validation-resume")
        with patch.object(HFPolicy, "generate", script()):
            resumed = train(self.cfg, result["checkpoint"])
        self.assertEqual(resumed["validation"]["best_step"], 0)
        self.assertTrue((self.root / "validation-resume" / "best_checkpoint.json").is_file())
        metrics = [json.loads(line) for line in (self.root / "validation-resume" / "dev_metrics.jsonl").read_text().splitlines()]
        self.assertEqual([r["step"] for r in metrics], [2])

    def test_pool_greedy_evaluation_matches_local(self):
        from rrcm_sql.runtime.rollout_pool import RolloutPool
        from rrcm_sql.runtime.evaluation_pool import EvaluationPool
        from rrcm_sql.evaluate import evaluate_policy
        from rrcm_sql.data import split_config
        model, tokenizer = load_model(self.cfg.model)
        policy = HFPolicy(model, tokenizer, self.cfg.model, self.cfg.rollout)
        cfg = split_config(self.cfg, "dev")
        local = evaluate_policy(cfg, self.root / "local-eval", policy=policy)
        pool = RolloutPool(self.cfg, ["cpu"])
        try:
            pool.sync(model, 0)
            parallel = evaluate_policy(cfg, self.root / "pool-eval", pool=pool)
        finally:
            pool.close()
        for key in ("execution_accuracy", "reward_mean", "output_tokens_mean", "data_sha256"):
            self.assertEqual(local[key], parallel[key])
        record = json.loads((self.root / "pool-eval" / "trajectories.jsonl").read_text().splitlines()[0])
        self.assertEqual(record["mode"], "free")
        standalone_pool = EvaluationPool(cfg, ["cpu"], policy_version=0)
        try:
            standalone = evaluate_policy(cfg, self.root / "standalone-pool-eval",
                                         pool=standalone_pool)
        finally:
            standalone_pool.close()
        for key in ("execution_accuracy", "reward_mean", "output_tokens_mean", "data_sha256"):
            self.assertEqual(local[key], standalone[key])
        self.assertEqual(standalone["evaluation_devices"], ["cpu"])

    def test_official_training_and_test_evaluation(self):
        rows = json.loads((self.root / "source.json").read_text())
        self.cfg.data.split_mode = "official"
        for split, row in (("train", rows[0]), ("dev", rows[1]), ("test", dict(rows[0], db_id="c"))):
            path = self.root / f"{split}.json"
            path.write_text(json.dumps([row]))
            setattr(self.cfg.data, f"{split}_json", str(path))
        database = self.root / "test-db" / "c"
        database.mkdir(parents=True)
        with sqlite3.connect(database / "c.sqlite") as conn:
            conn.execute("CREATE TABLE t(id INTEGER)")
        schema = json.loads((self.root / "tables.json").read_text())[0]
        path = self.root / "test-tables.json"
        path.write_text(json.dumps([dict(schema, db_id="c")]))
        self.cfg.data.test_tables = str(path)
        self.cfg.data.test_database_dir = str(self.root / "test-db")
        self.cfg.evaluation.enabled = True
        self.cfg.evaluation.every_steps = 1
        scripted = self.scripted()
        calls = [0]
        def generate(policy, messages, sample=True):
            if sample:
                return scripted(policy, messages, sample)
            calls[0] += 1
            sql = "2" if calls[0] == 1 else "1"
            text = f"<answer> SELECT {sql} </answer>"
            return Turn(policy.encode(messages), policy.tokenizer.encode(text, add_special_tokens=False), text)
        with patch.object(HFPolicy, "generate", generate):
            result = train(self.cfg)
            self.assertEqual(result["validation"]["best_step"], 1)
            metrics = evaluate(self.cfg, self.root / "test-result",
                               result["validation"]["best_checkpoint"], split="test")
        self.assertEqual(metrics["split"], "test")
        self.assertEqual(metrics["execution_accuracy"], 1)
        self.assertEqual(metrics["policy_version"], 1)
        prediction = json.loads((self.root / "test-result" / "trajectories.jsonl").read_text().splitlines()[0])
        self.assertEqual(prediction["policy_version"], 1)
        saved_cfg = json.loads((self.root / "test-result" / "config.json").read_text())
        self.assertEqual(saved_cfg["data"]["tables"], str(path))
        self.assertTrue((self.root / "run" / "data_manifest.json").is_file())
        # Changing dev contents at the same path must invalidate resume selection history.
        (self.root / "dev.json").write_text(json.dumps([dict(rows[1], query="SELECT 2")]))
        self.cfg.train.output_dir = str(self.root / "changed-dev")
        with self.assertRaisesRegex(ValueError, "Validation data/settings changed"):
            train(self.cfg, result["checkpoint"])

    def test_multiturn_grpo_and_official_evaluation(self):
        count = [0]
        def generate(policy, messages, sample=True):
            count[0] += 1
            texts = ["<intermediate> SELECT 1 </intermediate>",
                     "<answer> SELECT 1 </answer>", "<answer> SELECT 2 </answer>"]
            text = texts[(count[0] - 1) % 3]
            return Turn(policy.encode(messages), policy.tokenizer.encode(text, add_special_tokens=False), text)
        with patch.object(HFPolicy, "generate", generate):
            result = train(self.cfg)
        self.assertEqual(result["step"], 1)
        records = [json.loads(line) for line in (self.root / "run" / "trajectories.jsonl").read_text().splitlines()]
        self.assertEqual(len(records[0]["turns"]), 2)
        self.assertEqual(records[0]["intermediate_count"], 1)
        self.assertEqual(records[0]["outcome"], "correct")

    def test_mixed_group_is_three_free_and_three_prompted(self):
        free_count = [0]
        def obey(policy, messages, sample=True):
            instruction = messages[-1]["content"]
            if "one <intermediate>" in instruction:
                text = "<intermediate> SELECT 1 </intermediate>"
            elif "one <answer>" in instruction:
                text = "<answer> SELECT 1 </answer>"
            else:
                free_count[0] += 1
                sql = "1" if free_count[0] % 2 else "2"
                text = f"<answer> SELECT {sql} </answer>"
            return Turn(policy.encode(messages), policy.tokenizer.encode(text, add_special_tokens=False), text)
        self.cfg.rollout.group_size = 6
        self.cfg.rollout.prompted_trajectories = 3
        with patch.object(HFPolicy, "generate", obey):
            result = train(self.cfg)
        self.assertEqual(result["step"], 1)
        records = [json.loads(line) for line in
                   (self.root / "run" / "trajectories.jsonl").read_text().splitlines()]
        self.assertEqual([row["mode"] for row in records],
                         ["free"] * 3 + ["prompted_random"] * 3)
        self.assertTrue(all(action["requested"] is not None
                            for row in records[3:] for action in row["action_trace"]))

    def test_official_spider_integration(self):
        repository = Path(__file__).resolve().parents[2]
        evaluator = repository / "overall_pipeline/vendor/spider_test_suite_eval"
        data = repository / "data/spider_data"
        if not evaluator.is_dir() or not data.is_dir():
            self.skipTest("Optional repository Spider fixtures are unavailable")
        from rrcm_sql.sql import Executor, Judge
        from rrcm_sql.spider_metrics import run
        self.cfg.sql.evaluator_path = str(evaluator)
        self.cfg.sql.nltk_data = str(repository / "data/nltk_data")
        rows = json.loads((data / "train_spider.json").read_text())[:3]
        judge = Judge(Executor(self.cfg.sql), data / "database")
        for row in rows:
            self.assertEqual(judge.score(row, row["query"])["outcome"], "correct")
        result = run({"evaluator_path": str(evaluator), "nltk_data": self.cfg.sql.nltk_data,
                      "database_dir": str(data / "database"), "tables": str(data / "tables.json"),
                      "rows": [dict(row, prediction=row["query"]) for row in rows]})
        self.assertTrue(all(row["exact_match"] for row in result))
        # The generated tiny model can still run the full official evaluation CLI path.
        self.cfg.data.database_dir = str(data / "database")
        self.cfg.data.tables = str(data / "tables.json")
        selected = __import__("random").Random(self.cfg.train.seed).sample(
            json.loads((data / "train_spider.json").read_text()), 1)[0]
        def gold_policy(policy, messages, sample=True):
            text = f"<answer> {selected['query']} </answer>"
            return Turn(policy.encode(messages), [7], text)
        with patch.object(HFPolicy, "generate", gold_policy):
            metrics = evaluate(self.cfg, self.root / "official-eval", data_file=str(data / "train_spider.json"), limit=1)
        self.assertEqual(metrics["exact_match"], 1)
        self.assertTrue((self.root / "official-eval" / "difficulty.csv").is_file())

    def test_lora_sft_local_adapter_and_second_architecture(self):
        self.cfg.model.mode = "lora"
        result = sft(self.cfg)
        self.assertTrue((Path(result["checkpoint"]) / "adapter_config.json").is_file())
        cfg = replace(self.cfg.model, name_or_path=result["checkpoint"])
        model, tokenizer = load_model(cfg)
        self.assertTrue(any(p.requires_grad for p in model.parameters()))
        self.assertLess(sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters()))
        # Continue RL from a locally saved SFT adapter, including its reference policy.
        self.cfg.model = cfg
        self.cfg.train.output_dir = str(self.root / "adapter-rl")
        with patch.object(HFPolicy, "generate", self.scripted()):
            trained = train(self.cfg)
        self.assertEqual(trained["step"], 1)
        llama = self.root / "llama"
        LlamaForCausalLM(LlamaConfig(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                    max_position_embeddings=1024)).save_pretrained(llama)
        tokenizer.save_pretrained(llama)
        loaded, _ = load_model(replace(cfg, name_or_path=str(llama)))
        self.assertEqual(loaded.config.model_type, "llama")

    def test_special_action_tokens_preserved(self):
        _, tokenizer = load_model(self.cfg.model)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<answer>", "</answer>"]})
        ids = tokenizer.encode("<answer> SELECT 1 </answer>", add_special_tokens=False)
        self.assertIn("<answer>", decode_action(tokenizer, ids + [tokenizer.eos_token_id]))

    def test_cpu_rollout_worker_version_and_old_log_probs(self):
        from rrcm_sql.runtime.rollout_pool import RolloutPool
        from rrcm_sql.data import schema_map
        self.cfg.model.mode = "lora"
        model, _ = load_model(self.cfg.model)
        pool = RolloutPool(self.cfg, ["cpu"])
        try:
            pool.sync(model, policy_version=7)
            example = json.loads((self.root / "prepared" / "train.json").read_text())[0]
            schema = schema_map(self.cfg.data.tables)[example["db_id"]]
            group = pool.generate(example, schema, group_id=3, policy_version=7)
        finally:
            pool.close()
        self.assertEqual(len(group), self.cfg.rollout.group_size)
        self.assertTrue(all(t.policy_version == 7 for t in group))
        self.assertTrue(all(turn.old_log_probs is not None and
                            len(turn.old_log_probs) == len(turn.action_ids)
                            for trajectory in group for turn in trajectory.turns))

    @unittest.skipUnless(os.environ.get("RRCM_RUN_CUDA_FSDP_SMOKE") == "1",
                         "Set RRCM_RUN_CUDA_FSDP_SMOKE=1 with two visible idle GPUs")
    def test_cuda_fsdp_launch_and_export(self):
        self.cfg.evaluation.enabled = True
        self.cfg.evaluation.every_steps = 1
        self.cfg.model.device = "cuda:0"
        self.cfg.model.mode = "lora"
        self.cfg.model.gradient_checkpointing = False
        self.cfg.train.kl_coefficient = 0
        self.cfg.runtime.update_backend = "fsdp"
        self.cfg.runtime.update_devices = ["cuda:0", "cuda:1"]
        self.cfg.runtime.rollout_devices = ["cpu"]
        self.cfg.validate()
        with patch("rrcm_sql.runtime.fsdp._rank_main", _scripted_fsdp_rank):
            result = train(self.cfg)
        self.assertEqual(result["step"], 1)
        self.assertEqual(result["update_backend"], "fsdp")
        self.assertTrue((Path(result["checkpoint"]) / "complete.json").is_file())
        self.assertTrue((Path(result["validation"]["best_checkpoint"]) / "complete.json").is_file())
        self.assertTrue((Path(self.cfg.train.output_dir) / "dev_metrics.jsonl").is_file())
        dev = [json.loads(line) for line in (Path(self.cfg.train.output_dir) / "dev_metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["step"] for row in dev], [0, 1])
        self.cfg.train.max_steps = self.cfg.train.max_groups = 2
        self.cfg.train.output_dir = str(self.root / "fsdp-resumed")
        with patch("rrcm_sql.runtime.fsdp._rank_main", _scripted_fsdp_rank):
            resumed = train(self.cfg, result["checkpoint"])
        self.assertEqual(resumed["step"], 2)
        self.assertEqual(resumed["validation"]["best_step"], 0)
        import multiprocessing
        import socket
        import torch.multiprocessing as torch_mp
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        queue = multiprocessing.get_context("spawn").Queue()
        torch_mp.spawn(_fsdp_update_smoke, args=(port, queue), nprocs=2, join=True)
        self.assertEqual(queue.get(), (True, True))


if __name__ == "__main__":
    unittest.main()
