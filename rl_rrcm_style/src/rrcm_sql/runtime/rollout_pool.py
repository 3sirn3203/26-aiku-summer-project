"""Persistent one-model-per-device Hugging Face rollout workers."""
from copy import deepcopy
import multiprocessing as mp
import queue
import random
import traceback
from pathlib import Path
import atexit
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from ..exploration import group_modes, task_seed
from ..model import HFPolicy, action_log_probs, load_model
from ..rollout import rollout
from ..sql import Executor, Judge


def rollout_state(model, mode):
    """Return the state that replicas need after a policy update."""
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = model.state_dict()
    if mode == "lora":
        state = {name: value for name, value in state.items()
                 if name in trainable or ".lora_" in name or ".modules_to_save." in name}
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _worker(device, cfg, tasks, results):
    try:
        cfg = deepcopy(cfg)
        cfg.model.device = device
        random.seed(cfg.train.seed)
        torch.manual_seed(cfg.train.seed)
        if str(device).startswith("cuda"):
            torch.cuda.set_device(device)
        model, tokenizer = load_model(cfg.model, trainable=False)
        policy = HFPolicy(model, tokenizer, cfg.model, cfg.rollout)
        executor = Executor(cfg.sql)
        judge = Judge(executor, cfg.data.database_dir, cfg.data.tables)
        evaluation_resources = {}
        version = None
        results.put(("ready", device, None))
        while True:
            message = tasks.get()
            if message[0] == "close":
                return
            if message[0] == "sync":
                _, version, state_path = message
                state = torch.load(state_path, map_location="cpu", weights_only=True)
                incompatible = model.load_state_dict(state, strict=False)
                if incompatible.unexpected_keys:
                    raise RuntimeError(f"Unexpected rollout state keys: {incompatible.unexpected_keys[:5]}")
                results.put(("synced", device, version))
                continue
            if message[0] == "evaluate":
                _, job_id, example, schema, expected_version, eval_cfg = message
                if version != expected_version:
                    raise RuntimeError("Evaluation worker policy version mismatch")
                key = (eval_cfg.data.database_dir, eval_cfg.data.tables,
                       eval_cfg.sql.suite_database_dir, eval_cfg.sql.reward_metric)
                if key not in evaluation_resources:
                    eval_executor = Executor(eval_cfg.sql)
                    evaluation_resources[key] = (
                        eval_executor,
                        Judge(eval_executor, eval_cfg.data.database_dir, eval_cfg.data.tables))
                eval_executor, eval_judge = evaluation_resources[key]
                from ..validation import preserve_inference_state
                with preserve_inference_state(model):
                    trajectory = rollout(
                        policy, example, schema, eval_executor, eval_judge, eval_cfg.rollout,
                        mode="free", sample=False, evaluate_suite=bool(eval_cfg.sql.suite_database_dir),
                        policy_version=version)
                results.put(("evaluation", job_id, trajectory))
                continue
            _, job_id, example, schema, mode, seed, expected_version = message
            if version != expected_version:
                raise RuntimeError(f"Worker policy version {version}, expected {expected_version}")
            torch.manual_seed(seed)
            if str(device).startswith("cuda"):
                torch.cuda.manual_seed(seed)
            trajectory = rollout(policy, example, schema, executor, judge, cfg.rollout,
                                 mode=mode, seed=seed, policy_version=version)
            for turn in trajectory.turns:
                with torch.no_grad():
                    turn.old_log_probs = action_log_probs(
                        model, turn, cfg.rollout.temperature, cfg.model).detach().cpu().tolist()
            results.put(("result", job_id, trajectory))
    except BaseException:
        results.put(("error", device, traceback.format_exc()))


class RolloutPool:
    def __init__(self, cfg, devices, state_directory=".rollout_state"):
        if not devices:
            raise ValueError("RolloutPool needs at least one device")
        self.cfg, self.devices = cfg, list(devices)
        self.closed = False
        self.context = mp.get_context("spawn")
        self.state_dir = Path(cfg.train.output_dir) / state_directory
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "policy.pt"
        self.results = self.context.Queue()
        self.tasks, self.processes = [], []
        for device in self.devices:
            tasks = self.context.Queue()
            process = self.context.Process(target=_worker, args=(device, cfg, tasks, self.results))
            process.start()
            self.tasks.append(tasks)
            self.processes.append(process)
        self._wait_for("ready", len(self.processes))
        atexit.register(self.close)

    def _get(self, timeout=None):
        try:
            message = self.results.get(timeout=timeout or self.cfg.runtime.worker_timeout)
        except queue.Empty as exc:
            raise TimeoutError("Timed out waiting for rollout worker") from exc
        if message[0] == "error":
            raise RuntimeError(f"Rollout worker {message[1]} failed:\n{message[2]}")
        return message

    def _wait_for(self, kind, count, expected_version=None):
        seen = 0
        while seen < count:
            message = self._get()
            if message[0] != kind:
                raise RuntimeError(f"Unexpected rollout worker message: {message[0]}")
            if expected_version is not None and message[2] != expected_version:
                raise RuntimeError("Rollout worker acknowledged the wrong policy version")
            seen += 1

    def sync(self, model, policy_version):
        state = rollout_state(model, self.cfg.model.mode)
        self.sync_state(state, policy_version)

    def sync_state(self, state, policy_version):
        temporary = self.state_dir / f"policy-{policy_version}.tmp"
        torch.save(state, temporary)
        temporary.replace(self.state_path)
        for tasks in self.tasks:
            tasks.put(("sync", policy_version, str(self.state_path)))
        self._wait_for("synced", len(self.tasks), policy_version)

    def generate(self, example, schema, group_id, policy_version):
        modes = group_modes(self.cfg.rollout)
        for index, mode in enumerate(modes):
            task = ("rollout", index, example, schema, mode,
                    task_seed(self.cfg.train.seed, group_id, index), policy_version)
            self.tasks[index % len(self.tasks)].put(task)
        trajectories = {}
        while len(trajectories) < len(modes):
            message = self._get()
            if message[0] != "result":
                raise RuntimeError(f"Unexpected rollout worker message: {message[0]}")
            trajectories[message[1]] = message[2]
        return [trajectories[index] for index in range(len(modes))]

    def close(self):
        if self.closed:
            return
        self.closed = True
        for tasks in self.tasks:
            tasks.put(("close",))
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()

    def evaluate(self, rows, schemas, policy_version, cfg):
        # Keep only one outstanding question per worker; reuse faster workers.
        pending, results, next_index = {}, {}, 0
        deadline = time.monotonic() + cfg.evaluation.timeout_seconds
        for worker, tasks in enumerate(self.tasks):
            if next_index >= len(rows):
                break
            row = rows[next_index]
            tasks.put(("evaluate", next_index, row, schemas[row["db_id"]], policy_version, cfg))
            pending[next_index] = worker
            next_index += 1
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Evaluation exceeded timeout_seconds")
            kind, index, trajectory = self._get(timeout=min(remaining, self.cfg.runtime.worker_timeout))
            if kind != "evaluation" or index not in pending:
                raise RuntimeError("Unexpected evaluation response")
            worker = pending.pop(index)
            results[index] = trajectory
            if next_index < len(rows):
                row = rows[next_index]
                self.tasks[worker].put(("evaluate", next_index, row, schemas[row["db_id"]], policy_version, cfg))
                pending[next_index] = worker
                next_index += 1
        return [results[i] for i in range(len(rows))]


class CombinedEvaluationPool:
    """Evaluate disjoint row shards concurrently across multiple rollout pools."""

    def __init__(self, pools):
        self.pools = list(pools)
        if not self.pools:
            raise ValueError("CombinedEvaluationPool needs at least one pool")
        self.devices = [device for pool in self.pools for device in pool.devices]

    def evaluate(self, rows, schemas, policy_version, cfg):
        assignments = [[] for _ in self.pools]
        weighted = [index for index, pool in enumerate(self.pools) for _ in pool.devices]
        for index, row in enumerate(rows):
            assignments[weighted[index % len(weighted)]].append((index, row))

        def evaluate_shard(pool, shard):
            selected = [row for _, row in shard]
            return shard, pool.evaluate(selected, schemas, policy_version, cfg)

        merged = {}
        with ThreadPoolExecutor(max_workers=len(self.pools)) as threads:
            futures = [threads.submit(evaluate_shard, pool, shard)
                       for pool, shard in zip(self.pools, assignments) if shard]
            for future in futures:
                shard, trajectories = future.result()
                for (index, _), trajectory in zip(shard, trajectories):
                    merged[index] = trajectory
        if len(merged) != len(rows):
            raise RuntimeError("Combined validation lost or duplicated examples")
        return [merged[index] for index in range(len(rows))]
