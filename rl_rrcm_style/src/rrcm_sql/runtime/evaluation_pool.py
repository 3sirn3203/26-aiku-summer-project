"""Standalone evaluation workers with one model replica per device."""
from copy import deepcopy
import atexit
import multiprocessing as mp
import queue
import random
import time
import traceback

import torch

from ..model import HFPolicy, load_model
from ..rollout import rollout
from ..sql import Executor, Judge


def validate_devices(devices):
    devices = list(devices)
    if not devices:
        raise ValueError("Evaluation needs at least one device")
    if len(devices) != len(set(devices)):
        raise ValueError("Evaluation devices must be unique")
    if len(devices) > 1 and "cuda" in devices:
        raise ValueError("Use explicit cuda:N names when evaluating on multiple GPUs")
    for device in devices:
        if device == "cpu":
            continue
        if not device.startswith("cuda:") or not device[5:].isdigit():
            raise ValueError(f"Invalid evaluation device: {device}")
        index = int(device[5:])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(f"Evaluation device is unavailable: {device}")
    return devices


def _worker(device, cfg, checkpoint, policy_version, tasks, results):
    try:
        cfg = deepcopy(cfg)
        cfg.model.device = device
        random.seed(cfg.train.seed)
        torch.manual_seed(cfg.train.seed)
        if device.startswith("cuda"):
            torch.cuda.set_device(device)
            torch.cuda.manual_seed(cfg.train.seed)
        model, tokenizer = load_model(cfg.model, trainable=False, checkpoint=checkpoint)
        policy = HFPolicy(model, tokenizer, cfg.model, cfg.rollout)
        executor = Executor(cfg.sql)
        judge = Judge(executor, cfg.data.database_dir)
        results.put(("ready", device, None))
        while True:
            message = tasks.get()
            if message[0] == "close":
                return
            _, index, example, schema = message
            trajectory = rollout(
                policy, example, schema, executor, judge, cfg.rollout,
                mode="free", sample=False,
                evaluate_suite=bool(cfg.sql.suite_database_dir),
                policy_version=policy_version,
            )
            results.put(("evaluation", index, trajectory))
    except BaseException:
        results.put(("error", device, traceback.format_exc()))


class EvaluationPool:
    def __init__(self, cfg, devices, checkpoint=None, policy_version=0):
        self.cfg = cfg
        self.devices = validate_devices(devices)
        self.closed = False
        self.context = mp.get_context("spawn")
        self.tasks = self.context.Queue()
        self.results = self.context.Queue()
        self.processes = []
        for device in self.devices:
            process = self.context.Process(
                target=_worker,
                args=(device, cfg, checkpoint, policy_version, self.tasks, self.results),
            )
            process.start()
            self.processes.append(process)
        try:
            self._wait_until_ready()
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def _get(self, timeout):
        try:
            message = self.results.get(timeout=timeout)
        except queue.Empty as exc:
            failed = [p.exitcode for p in self.processes if p.exitcode not in (None, 0)]
            if failed:
                raise RuntimeError(f"Evaluation worker exited unexpectedly: {failed}") from exc
            return None
        if message[0] == "error":
            raise RuntimeError(f"Evaluation worker {message[1]} failed:\n{message[2]}")
        return message

    def _wait_until_ready(self):
        ready = set()
        deadline = time.monotonic() + self.cfg.runtime.worker_timeout
        while len(ready) < len(self.processes):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out starting evaluation workers")
            message = self._get(min(remaining, 1.0))
            if message is None:
                continue
            if message[0] != "ready":
                raise RuntimeError(f"Unexpected evaluation worker message: {message[0]}")
            ready.add(message[1])

    def evaluate(self, rows, schemas, policy_version, cfg):
        del policy_version  # Workers were loaded at this immutable checkpoint version.
        for index, row in enumerate(rows):
            self.tasks.put(("evaluate", index, row, schemas[row["db_id"]]))
        trajectories = {}
        deadline = time.monotonic() + cfg.evaluation.timeout_seconds
        while len(trajectories) < len(rows):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Evaluation exceeded timeout_seconds")
            message = self._get(min(remaining, 1.0))
            if message is None:
                continue
            if message[0] != "evaluation":
                raise RuntimeError(f"Unexpected evaluation worker message: {message[0]}")
            index, trajectory = message[1], message[2]
            if index in trajectories or not 0 <= index < len(rows):
                raise RuntimeError(f"Invalid or duplicate evaluation result index: {index}")
            trajectories[index] = trajectory
            print(f"{len(trajectories)}/{len(rows)} completed", flush=True)
        return [trajectories[index] for index in range(len(rows))]

    def close(self):
        if self.closed:
            return
        self.closed = True
        for _ in self.processes:
            self.tasks.put(("close",))
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
