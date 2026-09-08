from copy import deepcopy
import multiprocessing as mp
import queue
import random
import time
import traceback

import torch

from rrcm_sql.runtime.evaluation_pool import validate_devices
from rrcm_sql.sql import Executor

from .generation import generate
from .model import load_zero_shot_policy


def _worker(device, cfg, tasks, results):
    try:
        random.seed(cfg.evaluation.seed)
        torch.manual_seed(cfg.evaluation.seed)
        if device.startswith("cuda"):
            torch.cuda.set_device(device)
            torch.cuda.manual_seed(cfg.evaluation.seed)
        policy = load_zero_shot_policy(cfg, device)
        executor = Executor(cfg.sql)
        rollout_cfg = cfg.generation.rollout_config()
        results.put(("ready", device, None))
        while True:
            message = tasks.get()
            if message[0] == "close":
                return
            _, index, example, schema = message
            record = generate(policy, example, schema, executor,
                              cfg.data.test_database_dir, rollout_cfg)
            results.put(("result", index, record))
    except BaseException:
        results.put(("error", device, traceback.format_exc()))


class GenerationPool:
    def __init__(self, cfg, devices):
        self.cfg = deepcopy(cfg)
        self.devices = validate_devices(devices)
        self.context = mp.get_context("spawn")
        self.tasks = self.context.Queue()
        self.results = self.context.Queue()
        self.processes = []
        self.closed = False
        for device in self.devices:
            process = self.context.Process(
                target=_worker, args=(device, self.cfg, self.tasks, self.results))
            process.start()
            self.processes.append(process)
        try:
            self._wait_ready()
        except BaseException:
            self.close()
            raise

    def _get(self, timeout):
        try:
            message = self.results.get(timeout=timeout)
        except queue.Empty as exc:
            failed = [p.exitcode for p in self.processes if p.exitcode not in (None, 0)]
            if failed:
                raise RuntimeError(f"Generation worker exited unexpectedly: {failed}") from exc
            return None
        if message[0] == "error":
            raise RuntimeError(f"Generation worker {message[1]} failed:\n{message[2]}")
        return message

    def _wait_ready(self):
        ready = set()
        deadline = time.monotonic() + self.cfg.evaluation.worker_timeout
        while len(ready) < len(self.processes):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out starting generation workers")
            message = self._get(min(remaining, 1.0))
            if message is None:
                continue
            if message[0] != "ready":
                raise RuntimeError(f"Unexpected worker message: {message[0]}")
            ready.add(message[1])

    def run(self, jobs, on_result):
        for index, example, schema in jobs:
            self.tasks.put(("generate", index, example, schema))
        pending = {index for index, _, _ in jobs}
        deadline = time.monotonic() + self.cfg.evaluation.timeout_seconds
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Generation exceeded timeout_seconds")
            message = self._get(min(remaining, 1.0))
            if message is None:
                continue
            if message[0] != "result" or message[1] not in pending:
                raise RuntimeError("Unexpected or duplicate generation result")
            pending.remove(message[1])
            on_result(message[1], message[2])

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

