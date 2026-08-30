from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from text2sql.core.doctor import server_doctor
from text2sql.core.model_source import inspect_local_checkpoint
from text2sql.multi_turn_agent.config import AgentAppConfig, load_agent_config


def inspect_role_worker(config_path: Path, role: str, physical_gpu: int) -> Dict[str, Any]:
    config = load_agent_config(config_path)
    if role not in config.roles:
        raise ValueError("unknown agent role: %s" % role)
    role_config = config.roles[role]
    report = server_doctor("cuda:0")
    minimum = role_config.minimum_free_vram_bytes
    free = report.get("gpu_memory_free_bytes")
    errors = list(report.get("errors", []))
    if isinstance(free, int) and free < minimum:
        errors.append(
            "%s requires at least %d free GPU bytes; found %d"
            % (role, minimum, free)
        )
    report.update(
        {
            "role": role,
            "physical_gpu": physical_gpu,
            "logical_device": "cuda:0",
            "configured_model": role_config.model.model_id,
            "configured_model_source": role_config.model.source,
            "configured_revision": role_config.model.revision,
            "trainable": role_config.trainable,
            "role_minimum_free_vram_bytes": minimum,
            "errors": errors,
            "ok": not errors,
        }
    )
    return report


def run_agent_doctor(
    config: AgentAppConfig,
    gpu_pools: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    reports = []
    local_reports: Dict[str, Dict[str, Any]] = {}
    local_cache: Dict[str, Dict[str, Any]] = {}
    for role in ("planner", "coder", "verifier"):
        model = config.roles[role].model
        if model.source != "local":
            continue
        if model.model_id not in local_cache:
            local_cache[model.model_id] = inspect_local_checkpoint(
                Path(model.model_id)
            ).to_dict()
        local_reports[role] = local_cache[model.model_id]
    for role in ("planner", "coder", "verifier"):
        for physical_gpu in gpu_pools[role]:
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            command = [
                sys.executable,
                "-m",
                "text2sql",
                "_agent-doctor-worker",
                "--config",
                str(config.source_path),
                "--role",
                role,
                "--physical-gpu",
                str(physical_gpu),
            ]
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                check=False,
            )
            if completed.returncode not in (0, 1):
                reports.append(
                    {
                        "role": role,
                        "physical_gpu": physical_gpu,
                        "ok": False,
                        "errors": [
                            "doctor worker exited with code %d: %s"
                            % (completed.returncode, completed.stderr.strip()[:2000])
                        ],
                    }
                )
                continue
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                payload = {
                    "role": role,
                    "physical_gpu": physical_gpu,
                    "ok": False,
                    "errors": ["doctor worker returned invalid JSON: %s" % exc],
                }
            local_checkpoint = local_reports.get(role)
            if local_checkpoint is not None:
                payload["local_checkpoint"] = local_checkpoint
                if local_checkpoint.get("ok") is not True:
                    errors = list(payload.get("errors", []))
                    errors.extend(local_checkpoint.get("errors", []))
                    payload["errors"] = errors
                    payload["ok"] = False
            reports.append(payload)
    return {
        "ok": bool(reports) and all(report.get("ok") is True for report in reports),
        "worker_count": len(reports),
        "reports": reports,
    }
