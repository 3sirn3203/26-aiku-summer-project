from __future__ import annotations

import platform
import sqlite3
import sys
from typing import Any, Dict, List


EXPECTED_TORCH_VERSION = "2.5.1"
EXPECTED_TORCH_CUDA_VERSION = "12.1"
EXPECTED_TRANSFORMERS_VERSION = "4.46.3"
MINIMUM_FREE_VRAM_BYTES = 4 * 1024 * 1024 * 1024


def server_doctor(device_name: str = "cuda:0") -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    report: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "sqlite_version": sqlite3.sqlite_version,
        "configured_device": device_name,
        "expected_torch_version": EXPECTED_TORCH_VERSION,
        "expected_torch_cuda_version": EXPECTED_TORCH_CUDA_VERSION,
        "expected_transformers_version": EXPECTED_TRANSFORMERS_VERSION,
        "minimum_free_vram_bytes": MINIMUM_FREE_VRAM_BYTES,
    }
    report["recommended_python_version_met"] = sys.version_info >= (3, 11)
    report["sqlite_engine_length_limit_available"] = bool(
        hasattr(sqlite3.Connection, "setlimit")
        and hasattr(sqlite3, "SQLITE_LIMIT_LENGTH")
    )
    report["linux_server"] = platform.system() == "Linux"
    if not report["recommended_python_version_met"]:
        warnings.append("Python 3.11 or newer is recommended for the server smoke test")
    if not report["sqlite_engine_length_limit_available"]:
        warnings.append("Python sqlite3 does not expose the engine-level length limit")
    if not report["linux_server"]:
        warnings.append("The actual GPU smoke test is contracted for a Linux server")
    try:
        import resource

        report["linux_worker_memory_limit_available"] = bool(
            report["linux_server"] and hasattr(resource, "RLIMIT_AS")
        )
    except ImportError:
        report["linux_worker_memory_limit_available"] = False
    if report["linux_server"] and not report["linux_worker_memory_limit_available"]:
        warnings.append("Linux RLIMIT_AS is unavailable for the SQL worker")

    try:
        import torch

        report["torch_version"] = torch.__version__
        report["torch_cuda_version"] = torch.version.cuda
        if torch.__version__.split("+", 1)[0] != EXPECTED_TORCH_VERSION:
            errors.append(
                "Expected torch version %s; found %s"
                % (EXPECTED_TORCH_VERSION, torch.__version__)
            )
        if torch.version.cuda != EXPECTED_TORCH_CUDA_VERSION:
            errors.append(
                "Expected torch CUDA runtime %s; found %s"
                % (EXPECTED_TORCH_CUDA_VERSION, torch.version.cuda)
            )
        report["cuda_available"] = torch.cuda.is_available()
        if not torch.cuda.is_available():
            errors.append("torch.cuda.is_available() is false")
        elif not device_name.startswith("cuda"):
            errors.append("Configured device must be a CUDA device")
        else:
            try:
                device = torch.device(device_name)
                device_index = (
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                )
                capability = tuple(torch.cuda.get_device_capability(device_index))
                architectures = list(torch.cuda.get_arch_list())
                properties = torch.cuda.get_device_properties(device_index)
                free_memory, total_memory = torch.cuda.mem_get_info(device_index)
                report.update(
                    {
                        "gpu_name": torch.cuda.get_device_name(device_index),
                        "gpu_capability": list(capability),
                        "gpu_memory_total_bytes": int(total_memory),
                        "gpu_memory_free_bytes": int(free_memory),
                        "torch_arch_list": architectures,
                    }
                )
                if capability != (6, 1):
                    errors.append("Expected TITAN Xp compute capability 6.1")
                if int(free_memory) < MINIMUM_FREE_VRAM_BYTES:
                    errors.append(
                        "At least %d free GPU bytes are required; found %d"
                        % (MINIMUM_FREE_VRAM_BYTES, free_memory)
                    )
                try:
                    probe = torch.ones((2, 2), dtype=torch.float32, device=device)
                    probe_result = probe.matmul(probe)
                    torch.cuda.synchronize(device)
                    if float(probe_result[0, 0].item()) != 2.0:
                        raise RuntimeError("unexpected result")
                    report["cuda_fp32_probe"] = "passed"
                    del probe_result, probe
                except Exception as exc:
                    report["cuda_fp32_probe"] = "failed"
                    errors.append(
                        "FP32 CUDA probe failed: %s: %s"
                        % (type(exc).__name__, exc)
                    )
                if architectures and "sm_61" not in architectures:
                    warnings.append(
                        "sm_61 is not listed explicitly; the FP32 CUDA probe is the "
                        "authoritative compatibility check"
                    )
            except Exception as exc:
                errors.append(
                    "Could not inspect configured CUDA device: %s: %s"
                    % (type(exc).__name__, exc)
                )
    except ImportError:
        errors.append("torch is not installed")
    except Exception as exc:
        errors.append("torch could not be imported: %s: %s" % (type(exc).__name__, exc))

    try:
        import transformers

        report["transformers_version"] = transformers.__version__
        if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
            errors.append(
                "Expected transformers %s; found %s"
                % (EXPECTED_TRANSFORMERS_VERSION, transformers.__version__)
            )
    except ImportError:
        errors.append("transformers is not installed")
    except Exception as exc:
        errors.append(
            "transformers could not be imported: %s: %s" % (type(exc).__name__, exc)
        )

    report["errors"] = errors
    report["warnings"] = warnings
    report["ok"] = not errors
    report["model_loaded"] = False
    return report
