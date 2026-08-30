from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from text2sql.core.doctor import MINIMUM_FREE_VRAM_BYTES, server_doctor


class _Scalar:
    def item(self):
        return 2.0


class _Probe:
    def matmul(self, _other):
        return self

    def __getitem__(self, _key):
        return _Scalar()


def _modules(transformers_version="4.46.3"):
    cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        get_device_capability=lambda _index: (6, 1),
        get_arch_list=lambda: ["sm_60"],
        get_device_properties=lambda _index: SimpleNamespace(total_memory=12 * 1024**3),
        mem_get_info=lambda _index: (10 * 1024**3, 12 * 1024**3),
        get_device_name=lambda _index: "NVIDIA TITAN Xp",
        synchronize=lambda _device: None,
    )
    torch = types.ModuleType("torch")
    torch.__version__ = "2.5.1+cu121"
    torch.version = SimpleNamespace(cuda="12.1")
    torch.cuda = cuda
    torch.float32 = "float32"
    torch.device = lambda value: SimpleNamespace(index=0, value=value)
    torch.ones = lambda *_args, **_kwargs: _Probe()

    transformers = types.ModuleType("transformers")
    transformers.__version__ = transformers_version
    return torch, transformers


class DoctorTests(unittest.TestCase):
    def test_actual_cuda_probe_is_authoritative(self):
        torch, transformers = _modules()
        with mock.patch.dict(
            sys.modules,
            {"torch": torch, "transformers": transformers},
        ):
            report = server_doctor("cuda:0")

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["cuda_fp32_probe"], "passed")
        self.assertEqual(report["gpu_capability"], [6, 1])
        self.assertEqual(MINIMUM_FREE_VRAM_BYTES, 4 * 1024**3)
        self.assertTrue(any("sm_61" in item for item in report["warnings"]))

    def test_wrong_transformers_version_is_an_error(self):
        torch, transformers = _modules(transformers_version="4.47.0")
        with mock.patch.dict(
            sys.modules,
            {"torch": torch, "transformers": transformers},
        ):
            report = server_doctor("cuda:0")

        self.assertFalse(report["ok"])
        self.assertTrue(any("4.46.3" in item for item in report["errors"]))


if __name__ == "__main__":
    unittest.main()
