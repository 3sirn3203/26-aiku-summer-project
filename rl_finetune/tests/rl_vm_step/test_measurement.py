from __future__ import annotations

import unittest

from text2sql.core.models import ExecutionResult
from rl_vm_step.measurement import VMStepMeasurementError, measurement_from_execution


def _execution(**overrides):
    values = {
        "status": "success",
        "vm_steps_lower_bound": 2_000,
        "vm_steps_upper_bound_exclusive": 3_000,
        "vm_step_progress_interval": 1_000,
        "vm_step_measurement_complete": True,
    }
    values.update(overrides)
    return ExecutionResult(**values)


class VMStepMeasurementTests(unittest.TestCase):
    def test_uses_interval_midpoint(self) -> None:
        measurement = measurement_from_execution(_execution())
        self.assertEqual(measurement.estimate, 2_500.0)
        self.assertEqual(measurement.interval, 1_000)

    def test_accepts_zero_lower_bound(self) -> None:
        measurement = measurement_from_execution(
            _execution(
                vm_steps_lower_bound=0,
                vm_steps_upper_bound_exclusive=1_000,
            )
        )
        self.assertEqual(measurement.estimate, 500.0)

    def test_rejects_incomplete_measurement(self) -> None:
        with self.assertRaises(VMStepMeasurementError):
            measurement_from_execution(
                _execution(
                    vm_step_measurement_complete=False,
                    vm_steps_upper_bound_exclusive=None,
                )
            )

    def test_rejects_inconsistent_interval(self) -> None:
        with self.assertRaises(VMStepMeasurementError):
            measurement_from_execution(
                _execution(vm_steps_upper_bound_exclusive=2_500)
            )


if __name__ == "__main__":
    unittest.main()
