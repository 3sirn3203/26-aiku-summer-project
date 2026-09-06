from __future__ import annotations

from typing import Any, Mapping

from rl_vm_step.models import VMStepMeasurement


class VMStepMeasurementError(ValueError):
    """Raised when a completed execution lacks a valid VM-step interval."""


def _field(execution: Any, name: str) -> Any:
    if isinstance(execution, Mapping):
        return execution.get(name)
    return getattr(execution, name, None)


def measurement_from_execution(execution: Any) -> VMStepMeasurement:
    status = _field(execution, "status")
    complete = _field(execution, "vm_step_measurement_complete")
    lower = _field(execution, "vm_steps_lower_bound")
    upper = _field(execution, "vm_steps_upper_bound_exclusive")
    interval = _field(execution, "vm_step_progress_interval")

    if status != "success":
        raise VMStepMeasurementError("VM-step cost requires a successful execution")
    if complete is not True:
        raise VMStepMeasurementError("VM-step measurement is incomplete")
    values = {"lower bound": lower, "upper bound": upper, "interval": interval}
    for label, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise VMStepMeasurementError("VM-step %s is not an integer" % label)
    assert isinstance(lower, int) and isinstance(upper, int) and isinstance(interval, int)
    if lower < 0:
        raise VMStepMeasurementError("VM-step lower bound must be non-negative")
    if interval < 1:
        raise VMStepMeasurementError("VM-step interval must be positive")
    if upper <= lower:
        raise VMStepMeasurementError("VM-step upper bound must exceed lower bound")
    if upper - lower != interval:
        raise VMStepMeasurementError(
            "VM-step interval does not match the reported bounds"
        )
    return VMStepMeasurement(
        lower_bound=lower,
        upper_bound_exclusive=upper,
        interval=interval,
        estimate=(lower + upper) / 2.0,
    )
