from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def unwrap_model_for_fsdp_generation(
    model: Any,
    accelerator: Any,
    *,
    fallback: Any,
) -> Iterator[Any]:
    """Gather FSDP parameters while TRL calls ``generate``.

    TRL 0.14 only gathers DeepSpeed ZeRO-3 parameters in its generation
    helper.  An unwrapped FSDP module otherwise exposes sharded embedding
    weights, which causes generation to fail before the first optimizer step.
    """
    from accelerate.utils import DistributedType

    if accelerator.distributed_type != DistributedType.FSDP:
        with fallback(model, accelerator) as unwrapped:
            checkpointing_enabled = bool(
                getattr(unwrapped, "is_gradient_checkpointing", False)
            )
            if checkpointing_enabled:
                unwrapped.gradient_checkpointing_disable()
            previous_use_cache = getattr(unwrapped.config, "use_cache", None)
            unwrapped.config.use_cache = True
            try:
                yield unwrapped
            finally:
                if previous_use_cache is not None:
                    unwrapped.config.use_cache = previous_use_cache
                if checkpointing_enabled:
                    unwrapped.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                    enable_input_grads = getattr(
                        unwrapped, "enable_input_require_grads", None
                    )
                    if enable_input_grads is not None:
                        enable_input_grads()
        return

    from torch.distributed.fsdp import FullyShardedDataParallel

    if not isinstance(model, FullyShardedDataParallel):
        raise RuntimeError("Accelerate reported FSDP but model is not FSDP-wrapped")
    with FullyShardedDataParallel.summon_full_params(
        model, recurse=True, writeback=False, rank0_only=False
    ):
        unwrapped = accelerator.unwrap_model(model)
        checkpointing_enabled = bool(
            getattr(unwrapped, "is_gradient_checkpointing", False)
        )
        if checkpointing_enabled:
            unwrapped.gradient_checkpointing_disable()
        previous_use_cache = getattr(unwrapped.config, "use_cache", None)
        unwrapped.config.use_cache = True
        with accelerator.autocast():
            try:
                yield unwrapped
            finally:
                if previous_use_cache is not None:
                    unwrapped.config.use_cache = previous_use_cache
                if checkpointing_enabled:
                    # Match the non-reentrant mode configured by the trainer.
                    # Calling this without kwargs silently restores the
                    # Transformers default (reentrant=True) after generation.
                    unwrapped.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                    # PEFT removes this hook when checkpointing is disabled.
                    # Restore it so the frozen embedding output feeds a graph
                    # into the trainable LoRA parameters on the next forward.
                    enable_input_grads = getattr(
                        unwrapped, "enable_input_require_grads", None
                    )
                    if enable_input_grads is not None:
                        enable_input_grads()
