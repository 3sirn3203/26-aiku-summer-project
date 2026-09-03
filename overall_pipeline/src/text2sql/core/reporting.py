from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence


SUMMARY_METRIC_KEYS = (
    "test_suite_accuracy",
    "exact_set_match_accuracy",
    "result_match_accuracy",
    "mean_vm_steps",
    "mean_latency_ms",
)


def metric_summary(
    *,
    test_suite_accuracy: Optional[float],
    exact_set_match_accuracy: Optional[float],
    result_match_accuracy: Optional[float],
    prediction_vm_steps: Mapping[str, Any],
    prediction_query_timing: Mapping[str, Any],
) -> Dict[str, Optional[float]]:
    vm_aggregate = prediction_vm_steps.get("complete_vm_steps_lower_bound", {})
    latency_aggregate = prediction_query_timing.get("query_elapsed_ms", {})
    return {
        "test_suite_accuracy": test_suite_accuracy,
        "exact_set_match_accuracy": exact_set_match_accuracy,
        "result_match_accuracy": result_match_accuracy,
        "mean_vm_steps": (
            vm_aggregate.get("mean")
            if isinstance(vm_aggregate, Mapping)
            else None
        ),
        "mean_latency_ms": (
            latency_aggregate.get("mean")
            if isinstance(latency_aggregate, Mapping)
            else None
        ),
    }


def empty_metric_summary() -> Dict[str, None]:
    return {key: None for key in SUMMARY_METRIC_KEYS}


def _selected(mapping: Mapping[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    return {key: mapping[key] for key in keys if key in mapping}


def _compact_source(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    return _selected(
        value,
        ("package_version", "python_tree_sha256", "scope"),
    )


def _compact_dataset(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    return _selected(
        value,
        (
            "split",
            "selection",
            "total_examples",
            "example_file",
            "tables_file",
            "example_file_sha256",
            "tables_file_sha256",
            "contract_sha256",
            "selected_database_sha256_before",
            "selected_databases_unchanged",
        ),
    )


def _compact_official_evaluation(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    compact = _selected(value, ("enabled", "preflight_sha256"))
    preflight = value.get("preflight")
    if isinstance(preflight, Mapping):
        compact["preflight"] = _selected(
            preflight,
            ("ok", "actual_commit", "evaluator_tree_sha256", "errors"),
        )
    result = value.get("result")
    if isinstance(result, Mapping):
        compact["result"] = _selected(
            result,
            ("ok", "processed_examples", "batch_size"),
        )
    return compact


def _compact_worker_status(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    compact = _selected(
        value,
        (
            "worker_id",
            "physical_gpu",
            "logical_device",
            "attempt",
            "status",
            "assigned_examples",
            "completed_examples",
            "error_type",
            "error_message",
        ),
    )
    backend = value.get("backend")
    if isinstance(backend, Mapping):
        compact_backend = _selected(
            backend,
            ("backend", "resolved_revision"),
        )
        if compact_backend:
            compact["backend"] = compact_backend
    return compact


def _compact_generation(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    compact = _selected(
        value,
        (
            "contract_sha256",
            "completed_examples",
            "shard_warnings",
            "resolved_revision",
        ),
    )
    attempts = value.get("attempts")
    if isinstance(attempts, list):
        compact["attempt_count"] = len(attempts)
    statuses = value.get("worker_statuses")
    if isinstance(statuses, list):
        compact["worker_statuses"] = [
            item
            for item in (_compact_worker_status(status) for status in statuses)
            if item is not None
        ]
    return compact


def _compact_backend(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    return _selected(
        value,
        (
            "backend",
            "model_id",
            "model_source",
            "requested_revision",
            "resolved_revision",
            "checkpoint_identity",
        ),
    )


def _compact_worker_runtime(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    roles = value.get("roles")
    if not isinstance(roles, Mapping):
        return None
    compact_roles: Dict[str, Any] = {}
    for role, role_value in roles.items():
        if not isinstance(role_value, Mapping):
            continue
        workers = role_value.get("workers")
        if not isinstance(workers, list):
            continue
        compact_roles[str(role)] = {
            "workers": [
                _selected(
                    worker,
                    (
                        "worker_id",
                        "physical_gpu",
                        "state",
                        "last_return_code",
                    ),
                )
                for worker in workers
                if isinstance(worker, Mapping)
            ]
        }
    return {"roles": compact_roles}


def compact_run_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep one canonical config plus compact provenance and resume fields."""

    compact = _selected(
        manifest,
        (
            "schema_version",
            "run_id",
            "run_type",
            "status",
            "started_at",
            "finished_at",
            "elapsed_seconds",
            "backend_selector",
            "config_source_path",
            "source_config_sha256",
            "config_sha256",
            "experiment_config_sha256",
            "agent_contract_sha256",
            "invocation",
            "resume_invocations",
            "gpu_ids",
            "gpu_pool_history",
            "config",
            "runtime",
            "adapter",
            "failure",
            "artifacts",
        ),
    )

    source = _compact_source(manifest.get("source"))
    if source is not None:
        compact["source"] = source
    dataset = _compact_dataset(manifest.get("dataset"))
    if dataset is not None:
        compact["dataset"] = dataset
    official = _compact_official_evaluation(manifest.get("official_evaluation"))
    if official is not None:
        compact["official_evaluation"] = official
    generation = _compact_generation(manifest.get("generation"))
    if generation is not None:
        compact["generation"] = generation
    backend = _compact_backend(manifest.get("backend"))
    if backend is not None:
        compact["backend"] = backend
    worker_runtime = _compact_worker_runtime(manifest.get("worker_runtime"))
    if worker_runtime is not None:
        compact["worker_runtime"] = worker_runtime

    contract = manifest.get("contract")
    if isinstance(contract, Mapping):
        compact["contract"] = _selected(
            contract,
            (
                "run_type",
                "selection",
                "single_turn",
                "generation_calls_per_example",
                "execution_feedback",
                "self_correction",
                "few_shot",
                "database_values_in_prompt",
            ),
        )
    agent_contract = manifest.get("agent_contract")
    if isinstance(agent_contract, Mapping):
        compact["agent_contract"] = _selected(
            agent_contract,
            ("workflow", "selection", "max_iterations", "final_evaluation"),
        )
    roles = manifest.get("roles")
    if isinstance(roles, Mapping):
        compact["roles"] = {
            str(role): _selected(value, ("resolved_revisions",))
            for role, value in roles.items()
            if isinstance(value, Mapping)
        }
    return compact
