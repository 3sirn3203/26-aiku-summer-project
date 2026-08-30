from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from text2sql.config import (
    ConfigError,
    ExecutionConfig,
    GenerationConfig,
    ModelConfig,
    OfficialEvaluationConfig,
    OutputConfig,
    SmokeConfig,
    SmokeSampleConfig,
    SpiderConfig,
    _boolean,
    _integer,
    _mapping,
    _model_id,
    _model_source,
    _number,
    _resolve_path,
    _string,
)


ROLE_CONTRACTS = {
    "planner": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "max_new_tokens": 512,
        "minimum_free_vram_bytes": 8 * 1024**3,
    },
    "coder": {
        "model_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "revision": "ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
        "max_new_tokens": 512,
        "minimum_free_vram_bytes": 4 * 1024**3,
    },
    "verifier": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "max_new_tokens": 384,
        "minimum_free_vram_bytes": 8 * 1024**3,
    },
}


@dataclass(frozen=True)
class AgentRoleConfig:
    model: ModelConfig
    generation: GenerationConfig
    trainable: bool
    minimum_free_vram_bytes: int


@dataclass(frozen=True)
class AgentWorkflowConfig:
    max_iterations: int
    observation_max_rows: int
    observation_max_bytes: int
    infrastructure_retry_limit: int
    execution_concurrency: int


@dataclass(frozen=True)
class AgentGpuPoolsConfig:
    planner: Sequence[int]
    coder: Sequence[int]
    verifier: Sequence[int]


@dataclass(frozen=True)
class AgentAppConfig:
    source_path: Path
    spider: SpiderConfig
    smoke: SmokeConfig
    roles: Mapping[str, AgentRoleConfig]
    workflow: AgentWorkflowConfig
    gpu_pools: AgentGpuPoolsConfig
    execution: ExecutionConfig
    official_evaluation: OfficialEvaluationConfig
    output: OutputConfig
    raw: Mapping[str, Any]


def _require_exact_keys(
    payload: Mapping[str, Any], expected: Sequence[str], label: str
) -> None:
    actual = set(payload)
    wanted = set(expected)
    if actual != wanted:
        missing = sorted(wanted - actual)
        extra = sorted(actual - wanted)
        details = []
        if missing:
            details.append("missing=%s" % missing)
        if extra:
            details.append("extra=%s" % extra)
        raise ConfigError("%s fields do not match the contract (%s)" % (label, ", ".join(details)))


def _gpu_ids(parent: Mapping[str, Any], key: str) -> Sequence[int]:
    value = parent.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError("gpu_pools.%s must be a non-empty list" % key)
    result = []
    for position, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ConfigError(
                "gpu_pools.%s[%d] must be a non-negative integer"
                % (key, position)
            )
        result.append(item)
    if len(set(result)) != len(result):
        raise ConfigError("gpu_pools.%s must not contain duplicates" % key)
    return tuple(result)


def _role_config(base: Path, roles: Mapping[str, Any], role: str) -> AgentRoleConfig:
    role_raw = _mapping(roles, role)
    model_raw = _mapping(role_raw, "model")
    generation_raw = _mapping(role_raw, "generation")
    _require_exact_keys(
        role_raw,
        ("model", "generation", "trainable", "minimum_free_vram_bytes"),
        "roles.%s" % role,
    )
    model_fields = (
        "id",
        "revision",
        "dtype",
        "device",
        "attention_implementation",
        "trust_remote_code",
        "cache_dir",
    )
    if "source" in model_raw:
        model_fields = model_fields + ("source",)
    _require_exact_keys(model_raw, model_fields, "roles.%s.model" % role)
    _require_exact_keys(
        generation_raw,
        (
            "do_sample",
            "num_beams",
            "repetition_penalty",
            "max_time_seconds",
            "max_input_tokens",
            "max_new_tokens",
            "batch_size",
        ),
        "roles.%s.generation" % role,
    )
    cache_value = model_raw.get("cache_dir")
    if cache_value is not None and not isinstance(cache_value, str):
        raise ConfigError("roles.%s.model.cache_dir must be a string or null" % role)
    source = _model_source(model_raw)
    model = ModelConfig(
        model_id=_model_id(base, model_raw, source),
        revision=_string(model_raw, "revision"),
        dtype=_string(model_raw, "dtype"),
        device=_string(model_raw, "device"),
        attention_implementation=_string(model_raw, "attention_implementation"),
        trust_remote_code=_boolean(model_raw, "trust_remote_code"),
        cache_dir=_resolve_path(base, cache_value),
        source=source,
    )
    generation = GenerationConfig(
        do_sample=_boolean(generation_raw, "do_sample"),
        num_beams=_integer(generation_raw, "num_beams", minimum=1),
        repetition_penalty=_number(
            generation_raw, "repetition_penalty", minimum=0.0
        ),
        max_time_seconds=_number(generation_raw, "max_time_seconds"),
        max_input_tokens=_integer(generation_raw, "max_input_tokens", minimum=1),
        max_new_tokens=_integer(generation_raw, "max_new_tokens", minimum=1),
        batch_size=_integer(generation_raw, "batch_size", minimum=1),
    )
    if model.source == "hub":
        if re.fullmatch(r"[0-9a-f]{40}", model.revision) is None:
            raise ConfigError("roles.%s.model.revision must be a pinned commit" % role)
    elif model.revision != "local":
        raise ConfigError("roles.%s local model revision must be 'local'" % role)
    if model.dtype != "float32" or model.attention_implementation != "eager":
        raise ConfigError("roles.%s must use float32 and eager attention" % role)
    if model.trust_remote_code:
        raise ConfigError("roles.%s must disable trust_remote_code" % role)
    if model.device != "cuda:0":
        raise ConfigError("roles.%s.model.device must be cuda:0" % role)
    if (
        generation.do_sample
        or generation.num_beams != 1
        or generation.repetition_penalty != 1.0
        or generation.batch_size != 1
    ):
        raise ConfigError("roles.%s must use deterministic greedy batch-1 decoding" % role)
    result = AgentRoleConfig(
        model=model,
        generation=generation,
        trainable=_boolean(role_raw, "trainable"),
        minimum_free_vram_bytes=_integer(
            role_raw, "minimum_free_vram_bytes", minimum=1
        ),
    )
    contract = ROLE_CONTRACTS[role]
    if result.model.source == "hub" and (
        result.model.model_id != contract["model_id"]
        or result.model.revision != contract["revision"]
    ):
        raise ConfigError(
            "roles.%s.model must use the contracted model ID and revision" % role
        )
    if (
        result.generation.max_time_seconds != 120.0
        or result.generation.max_input_tokens != 8192
        or result.generation.max_new_tokens != contract["max_new_tokens"]
    ):
        raise ConfigError(
            "roles.%s generation limits must match the agent experiment contract"
            % role
        )
    if (
        result.minimum_free_vram_bytes
        != contract["minimum_free_vram_bytes"]
    ):
        raise ConfigError(
            "roles.%s.minimum_free_vram_bytes must match the role contract" % role
        )
    return result


def load_agent_config(path: Path) -> AgentAppConfig:
    source_path = Path(path).expanduser().resolve()
    try:
        raw: Dict[str, Any] = json.loads(source_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError("Configuration file does not exist: %s" % source_path) from exc
    except json.JSONDecodeError as exc:
        raise ConfigError("Invalid JSON configuration: %s" % exc) from exc
    if not isinstance(raw, dict):
        raise ConfigError("Top-level configuration must be an object")
    _require_exact_keys(
        raw,
        (
            "spider",
            "smoke",
            "roles",
            "workflow",
            "gpu_pools",
            "execution",
            "official_evaluation",
            "output",
        ),
        "top-level configuration",
    )

    base = source_path.parent
    spider_raw = _mapping(raw, "spider")
    smoke_raw = _mapping(raw, "smoke")
    roles_raw = _mapping(raw, "roles")
    workflow_raw = _mapping(raw, "workflow")
    gpu_raw = _mapping(raw, "gpu_pools")
    execution_raw = _mapping(raw, "execution")
    official_raw = _mapping(raw, "official_evaluation")
    output_raw = _mapping(raw, "output")
    _require_exact_keys(
        spider_raw,
        ("root", "examples_file", "tables_file", "database_dir", "split"),
        "spider",
    )
    _require_exact_keys(smoke_raw, ("samples", "minimum_executable"), "smoke")
    _require_exact_keys(roles_raw, ("planner", "coder", "verifier"), "roles")
    _require_exact_keys(
        workflow_raw,
        (
            "max_iterations",
            "observation_max_rows",
            "observation_max_bytes",
            "infrastructure_retry_limit",
            "execution_concurrency",
        ),
        "workflow",
    )
    _require_exact_keys(gpu_raw, ("planner", "coder", "verifier"), "gpu_pools")
    _require_exact_keys(
        execution_raw,
        (
            "timeout_seconds",
            "max_sql_bytes",
            "max_result_rows",
            "max_result_bytes",
            "worker_memory_limit_bytes",
        ),
        "execution",
    )
    _require_exact_keys(
        official_raw,
        (
            "enabled",
            "evaluator_root",
            "test_suite_database_root",
            "upstream_url",
            "upstream_commit",
            "plug_value",
            "keep_distinct",
            "timeout_seconds",
            "nltk_data_dir",
        ),
        "official_evaluation",
    )
    _require_exact_keys(output_raw, ("directory",), "output")

    samples_raw = smoke_raw.get("samples")
    if not isinstance(samples_raw, list) or not samples_raw:
        raise ConfigError("smoke.samples must be a non-empty list")
    samples = []
    for position, sample_raw in enumerate(samples_raw):
        if not isinstance(sample_raw, dict):
            raise ConfigError("smoke.samples[%d] must be an object" % position)
        _require_exact_keys(
            sample_raw,
            (
                "index",
                "db_id",
                "category",
                "question_sha256",
                "gold_sql_sha256",
            ),
            "smoke.samples[%d]" % position,
        )
        sample = SmokeSampleConfig(
            index=_integer(sample_raw, "index", minimum=0),
            db_id=_string(sample_raw, "db_id"),
            category=_string(sample_raw, "category"),
            question_sha256=_string(sample_raw, "question_sha256"),
            gold_sql_sha256=_string(sample_raw, "gold_sql_sha256"),
        )
        for digest in (sample.question_sha256, sample.gold_sql_sha256):
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ConfigError("smoke sample hashes must be lowercase SHA-256")
        samples.append(sample)
    if len({sample.index for sample in samples}) != len(samples):
        raise ConfigError("smoke.samples must not contain duplicate indices")

    spider_root = _resolve_path(base, _string(spider_raw, "root"))
    evaluator_root = _resolve_path(base, _string(official_raw, "evaluator_root"))
    test_suite_root = _resolve_path(
        base, _string(official_raw, "test_suite_database_root")
    )
    output_dir = _resolve_path(base, _string(output_raw, "directory"))
    nltk_value: Optional[str] = official_raw.get("nltk_data_dir")
    if nltk_value is not None and not isinstance(nltk_value, str):
        raise ConfigError("official_evaluation.nltk_data_dir must be a string or null")
    assert spider_root is not None
    assert evaluator_root is not None
    assert test_suite_root is not None
    assert output_dir is not None

    roles = {
        role: _role_config(base, roles_raw, role)
        for role in ("planner", "coder", "verifier")
    }
    if not roles["planner"].trainable or not roles["coder"].trainable:
        raise ConfigError("planner and coder must be marked trainable")
    if roles["verifier"].trainable:
        raise ConfigError("verifier must be marked frozen")

    gpu_pools = AgentGpuPoolsConfig(
        planner=_gpu_ids(gpu_raw, "planner"),
        coder=_gpu_ids(gpu_raw, "coder"),
        verifier=_gpu_ids(gpu_raw, "verifier"),
    )
    all_gpu_ids = tuple(gpu_pools.planner + gpu_pools.coder + gpu_pools.verifier)
    if len(set(all_gpu_ids)) != len(all_gpu_ids):
        raise ConfigError("GPU IDs must not overlap across role pools")

    config = AgentAppConfig(
        source_path=source_path,
        spider=SpiderConfig(
            root=spider_root,
            examples_file=_string(spider_raw, "examples_file"),
            tables_file=_string(spider_raw, "tables_file"),
            database_dir=_string(spider_raw, "database_dir"),
            split=_string(spider_raw, "split"),
        ),
        smoke=SmokeConfig(
            samples=tuple(samples),
            minimum_executable=_integer(smoke_raw, "minimum_executable", minimum=1),
        ),
        roles=roles,
        workflow=AgentWorkflowConfig(
            max_iterations=_integer(workflow_raw, "max_iterations", minimum=1),
            observation_max_rows=_integer(
                workflow_raw, "observation_max_rows", minimum=1
            ),
            observation_max_bytes=_integer(
                workflow_raw, "observation_max_bytes", minimum=1
            ),
            infrastructure_retry_limit=_integer(
                workflow_raw, "infrastructure_retry_limit", minimum=0
            ),
            execution_concurrency=_integer(
                workflow_raw, "execution_concurrency", minimum=1
            ),
        ),
        gpu_pools=gpu_pools,
        execution=ExecutionConfig(
            timeout_seconds=_number(execution_raw, "timeout_seconds"),
            max_sql_bytes=_integer(execution_raw, "max_sql_bytes", minimum=1),
            max_result_rows=_integer(
                execution_raw, "max_result_rows", minimum=1
            ),
            max_result_bytes=_integer(
                execution_raw, "max_result_bytes", minimum=1
            ),
            worker_memory_limit_bytes=_integer(
                execution_raw, "worker_memory_limit_bytes", minimum=1
            ),
        ),
        official_evaluation=OfficialEvaluationConfig(
            enabled=_boolean(official_raw, "enabled"),
            evaluator_root=evaluator_root,
            test_suite_database_root=test_suite_root,
            upstream_url=_string(official_raw, "upstream_url"),
            upstream_commit=_string(official_raw, "upstream_commit"),
            plug_value=_boolean(official_raw, "plug_value"),
            keep_distinct=_boolean(official_raw, "keep_distinct"),
            timeout_seconds=_number(official_raw, "timeout_seconds"),
            nltk_data_dir=_resolve_path(base, nltk_value),
        ),
        output=OutputConfig(directory=output_dir),
        raw=raw,
    )
    if config.workflow.max_iterations != 3:
        raise ConfigError("workflow.max_iterations must be exactly 3")
    if config.workflow.observation_max_rows != 5:
        raise ConfigError("workflow.observation_max_rows must be exactly 5")
    if config.workflow.observation_max_bytes != 4096:
        raise ConfigError("workflow.observation_max_bytes must be exactly 4096")
    if config.workflow.infrastructure_retry_limit != 1:
        raise ConfigError("workflow.infrastructure_retry_limit must be exactly 1")
    if config.workflow.execution_concurrency != 2:
        raise ConfigError("workflow.execution_concurrency must be exactly 2")
    if config.smoke.minimum_executable > len(config.smoke.samples):
        raise ConfigError("smoke.minimum_executable cannot exceed sample count")
    if re.fullmatch(r"[0-9a-f]{40}", config.official_evaluation.upstream_commit) is None:
        raise ConfigError("official evaluator commit must be pinned")
    if not config.official_evaluation.enabled:
        raise ConfigError("official evaluation must remain enabled")
    if config.official_evaluation.plug_value or config.official_evaluation.keep_distinct:
        raise ConfigError("official evaluator flags must remain disabled")
    return config
