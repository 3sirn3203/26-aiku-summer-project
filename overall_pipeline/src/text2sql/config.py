from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


class ConfigError(ValueError):
    """Raised when the smoke-test configuration is invalid."""


@dataclass(frozen=True)
class SpiderConfig:
    root: Path
    examples_file: str
    tables_file: str
    database_dir: str
    split: str


@dataclass(frozen=True)
class SmokeSampleConfig:
    index: int
    db_id: str
    category: str
    question_sha256: str
    gold_sql_sha256: str


@dataclass(frozen=True)
class SmokeConfig:
    samples: Sequence[SmokeSampleConfig]
    minimum_executable: int


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    revision: str
    dtype: str
    device: str
    attention_implementation: str
    trust_remote_code: bool
    cache_dir: Optional[Path]
    source: str = "hub"
    checkpoint_identity: Optional[str] = None


@dataclass(frozen=True)
class GenerationConfig:
    do_sample: bool
    num_beams: int
    repetition_penalty: float
    max_time_seconds: float
    max_input_tokens: int
    max_new_tokens: int
    batch_size: int


@dataclass(frozen=True)
class ExecutionConfig:
    timeout_seconds: float
    max_sql_bytes: int
    # Compatibility names: these bound the retained/IPC row prefix, not the
    # number of rows consumed from a successful cursor. max_result_bytes also
    # bounds one encoded row and, where supported, a SQLite value.
    max_result_rows: int
    max_result_bytes: int
    worker_memory_limit_bytes: int


@dataclass(frozen=True)
class OfficialEvaluationConfig:
    enabled: bool
    evaluator_root: Path
    test_suite_database_root: Path
    upstream_url: str
    upstream_commit: str
    plug_value: bool
    keep_distinct: bool
    timeout_seconds: float
    nltk_data_dir: Optional[Path]


@dataclass(frozen=True)
class OutputConfig:
    directory: Path


@dataclass(frozen=True)
class AppConfig:
    source_path: Path
    spider: SpiderConfig
    smoke: SmokeConfig
    model: ModelConfig
    generation: GenerationConfig
    execution: ExecutionConfig
    official_evaluation: OfficialEvaluationConfig
    output: OutputConfig
    raw: Mapping[str, Any]


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigError("Configuration field %r must be an object" % key)
    return value


def _string(parent: Mapping[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("Configuration field %r must be a non-empty string" % key)
    return value


def _integer(parent: Mapping[str, Any], key: str, minimum: int = 0) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(
            "Configuration field %r must be an integer >= %d" % (key, minimum)
        )
    return value


def _number(parent: Mapping[str, Any], key: str, minimum: float = 0.0) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("Configuration field %r must be a number" % key)
    number = float(value)
    if not math.isfinite(number) or number <= minimum:
        raise ConfigError("Configuration field %r must be > %s" % (key, minimum))
    return number


def _boolean(parent: Mapping[str, Any], key: str) -> bool:
    value = parent.get(key)
    if not isinstance(value, bool):
        raise ConfigError("Configuration field %r must be a boolean" % key)
    return value


def _resolve_path(base: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("Configured path must be a non-empty string or null")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _model_source(parent: Mapping[str, Any]) -> str:
    value = parent.get("source", "hub")
    if value not in {"hub", "local"}:
        raise ConfigError("model.source must be either 'hub' or 'local'")
    return str(value)


def _model_id(base: Path, parent: Mapping[str, Any], source: str) -> str:
    value = _string(parent, "id")
    if source == "hub":
        return value
    path = _resolve_path(base, value)
    assert path is not None
    return str(path)


def load_config(path: Path) -> AppConfig:
    source_path = Path(path).expanduser().resolve()
    try:
        with source_path.open("r", encoding="utf-8") as handle:
            raw: Dict[str, Any] = json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError("Configuration file does not exist: %s" % source_path) from exc
    except json.JSONDecodeError as exc:
        raise ConfigError("Invalid JSON configuration: %s" % exc) from exc

    if not isinstance(raw, dict):
        raise ConfigError("Top-level configuration must be an object")

    base = source_path.parent
    spider_raw = _mapping(raw, "spider")
    smoke_raw = _mapping(raw, "smoke")
    model_raw = _mapping(raw, "model")
    generation_raw = _mapping(raw, "generation")
    execution_raw = _mapping(raw, "execution")
    official_evaluation_raw = _mapping(raw, "official_evaluation")
    output_raw = _mapping(raw, "output")

    samples_raw = smoke_raw.get("samples")
    if not isinstance(samples_raw, list) or not samples_raw:
        raise ConfigError("smoke.samples must be a non-empty list")
    samples = []
    for position, sample_raw in enumerate(samples_raw):
        if not isinstance(sample_raw, dict):
            raise ConfigError("smoke.samples[%d] must be an object" % position)
        samples.append(
            SmokeSampleConfig(
                index=_integer(sample_raw, "index", minimum=0),
                db_id=_string(sample_raw, "db_id"),
                category=_string(sample_raw, "category"),
                question_sha256=_string(sample_raw, "question_sha256"),
                gold_sql_sha256=_string(sample_raw, "gold_sql_sha256"),
            )
        )
    indices = [sample.index for sample in samples]
    if len(set(indices)) != len(indices):
        raise ConfigError("smoke.samples must not contain duplicate indices")
    for sample in samples:
        for label, digest in (
            ("question_sha256", sample.question_sha256),
            ("gold_sql_sha256", sample.gold_sql_sha256),
        ):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ConfigError(
                    "smoke sample %d has an invalid %s" % (sample.index, label)
                )

    cache_value = model_raw.get("cache_dir")
    if cache_value is not None and not isinstance(cache_value, str):
        raise ConfigError("model.cache_dir must be a string or null")

    spider_root = _resolve_path(base, _string(spider_raw, "root"))
    evaluator_root = _resolve_path(
        base, _string(official_evaluation_raw, "evaluator_root")
    )
    test_suite_database_root = _resolve_path(
        base, _string(official_evaluation_raw, "test_suite_database_root")
    )
    nltk_data_value = official_evaluation_raw.get("nltk_data_dir")
    if nltk_data_value is not None and not isinstance(nltk_data_value, str):
        raise ConfigError(
            "official_evaluation.nltk_data_dir must be a string or null"
        )
    output_directory = _resolve_path(base, _string(output_raw, "directory"))
    assert spider_root is not None
    assert evaluator_root is not None
    assert test_suite_database_root is not None
    assert output_directory is not None

    config = AppConfig(
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
        model=ModelConfig(
            model_id=_model_id(base, model_raw, _model_source(model_raw)),
            revision=_string(model_raw, "revision"),
            dtype=_string(model_raw, "dtype"),
            device=_string(model_raw, "device"),
            attention_implementation=_string(model_raw, "attention_implementation"),
            trust_remote_code=_boolean(model_raw, "trust_remote_code"),
            cache_dir=_resolve_path(base, cache_value),
            source=_model_source(model_raw),
        ),
        generation=GenerationConfig(
            do_sample=_boolean(generation_raw, "do_sample"),
            num_beams=_integer(generation_raw, "num_beams", minimum=1),
            repetition_penalty=_number(
                generation_raw, "repetition_penalty", minimum=0.0
            ),
            max_time_seconds=_number(generation_raw, "max_time_seconds"),
            max_input_tokens=_integer(generation_raw, "max_input_tokens", minimum=1),
            max_new_tokens=_integer(generation_raw, "max_new_tokens", minimum=1),
            batch_size=_integer(generation_raw, "batch_size", minimum=1),
        ),
        execution=ExecutionConfig(
            timeout_seconds=_number(execution_raw, "timeout_seconds"),
            max_sql_bytes=_integer(execution_raw, "max_sql_bytes", minimum=1),
            max_result_rows=_integer(execution_raw, "max_result_rows", minimum=1),
            max_result_bytes=_integer(execution_raw, "max_result_bytes", minimum=1),
            worker_memory_limit_bytes=_integer(
                execution_raw, "worker_memory_limit_bytes", minimum=1
            ),
        ),
        official_evaluation=OfficialEvaluationConfig(
            enabled=_boolean(official_evaluation_raw, "enabled"),
            evaluator_root=evaluator_root,
            test_suite_database_root=test_suite_database_root,
            upstream_url=_string(official_evaluation_raw, "upstream_url"),
            upstream_commit=_string(official_evaluation_raw, "upstream_commit"),
            plug_value=_boolean(official_evaluation_raw, "plug_value"),
            keep_distinct=_boolean(official_evaluation_raw, "keep_distinct"),
            timeout_seconds=_number(official_evaluation_raw, "timeout_seconds"),
            nltk_data_dir=_resolve_path(base, nltk_data_value),
        ),
        output=OutputConfig(directory=output_directory),
        raw=raw,
    )

    if config.smoke.minimum_executable > len(config.smoke.samples):
        raise ConfigError("smoke.minimum_executable cannot exceed the sample count")
    if config.generation.batch_size != 1:
        raise ConfigError("The single-turn smoke test currently requires batch_size=1")
    if config.model.trust_remote_code:
        raise ConfigError("trust_remote_code must remain false for this project")
    if config.model.source == "local" and config.model.revision != "local":
        raise ConfigError("local model.revision must be exactly 'local'")
    if config.model.dtype != "float32":
        raise ConfigError("The initial TITAN Xp smoke test requires model.dtype=float32")
    if config.model.attention_implementation != "eager":
        raise ConfigError("The TITAN Xp smoke test requires eager attention")
    if re.fullmatch(r"cuda:[0-9]+", config.model.device) is None:
        raise ConfigError("model.device must use the explicit cuda:<index> form")
    if config.generation.do_sample:
        raise ConfigError("The single-turn baseline requires do_sample=false")
    if config.generation.num_beams != 1:
        raise ConfigError("The single-turn baseline requires num_beams=1")
    if config.generation.repetition_penalty != 1.0:
        raise ConfigError(
            "The single-turn baseline requires repetition_penalty=1.0"
        )
    if (
        re.fullmatch(
            r"[0-9a-f]{40}", config.official_evaluation.upstream_commit
        )
        is None
    ):
        raise ConfigError(
            "official_evaluation.upstream_commit must be a 40-character "
            "lowercase hexadecimal Git commit"
        )
    if config.official_evaluation.plug_value:
        raise ConfigError(
            "The official Spider evaluation contract requires plug_value=false"
        )
    if config.official_evaluation.keep_distinct:
        raise ConfigError(
            "The official Spider evaluation contract requires keep_distinct=false"
        )
    return config
