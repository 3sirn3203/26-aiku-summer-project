from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


class ConfigError(ValueError):
    """Raised when the application configuration is invalid."""


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


# The smoke suite is an experiment-wide invariant, not a per-model setting.
FIXED_SMOKE_CONFIG = SmokeConfig(
    samples=(
        SmokeSampleConfig(
            index=864,
            db_id="network_1",
            category="single_table_select",
            question_sha256="f2c769010b3b6bf582b49204a53352980de393a2d78fde7281b4fe12c8f01bfb",
            gold_sql_sha256="72bdc66d75b466d16da13b1365b80926bf7713ee64c4382e5d82a5b4de460d2c",
        ),
        SmokeSampleConfig(
            index=702,
            db_id="world_1",
            category="where",
            question_sha256="6279df2ef3713802c1faa8acab924764cd865afbb7f4f76c595890c14b2378e7",
            gold_sql_sha256="f5d1e471ddde0f87f45a12a98910e52eef3767ba956c15d896e4005c8d4f5d07",
        ),
        SmokeSampleConfig(
            index=976,
            db_id="dog_kennels",
            category="order_by_limit",
            question_sha256="60b4d8c56aa0a3e907b16d0fe12f1041b1c3208a09e8f21253086aacd6e87b88",
            gold_sql_sha256="09496132855b66239bce1c5b1757e2b3b3e375fc132038d5121fe46735d1f48f",
        ),
        SmokeSampleConfig(
            index=424,
            db_id="museum_visit",
            category="aggregate",
            question_sha256="c95f44a014a5c297accc822d5e2c12c6df423eb59898c672729745ae28b14896",
            gold_sql_sha256="199fc2389108e5e6651329beedae26cec0fd0d08c8ef88366049f0722413816c",
        ),
        SmokeSampleConfig(
            index=848,
            db_id="orchestra",
            category="group_by",
            question_sha256="3b52658be21e9a368cfaa3f24a820fb58badcb9c9af11d39c6e57dd45414fb1d",
            gold_sql_sha256="cf0827fcd4b1521032bd50e17701f97aeddebc1df47bcb353021aded676070da",
        ),
        SmokeSampleConfig(
            index=1018,
            db_id="singer",
            category="join",
            question_sha256="8fbec14fceccb1da07c5493fd307b95156a38bc0048c300d979766b62c7071a5",
            gold_sql_sha256="b0a0aee65fede55e2a5b068da4fc42544f2eed3180049d7bb9f6ff79c2f36334",
        ),
        SmokeSampleConfig(
            index=409,
            db_id="course_teach",
            category="nested_query",
            question_sha256="949ead0d7eb7a8b6616b2cb7039b3515f75988c96c31ab788be3388ec15687f9",
            gold_sql_sha256="4ab59ff4baefb501e2f8b0df53171a845690c33be1e00726c46958033c2233e6",
        ),
        SmokeSampleConfig(
            index=499,
            db_id="battle_death",
            category="complex",
            question_sha256="819d2bc2f7a8c42ac130d80bd9243ec54d7839dc0b46f2a9027260febec4e368",
            gold_sql_sha256="a9a68b2626057f7ed05a7d27e5c2690c6e44465f90bb274e0921fc0f2a3e7d1b",
        ),
    ),
    minimum_executable=1,
)


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
    model_raw = _mapping(raw, "model")
    generation_raw = _mapping(raw, "generation")
    execution_raw = _mapping(raw, "execution")
    official_evaluation_raw = _mapping(raw, "official_evaluation")
    output_raw = _mapping(raw, "output")

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
        smoke=FIXED_SMOKE_CONFIG,
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
