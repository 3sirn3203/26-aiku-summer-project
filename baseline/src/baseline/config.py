from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from rrcm_sql.config import ModelConfig as RRCMModelConfig, RolloutConfig, SQLConfig


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen3-1.7B"
    tokenizer_name_or_path: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = True
    dtype: str = "float16"
    device: str = "cuda:0"
    chat_format: str = "chat"
    chat_template: str | None = None
    chat_template_kwargs: dict = field(default_factory=lambda: {"enable_thinking": False})
    model_kwargs: dict = field(default_factory=lambda: {"attn_implementation": "sdpa"})
    tokenizer_kwargs: dict = field(default_factory=dict)

    def policy_config(self, device=None):
        return RRCMModelConfig(
            name_or_path=self.name_or_path,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
            dtype=self.dtype,
            device=device or self.device,
            mode="full",
            gradient_checkpointing=False,
            chat_format=self.chat_format,
            chat_template=self.chat_template,
            chat_template_kwargs=self.chat_template_kwargs,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
        )


@dataclass
class DataConfig:
    test_json: str = "data/spider_data/test.json"
    test_tables: str = "data/spider_data/test_tables.json"
    test_database_dir: str = "data/spider_data/test_database"


@dataclass
class GenerationConfig:
    mode: str = "single_turn"
    max_intermediate: int = 0
    max_context_tokens: int = 4096
    max_new_tokens: int = 256

    def rollout_config(self):
        return RolloutConfig(
            max_intermediate=self.max_intermediate,
            max_context_tokens=self.max_context_tokens,
            max_new_tokens=self.max_new_tokens,
            temperature=1.0,
        )


@dataclass
class EvaluationConfig:
    seed: int = 42
    timeout_seconds: float = 172800
    worker_timeout: float = 1800
    evaluator_path: str | None = "overall_pipeline/vendor/spider_test_suite_eval"
    nltk_data: str | None = "data/nltk_data"
    test_suite_database_dir: str | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    sql: SQLConfig = field(default_factory=SQLConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self):
        if self.generation.mode not in {"single_turn", "multi_turn"}:
            raise ValueError("generation.mode must be single_turn or multi_turn")
        expected = 0 if self.generation.mode == "single_turn" else 3
        if self.generation.max_intermediate != expected:
            raise ValueError(f"{self.generation.mode} requires max_intermediate={expected}")
        if self.model.dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("Invalid model dtype")
        if self.model.chat_format not in {"auto", "chat", "plain"}:
            raise ValueError("Invalid chat format")
        if min(self.generation.max_context_tokens, self.generation.max_new_tokens,
               self.evaluation.timeout_seconds, self.evaluation.worker_timeout) <= 0:
            raise ValueError("Token limits and timeouts must be positive")
        for path in (self.data.test_json, self.data.test_tables, self.data.test_database_dir):
            if not Path(path).exists():
                raise FileNotFoundError(path)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


def load_config(path):
    raw = json.loads(Path(path).read_text())
    allowed = {"model", "data", "generation", "sql", "evaluation"}
    if raw.keys() - allowed:
        raise ValueError(f"Unknown config sections: {raw.keys() - allowed}")
    cfg = Config(
        model=ModelConfig(**raw.get("model", {})),
        data=DataConfig(**raw.get("data", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
        sql=SQLConfig(**raw.get("sql", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
    )
    cfg.validate()
    return cfg

