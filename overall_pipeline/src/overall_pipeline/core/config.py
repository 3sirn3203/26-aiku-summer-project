from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name_or_path: str = ""
    tokenizer_name_or_path: str | None = None
    adapter_name_or_path: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    dtype: str = "auto"
    device: str = "auto"
    mode: str = "full"
    gradient_checkpointing: bool = False
    chat_format: str = "auto"
    chat_template: str | None = None
    chat_template_kwargs: dict = field(default_factory=dict)
    model_kwargs: dict = field(default_factory=dict)
    tokenizer_kwargs: dict = field(default_factory=dict)


@dataclass
class RolloutConfig:
    max_intermediate: int = 3
    max_context_tokens: int = 4096
    max_new_tokens: int = 256
    temperature: float = 1.0
    exact_match_alpha: float = 1.5
    efficiency_beta: float = 0.2
    non_executable_penalty: float = 0.25


@dataclass
class SQLConfig:
    timeout_seconds: float = 5.0
    max_rows: int = 20
    max_response_chars: int = 6000
    max_sql_chars: int = 20000
    max_result_rows: int = 100000
    max_result_bytes: int = 16000000
    worker_memory_mb: int = 1024
    evaluator_path: str | None = None
    nltk_data: str | None = None
    suite_database_dir: str | None = None
    reward_metric: str = "execution"

