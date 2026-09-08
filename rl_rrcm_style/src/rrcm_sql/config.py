from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


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
    reference_device: str | None = None
    mode: str = "lora"  # lora, full
    quantization: str | None = None  # 4bit, 8bit
    lora_rank: int = 16
    lora_alpha: int = 32
    target_modules: str | list[str] = "all-linear"
    gradient_checkpointing: bool = True
    chat_format: str = "auto"  # auto, chat, plain
    chat_template: str | None = None  # Jinja file
    chat_template_kwargs: dict = field(default_factory=dict)
    model_kwargs: dict = field(default_factory=dict)
    tokenizer_kwargs: dict = field(default_factory=dict)


@dataclass
class DataConfig:
    split_mode: str = "legacy_internal"
    train_json: str = "data/spider_data/train_spider.json"
    dev_json: str = "data/spider_data/dev.json"
    test_json: str = "data/spider_data/test.json"
    train_others_json: str | None = None
    test_database_dir: str = "data/spider_data/test_database"
    test_tables: str = "data/spider_data/test_tables.json"
    prepared_dir: str = "rl_rrcm_style/prepared"
    database_dir: str = "data/spider_data/database"
    tables: str = "data/spider_data/tables.json"


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
    reward_metric: str = "execution"  # execution, test_suite


@dataclass
class RolloutConfig:
    max_intermediate: int = 3
    max_context_tokens: int = 4096
    max_new_tokens: int = 256
    temperature: float = 1.0
    group_size: int = 4
    efficiency_beta: float = 0.2
    non_executable_penalty: float = 0.25
    prompted_trajectories: int = 0
    answer_probability: float = 0.5


@dataclass
class RuntimeConfig:
    update_backend: str = "single"
    update_devices: list[str] = field(default_factory=list)
    rollout_devices: list[str] = field(default_factory=list)
    worker_timeout: float = 1800
    fsdp_wrap_classes: list[str] = field(default_factory=list)


@dataclass
class TrainConfig:
    output_dir: str = "rl_rrcm_style/runs/default"
    seed: int = 42
    max_steps: int = 100
    max_groups: int = 2000
    groups_per_update: int = 1
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    kl_coefficient: float = 0.001
    clip_ratio: float = 0.2
    update_epochs: int = 1
    max_grad_norm: float = 1.0
    max_consecutive_amp_skips: int = 5
    save_steps: int = 25
    sft_epochs: int = 1


@dataclass
class EvaluationConfig:
    enabled: bool = False
    every_steps: int = 25
    at_start: bool = True
    at_end: bool = True
    limit: int | None = None
    selection_metric: str = "execution_accuracy"
    timeout_seconds: float = 86400
    dev_suite_database_dir: str | None = None
    test_suite_database_dir: str | None = None


@dataclass
class TrackingConfig:
    backend: str = "none"
    project: str = "rl-rrcm-style"
    entity: str | None = None
    run_name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str = "online"
    log_trajectories: bool = False
    upload_checkpoints: bool = False
    resume_run: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sql: SQLConfig = field(default_factory=SQLConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def validate(self):
        if self.data.split_mode not in {"official", "legacy_internal"}:
            raise ValueError("Unknown data.split_mode")
        e = self.evaluation
        if e.every_steps <= 0 or e.timeout_seconds <= 0 or (e.limit is not None and e.limit <= 0):
            raise ValueError("Evaluation intervals, timeout and limit must be positive")
        if e.selection_metric not in {"execution_accuracy", "test_suite_accuracy", "exact_match"}:
            raise ValueError("Unknown checkpoint selection metric")
        if e.enabled and e.selection_metric == "test_suite_accuracy" and not e.dev_suite_database_dir:
            raise ValueError("Test-suite selection requires evaluation.dev_suite_database_dir")
        if e.enabled and e.selection_metric == "exact_match" and not self.sql.evaluator_path:
            raise ValueError("Exact-match selection requires sql.evaluator_path")
        if self.tracking.backend not in {"none", "wandb"} or self.tracking.mode not in {"online", "offline", "disabled"}:
            raise ValueError("Invalid tracking backend/mode")
        r = self.runtime
        if r.update_backend not in {"single", "fsdp"}:
            raise ValueError("Invalid runtime backend")
        if not 0 <= self.rollout.prompted_trajectories <= self.rollout.group_size:
            raise ValueError("prompted_trajectories must be within group_size")
        if not 0 <= self.rollout.answer_probability <= 1:
            raise ValueError("answer_probability must be in [0, 1]")
        if r.worker_timeout <= 0:
            raise ValueError("worker_timeout must be positive")
        if r.update_backend == "single" and len(r.update_devices) > 1:
            raise ValueError("single update requires at most one device")
        if r.update_backend == "single" and r.update_devices and r.update_devices[0] != self.model.device:
            raise ValueError("single update_devices[0] must match model.device")
        if r.update_backend == "fsdp":
            if len(r.update_devices) != 2 or not r.rollout_devices:
                raise ValueError("FSDP requires two update_devices and dedicated rollout_devices (cpu allowed)")
            if self.model.quantization:
                raise ValueError("FSDP quantized storage is not validated; use single backend for QLoRA")
            if self.train.groups_per_update != 1 or self.train.update_epochs != 1:
                raise ValueError("The FSDP backend currently requires groups_per_update=update_epochs=1")
        update = r.update_devices or [self.model.device]
        devices = update + r.rollout_devices + ([self.model.reference_device or update[0]] if self.train.kl_coefficient else [])
        gpu = [d for d in devices if d.startswith("cuda")]
        # Legacy policy/reference sharing remains supported without a runtime topology.
        if (r.update_devices or r.rollout_devices) and len(gpu) != len(set(gpu)):
            raise ValueError("Runtime GPU roles must not overlap; reference_device may be cpu")
        if not self.model.name_or_path:
            raise ValueError("model.name_or_path must be a Hugging Face ID or local path")
        if self.model.mode not in {"full", "lora"}:
            raise ValueError("model.mode must be full or lora")
        if self.model.quantization not in {None, "4bit", "8bit"}:
            raise ValueError("quantization must be null, 4bit or 8bit")
        if self.model.quantization and self.model.mode != "lora":
            raise ValueError("Quantized training requires LoRA")
        if self.model.chat_format not in {"auto", "chat", "plain"}:
            raise ValueError("chat_format must be auto, chat or plain")
        if self.sql.reward_metric not in {"execution", "test_suite"}:
            raise ValueError("Unknown reward_metric")
        if self.sql.reward_metric == "test_suite" and not self.sql.suite_database_dir:
            raise ValueError("test_suite rewards require suite_database_dir")
        for name, value in {
            "max_steps": self.train.max_steps, "max_groups": self.train.max_groups,
            "groups_per_update": self.train.groups_per_update,
            "update_epochs": self.train.update_epochs, "save_steps": self.train.save_steps,
            "max_consecutive_amp_skips": self.train.max_consecutive_amp_skips,
            "sft_epochs": self.train.sft_epochs, "learning_rate": self.train.learning_rate,
            "max_context_tokens": self.rollout.max_context_tokens,
            "max_new_tokens": self.rollout.max_new_tokens,
            "temperature": self.rollout.temperature, "timeout_seconds": self.sql.timeout_seconds,
            "max_result_rows": self.sql.max_result_rows, "max_result_bytes": self.sql.max_result_bytes,
            "worker_memory_mb": self.sql.worker_memory_mb, "max_sql_chars": self.sql.max_sql_chars,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rollout.group_size < 2 or self.rollout.max_intermediate < 0:
            raise ValueError("group_size >= 2 and max_intermediate >= 0 are required")
        if self.sql.max_rows < 0 or self.sql.max_response_chars < 256:
            raise ValueError("max_rows >= 0 and max_response_chars >= 256 are required")
        if min(self.rollout.efficiency_beta, self.rollout.non_executable_penalty,
               self.train.kl_coefficient, self.train.weight_decay) < 0:
            raise ValueError("Reward weights, KL and weight decay must be nonnegative")
        if not 0 < self.train.clip_ratio < 1 or self.train.max_grad_norm <= 0:
            raise ValueError("Invalid clipping settings")

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


def load_config(path):
    raw = json.loads(Path(path).read_text())
    classes = {"model": ModelConfig, "data": DataConfig, "sql": SQLConfig,
               "rollout": RolloutConfig, "train": TrainConfig, "runtime": RuntimeConfig,
               "evaluation": EvaluationConfig, "tracking": TrackingConfig}
    extra = raw.keys() - classes.keys()
    if extra:
        raise ValueError(f"Unknown config sections: {extra}")
    config = Config(**{key: cls(**raw.get(key, {})) for key, cls in classes.items()})
    config.validate()
    return config
