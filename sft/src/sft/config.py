from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen3-1.7B"
    tokenizer_name_or_path: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = True
    dtype: str = "float16"
    attn_implementation: str | None = "sdpa"
    chat_template: str | None = None
    chat_template_kwargs: dict = field(default_factory=lambda: {"enable_thinking": False})


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: str | list[str] = "all-linear"
    bias: str = "none"


@dataclass
class DataConfig:
    train_json: str = "data/spider_data/train_spider.json"
    dev_json: str = "data/spider_data/dev.json"
    tables: str = "data/spider_data/tables.json"
    database_dir: str = "data/spider_data/database"


@dataclass
class TrainingConfig:
    output_dir: str = "sft/outputs/qwen3_sft"
    seed: int = 42
    epochs: float = 1.0
    max_steps: int = -1
    max_length: int = 4096
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    effective_batch_size: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    logging_steps: int = 20
    save_steps: int = 1000
    save_total_limit: int = 2
    resume_from_checkpoint: str | None = None


@dataclass
class ValidationConfig:
    enabled: bool = True
    strategy: str = "epoch"
    steps: int | None = None
    batch_size: int = 1
    metric: str = "eval_loss"
    generation_at_end: bool = True
    generation_limit: int | None = None
    max_new_tokens: int = 256
    evaluator_path: str | None = "overall_pipeline/vendor/spider_test_suite_eval"
    nltk_data: str | None = "data/nltk_data"
    dev_suite_database_dir: str | None = None


@dataclass
class RuntimeConfig:
    device: str = "cuda:0"
    expected_world_size: int = 1
    generation_devices: list[str] = field(default_factory=list)
    report_to: str = "none"
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    max_vram_fraction: float = 0.92
    min_free_vram_mb: int = 1000
    min_free_ram_gb: float = 8.0


@dataclass
class ExportConfig:
    save_adapter: bool = True
    merge: bool = True
    merged_subdir: str = "merged"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def validate(self, check_paths=True):
        if not self.model.name_or_path:
            raise ValueError("model.name_or_path is required")
        if self.model.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("model.dtype must be float16, bfloat16 or float32")
        if self.model.chat_template_kwargs.get("enable_thinking", False):
            raise ValueError("SFT uses answer-only output; enable_thinking must be false")
        if self.lora.rank <= 0 or self.lora.alpha <= 0 or not 0 <= self.lora.dropout < 1:
            raise ValueError("Invalid LoRA rank, alpha or dropout")
        if self.lora.bias not in {"none", "all", "lora_only"}:
            raise ValueError("Invalid LoRA bias")
        t, v = self.training, self.validation
        for name, value in {"epochs": t.epochs, "max_length": t.max_length,
                            "per_device_batch_size": t.per_device_batch_size,
                            "gradient_accumulation_steps": t.gradient_accumulation_steps,
                            "learning_rate": t.learning_rate, "save_steps": t.save_steps,
                            "logging_steps": t.logging_steps, "validation.batch_size": v.batch_size}.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if t.max_steps == 0 or t.max_steps < -1:
            raise ValueError("max_steps must be -1 or positive")
        actual_batch_size = (t.per_device_batch_size * t.gradient_accumulation_steps
                             * self.runtime.expected_world_size)
        if t.effective_batch_size != actual_batch_size:
            raise ValueError(
                f"training.effective_batch_size={t.effective_batch_size}, but per-device "
                f"batch × accumulation × world size is {actual_batch_size}")
        if v.strategy not in {"no", "steps", "epoch"}:
            raise ValueError("validation.strategy must be no, steps or epoch")
        if v.metric != "eval_loss":
            raise ValueError("Only eval_loss checkpoint selection is supported during SFT")
        if v.strategy == "steps" and (v.steps is None or v.steps <= 0):
            raise ValueError("validation.steps must be positive for steps strategy")
        if v.generation_limit is not None and v.generation_limit <= 0:
            raise ValueError("validation.generation_limit must be positive")
        if v.max_new_tokens <= 0:
            raise ValueError("validation.max_new_tokens must be positive")
        if not 0 < self.runtime.max_vram_fraction <= 1:
            raise ValueError("runtime.max_vram_fraction must be in (0, 1]")
        if self.runtime.min_free_vram_mb < 0 or self.runtime.min_free_ram_gb < 0:
            raise ValueError("Resource safety thresholds must be nonnegative")
        if self.runtime.expected_world_size <= 0:
            raise ValueError("runtime.expected_world_size must be positive")
        devices = self.runtime.generation_devices
        if len(devices) != len(set(devices)):
            raise ValueError("runtime.generation_devices must be unique")
        for device in devices:
            if not device.startswith("cuda:") or not device[5:].isdigit():
                raise ValueError(f"Invalid generation device: {device}")
        if self.runtime.report_to == "wandb" and not self.runtime.wandb_project:
            raise ValueError("runtime.wandb_project is required when report_to is wandb")
        if not self.export.save_adapter:
            raise ValueError("export.save_adapter must remain true for reusable LoRA output")
        if check_paths:
            for path in (self.data.train_json, self.data.dev_json, self.data.tables):
                if not Path(path).exists():
                    raise FileNotFoundError(path)
            if self.model.chat_template and not Path(self.model.chat_template).is_file():
                raise FileNotFoundError(self.model.chat_template)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


def load_config(path, check_paths=True):
    source = Path(path).resolve()
    raw = json.loads(source.read_text())
    classes = {"model": ModelConfig, "lora": LoRAConfig, "data": DataConfig,
               "training": TrainingConfig, "validation": ValidationConfig,
               "runtime": RuntimeConfig, "export": ExportConfig}
    if raw.keys() - classes.keys():
        raise ValueError(f"Unknown config sections: {raw.keys() - classes.keys()}")
    cfg = Config(**{name: cls(**raw.get(name, {})) for name, cls in classes.items()})
    if check_paths:
        root = next((directory for directory in (source.parent, *source.parents)
                     if (directory / "pyproject.toml").is_file()), None)
        if root is None:
            raise FileNotFoundError(
                f"Could not find project pyproject.toml above {source.parent}")
        def resolve(value):
            if value is None:
                return None
            candidate = Path(value).expanduser()
            return str(candidate if candidate.is_absolute() else (root / candidate).resolve())
        for name in ("train_json", "dev_json", "tables", "database_dir"):
            setattr(cfg.data, name, resolve(getattr(cfg.data, name)))
        cfg.training.output_dir = resolve(cfg.training.output_dir)
        cfg.training.resume_from_checkpoint = resolve(cfg.training.resume_from_checkpoint)
        cfg.validation.evaluator_path = resolve(cfg.validation.evaluator_path)
        cfg.validation.nltk_data = resolve(cfg.validation.nltk_data)
        cfg.validation.dev_suite_database_dir = resolve(cfg.validation.dev_suite_database_dir)
        if cfg.model.chat_template:
            cfg.model.chat_template = resolve(cfg.model.chat_template)
    cfg.validate(check_paths=check_paths)
    return cfg
