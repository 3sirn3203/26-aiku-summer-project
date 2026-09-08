from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from pathlib import Path
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StoppingCriteria, StoppingCriteriaList


def device_for(config):
    return config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")


def load_model(config, trainable=True, checkpoint=None):
    """No architecture whitelist; any Transformers causal LM with forward/generate works."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This trainer is single-process. Do not launch with torchrun/accelerate multi-process.")
    name = config.name_or_path
    adapter = config.adapter_name_or_path
    shared = {"trust_remote_code": config.trust_remote_code,
              "local_files_only": config.local_files_only}
    if config.revision:
        shared["revision"] = config.revision
    # Adapter-only SFT directories are accepted as the main model path too.
    if (Path(name) / "adapter_config.json").is_file():
        from peft import PeftConfig
        if adapter:
            raise ValueError("Specify an adapter once: name_or_path or adapter_name_or_path")
        adapter = name
        name = PeftConfig.from_pretrained(adapter, **shared).base_model_name_or_path
    tokenizer_source = config.tokenizer_name_or_path or config.name_or_path
    if adapter and not (Path(tokenizer_source) / "tokenizer_config.json").is_file() and Path(tokenizer_source).is_dir():
        tokenizer_source = name
    if checkpoint:
        if config.mode == "lora":
            adapter = str(checkpoint)
        else:
            name, adapter = str(checkpoint), None
        tokenizer_source = str(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **{**shared, **config.tokenizer_kwargs})
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs pad_token or eos_token; configure tokenizer_kwargs")
        tokenizer.pad_token = tokenizer.eos_token
    if config.chat_template:
        tokenizer.chat_template = Path(config.chat_template).read_text()
    device = device_for(config)
    kwargs = {**shared, **config.model_kwargs}
    if "device_map" in kwargs:
        raise ValueError("Use model.device; inference device_map offloading is not supported for training")
    if config.dtype != "auto":
        if config.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be auto, float32, float16 or bfloat16")
        kwargs.setdefault("torch_dtype", getattr(torch, config.dtype))
    else:
        kwargs.setdefault("torch_dtype", "auto")
    if config.quantization:
        from transformers import BitsAndBytesConfig
        if not str(device).startswith("cuda"):
            raise ValueError("This QLoRA loader requires a CUDA device")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=config.quantization == "4bit", load_in_8bit=config.quantization == "8bit",
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=kwargs.get("torch_dtype") if isinstance(kwargs.get("torch_dtype"), torch.dtype) else torch.float32,
        )
        kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if not config.quantization:
        model.to(device)
    if config.quantization and trainable:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable, **shared)
        if config.mode == "full":
            model = model.merge_and_unload()
    elif config.mode == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=config.lora_rank, lora_alpha=config.lora_alpha,
            lora_dropout=0.0, target_modules=config.target_modules,
        ))
    if config.mode == "full":
        model.requires_grad_(trainable)
        # Keep FP32 trainable master weights; forward uses autocast below.
        model.float()
    if trainable and config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if not trainable:
        model.requires_grad_(False)
    # Reproducible rollout/old/current likelihoods: eliminate dropout, including LoRA dropout.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        for name in ("attention_dropout", "hidden_dropout", "dropout", "attn_dropout"):
            if isinstance(getattr(module, name, None), float):
                setattr(module, name, 0.0)
    model.config.use_cache = False
    return model, tokenizer


def autocast_context(model, config):
    from contextlib import nullcontext
    device = next(model.parameters()).device
    dtype = getattr(torch, config.dtype, None)
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


@dataclass
class Turn:
    prompt_ids: list[int]
    action_ids: list[int]
    text: str
    old_log_probs: list[float] | None = None
    reference_log_probs: list[float] | None = None


def normalize_token_ids(encoded):
    """Convert tokenizer outputs from different Transformers versions to one ID list."""
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise TypeError("Tokenizer output does not contain input_ids")
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "ids"):
        encoded = encoded.ids
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu().tolist()
    elif hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if (isinstance(encoded, Sequence) and len(encoded) == 1
            and isinstance(encoded[0], Sequence) and not isinstance(encoded[0], (str, bytes))):
        encoded = encoded[0]
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes)):
        raise TypeError(f"Unsupported tokenizer output type: {type(encoded).__name__}")
    ids = list(encoded)
    if not all(isinstance(token, int) and not isinstance(token, bool) for token in ids):
        raise TypeError("Tokenizer output must be a single sequence of integer token IDs")
    return ids


def decode_action(tokenizer, ids):
    # Preserve action tags registered as special tokens by an SFT tokenizer.
    boundary_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id}
    return tokenizer.decode([int(i) for i in ids if int(i) not in boundary_ids], skip_special_tokens=False)


class EndAction(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer, self.prompt_length = tokenizer, prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        text = decode_action(self.tokenizer, input_ids[0, self.prompt_length:])
        return "</answer>" in text or "</intermediate>" in text


class HFPolicy:
    def __init__(self, model, tokenizer, model_config, rollout_config):
        self.model, self.tokenizer = model, tokenizer
        self.model_config, self.config = model_config, rollout_config

    def encode(self, messages):
        cfg = self.model_config
        use_chat = cfg.chat_format == "chat" or (cfg.chat_format == "auto" and self.tokenizer.chat_template)
        if use_chat:
            encoded = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, **cfg.chat_template_kwargs)
            return normalize_token_ids(encoded)
        # Deterministic plain-text fallback for base/SFT models without a template.
        prompt = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages) + "\n\nASSISTANT:\n"
        return normalize_token_ids(self.tokenizer.encode(prompt, add_special_tokens=True))

    def generate(self, messages, sample=True):
        ids = self.encode(messages)
        capacity = self.config.max_context_tokens
        positional = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(positional, int) and positional > 0:
            capacity = min(capacity, positional)
        remaining = capacity - len(ids)
        if remaining <= 0:
            return None
        device = self.model.get_input_embeddings().weight.device
        inputs = torch.tensor([ids], device=device)
        kwargs = dict(do_sample=sample, max_new_tokens=min(remaining, self.config.max_new_tokens),
                      pad_token_id=self.tokenizer.pad_token_id, num_beams=1,
                      repetition_penalty=1.0, no_repeat_ngram_size=0,
                      forced_bos_token_id=None, forced_eos_token_id=None,
                      suppress_tokens=None, begin_suppress_tokens=None,
                      use_cache=True, renormalize_logits=False,
                      stopping_criteria=StoppingCriteriaList([EndAction(self.tokenizer, len(ids))]))
        # Do not inherit checkpoint-specific top-k, bad-word or min-length processors:
        # old/current log probabilities must describe the actual sampling distribution.
        kwargs["generation_config"] = GenerationConfig(
            bos_token_id=self.model.generation_config.bos_token_id,
            eos_token_id=self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if sample:
            kwargs.update(temperature=self.config.temperature, top_k=0, top_p=1.0,
                          typical_p=1.0, min_p=None, epsilon_cutoff=0.0, eta_cutoff=0.0)
        self.model.eval()
        with torch.inference_mode(), autocast_context(self.model, self.model_config):
            output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), **kwargs)
        tail = output[0, len(ids):].tolist()
        return Turn(ids, tail, decode_action(self.tokenizer, tail))


def action_log_probs(model, turn, temperature, model_config):
    device = next((p.device for p in model.parameters() if p.numel()), torch.device("cpu"))
    ids = torch.tensor([turn.prompt_ids + turn.action_ids], device=device)
    start = len(turn.prompt_ids) - 1
    with autocast_context(model, model_config):
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    logits = logits[0, start:-1].float() / temperature
    targets = ids[0, start + 1:]
    # Only generated tokens receive a loss; prompt/schema/DB observations are masked by slicing.
    return logits.gather(-1, targets[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)
