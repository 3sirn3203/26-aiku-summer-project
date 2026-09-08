from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rrcm_sql.model import HFPolicy


def load_zero_shot_policy(cfg, device):
    source = Path(cfg.model.name_or_path)
    if source.is_dir() and (source / "adapter_config.json").exists():
        raise ValueError("Zero-shot baseline cannot load a PEFT adapter")
    shared = {"trust_remote_code": cfg.model.trust_remote_code,
              "local_files_only": cfg.model.local_files_only}
    if cfg.model.revision:
        shared["revision"] = cfg.model.revision
    tokenizer_source = cfg.model.tokenizer_name_or_path or cfg.model.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, **{**shared, **cfg.model.tokenizer_kwargs})
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad or eos token")
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.model.chat_template:
        tokenizer.chat_template = Path(cfg.model.chat_template).read_text()
    kwargs = {**shared, **cfg.model.model_kwargs}
    if "device_map" in kwargs:
        raise ValueError("Configure devices through --devices")
    kwargs["torch_dtype"] = ("auto" if cfg.model.dtype == "auto"
                             else getattr(torch, cfg.model.dtype))
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, **kwargs)
    model.to(device)
    model.requires_grad_(False)
    model.eval()
    policy_cfg = cfg.model.policy_config(device)
    return HFPolicy(model, tokenizer, policy_cfg, cfg.generation.rollout_config())

