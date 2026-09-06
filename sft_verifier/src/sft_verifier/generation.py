from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


class ModelGenerator:
    """Small lazy Transformers wrapper used only by data/evaluation CLIs."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: Optional[str],
        device: str,
        adapter: Optional[Path] = None,
        allow_download: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        source = Path(model_id).expanduser()
        local = source.is_dir()
        resolved_id = str(source.resolve()) if local else model_id
        kwargs: Dict[str, Any] = {
            "trust_remote_code": False,
            "local_files_only": local or not allow_download,
        }
        if revision and not local:
            kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_id, **kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved_id,
            torch_dtype=torch.float16,
            attn_implementation="eager",
            **kwargs,
        )
        if adapter is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model, str(adapter.resolve()), is_trainable=False
            )
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.torch = torch
        self.model_id = model_id
        self.revision = revision or "local"
        self.adapter = str(adapter.resolve()) if adapter is not None else None

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int = 0,
    ) -> Dict[str, Any]:
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        encoded = self.tokenizer.apply_chat_template(
            list(messages),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_tokens = int(encoded["input_ids"].shape[-1])
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            kwargs.update({"temperature": temperature, "top_p": top_p})
        with self.torch.inference_mode():
            output = self.model.generate(**encoded, **kwargs)
        generated = output[0, input_tokens:]
        raw_output = self.tokenizer.decode(generated, skip_special_tokens=True)
        rendered = self.tokenizer.apply_chat_template(
            list(messages), add_generation_prompt=True, tokenize=False
        )
        return {
            "raw_output": raw_output,
            "input_tokens": input_tokens,
            "output_tokens": int(generated.shape[-1]),
            "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "model_id": self.model_id,
            "revision": self.revision,
            "adapter": self.adapter,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
            "seed": seed,
        }

