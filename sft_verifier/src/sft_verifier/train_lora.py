from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from . import BASE_MODEL_ID, BASE_MODEL_REVISION
from .common import read_jsonl, write_json


class VerifierCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        import torch

        prompt_texts = [
            self.tokenizer.apply_chat_template(
                list(record["messages"]), add_generation_prompt=True, tokenize=False
            )
            for record in records
        ]
        full_texts = [
            self.tokenizer.apply_chat_template(
                list(record["messages"])
                + [{"role": "assistant", "content": str(record["response"])}],
                add_generation_prompt=False,
                tokenize=False,
            )
            for record in records
        ]
        batch = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        labels = batch["input_ids"].clone()
        for index, prompt in enumerate(prompt_texts):
            prompt_ids = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )["input_ids"]
            prompt_length = min(len(prompt_ids), labels.shape[1])
            labels[index, :prompt_length] = -100
            if torch.all(labels[index] == -100):
                raise ValueError(
                    "example %s has no response tokens after truncation"
                    % records[index].get("example_id")
                )
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


class RecordsDataset:
    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the verifier with assistant-only LoRA SFT")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--base-revision", default=BASE_MODEL_REVISION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merge-output", type=Path)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--effective-batch-size", type=int, default=16)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    train_records = read_jsonl(args.train_file)
    validation_records = read_jsonl(args.validation_file)
    if not train_records or not validation_records:
        raise ValueError("train and validation datasets must both be non-empty")
    local_model = Path(args.base_model).expanduser().is_dir()
    model_id = str(Path(args.base_model).resolve()) if local_model else args.base_model
    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": False,
        "local_files_only": local_model or not args.allow_model_download,
    }
    if not local_model:
        load_kwargs["revision"] = args.base_revision
    tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        **load_kwargs,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    denominator = world_size * args.per_device_batch_size
    gradient_accumulation = max(1, (args.effective_batch_size + denominator - 1) // denominator)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        bf16=False,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=RecordsDataset(train_records),
        eval_dataset=RecordsDataset(validation_records),
        data_collator=VerifierCollator(tokenizer, args.max_length),
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        adapter_dir = args.output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        write_json(
            adapter_dir / "text2sql_adapter_provenance.json",
            {
                "base_model_id": args.base_model,
                "base_model_revision": args.base_revision if not local_model else "local",
                "task": "verifier_sft",
                "train_file": str(args.train_file.resolve()),
                "validation_file": str(args.validation_file.resolve()),
                "seed": args.seed,
            },
        )
        write_json(
            args.output_dir / "training_manifest.json",
            {
                "base_model": args.base_model,
                "base_revision": args.base_revision,
                "train_examples": len(train_records),
                "validation_examples": len(validation_records),
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length,
                "effective_batch_size_requested": args.effective_batch_size,
                "effective_batch_size_actual": denominator * gradient_accumulation,
                "adapter": str(adapter_dir.resolve()),
            },
        )
        if args.merge_output is not None:
            merged = trainer.model.merge_and_unload()
            args.merge_output.mkdir(parents=True, exist_ok=True)
            merged.save_pretrained(args.merge_output, safe_serialization=True)
            tokenizer.save_pretrained(args.merge_output)
        print("saved verifier adapter to %s" % adapter_dir)


if __name__ == "__main__":
    main()
