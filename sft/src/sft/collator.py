from collections.abc import Mapping

import torch


class AnswerOnlyCollator:
    """Tokenize chat prompts and apply loss only to answer and EOS tokens."""

    def __init__(self, tokenizer, max_length, chat_template_kwargs=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chat_template_kwargs = chat_template_kwargs or {}

    def encode(self, example):
        if "_input_ids" in example:
            return list(example["_input_ids"]), list(example["_labels"])
        prompt = self.tokenizer.apply_chat_template(
            example["messages"], tokenize=True, add_generation_prompt=True,
            **self.chat_template_kwargs)
        if isinstance(prompt, Mapping):
            prompt = prompt["input_ids"]
        if hasattr(prompt, "ids"):
            prompt = prompt.ids
        response = self.tokenizer.encode(example["response"], add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            response = response + [self.tokenizer.eos_token_id]
        prompt = list(prompt)
        if len(prompt) + len(response) > self.max_length:
            raise SequenceTooLong(example.get("example_id", "example"),
                                  len(prompt), len(response), self.max_length)
        ids = prompt + response
        return ids, [-100] * len(prompt) + response

    def __call__(self, examples):
        encoded = [self.encode(example) for example in examples]
        width = max(len(ids) for ids, _ in encoded)
        pad = self.tokenizer.pad_token_id
        input_ids, labels, attention = [], [], []
        for ids, target in encoded:
            n = width - len(ids)
            input_ids.append(ids + [pad] * n)
            labels.append(target + [-100] * n)
            attention.append([1] * len(ids) + [0] * n)
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attention)}


class SequenceTooLong(ValueError):
    def __init__(self, example_id, prompt_tokens, response_tokens, max_length):
        self.record = {"example_id": example_id, "prompt_tokens": prompt_tokens,
                       "response_tokens": response_tokens, "max_length": max_length}
        super().__init__(f"Sequence exceeds max_length: {self.record}")


def pretokenize(examples, collator):
    accepted, skipped = [], []
    for example in examples:
        try:
            ids, labels = collator.encode(example)
        except SequenceTooLong as exc:
            skipped.append(exc.record)
            continue
        accepted.append({**example, "_input_ids": ids, "_labels": labels})
    if not accepted:
        raise ValueError("No examples fit within max_length")
    return accepted, skipped
