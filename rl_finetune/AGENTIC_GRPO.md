# Two-turn workflow GRPO (v1)

This path is separate from `train_grpo_lora.py` and keeps the existing
single-turn trainer available without CLI changes.

Agentic v1 accepts only the pinned
`Qwen/Qwen2.5-Coder-0.5B-Instruct@ea3f2471cf1b1f0db85067f1ef93848e38e88c25`
base. Unverified SFT or general-Qwen checkpoints are rejected before a run is
created.

The rollout is fixed to two model actions:

1. Generate draft SQL from the existing single-turn prompt.
2. Execute the draft with the existing isolated SQLite executor.
3. Add the bounded execution observation to the conversation.
4. Generate final SQL with the same LoRA policy.
5. Apply the existing reward contract to final raw output only.
6. Apply the group-normalized terminal advantage to generated action tokens.

Gold SQL and gold execution results are reward metadata and are never passed to
the rollout runner. Gold execution failure skips the whole trajectory group.
The v1 reward always uses `latency_weight=0.0`.

Rollout sampling explicitly uses `top_p=1.0` and `top_k=0`. Draft/final
old-policy, current-policy, and reference-policy log-probabilities all use the
same configured temperature, so the GRPO objective matches the distribution
that generated the actions.

## Dataset dry-run

```bash
cd /home/aikusrv02/aiku/spider-env
PYTHONDONTWRITEBYTECODE=1 \
python -m rl_finetune.train_agentic_grpo_lora \
  --baseline-config overall_pipeline/configs/evaluate_dev.json \
  --examples-file train_spider.json \
  --split train \
  --output-dir rl_finetune/outputs \
  --run-name agentic-dataset-smoke-001 \
  --limit 8 \
  --dry-run-dataset
```

## Tiny GPU smoke

Run this only in the pinned RL environment described by the parent README.
One optimizer step includes four two-turn trajectories by default.

```bash
cd /home/aikusrv02/aiku/spider-env
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 \
python -m rl_finetune.train_agentic_grpo_lora \
  --baseline-config overall_pipeline/configs/evaluate_dev.json \
  --examples-file train_spider.json \
  --split train \
  --output-dir rl_finetune/outputs \
  --run-name agentic-grpo-smoke-001 \
  --limit 8 \
  --max-steps 1 \
  --num-generations 4 \
  --max-draft-tokens 256 \
  --max-final-tokens 256 \
  --credit-mode all_actions
```

The command fails rather than silently converting model/OOM/executor
infrastructure problems into rewards. On success it also unloads the training
model, reloads `adapter/` onto a fresh base model, and performs one complete
draft-execute-final inference.

Artifacts are `run_manifest.json`, `train_dataset.jsonl`,
`trajectory_trace.jsonl`, `reward_trace.jsonl`, `trainer/checkpoint-N/`, and
`adapter/`. A checkpoint contains adapter weights, optimizer, scheduler, RNG,
trainer state, the dataset cursor, and model/dataset/LoRA provenance. Resume
requires the same dataset selection, rollout settings, and trainer settings
except that `--max-steps` may be increased. Resume into a new, non-existing run
directory with:

```bash
python -m rl_finetune.train_agentic_grpo_lora \
  ... \
  --run-name agentic-grpo-resumed-001 \
  --max-steps 3 \
  --resume-from-checkpoint \
  rl_finetune/outputs/agentic-grpo-smoke-001/trainer/checkpoint-1
```

## Distributed dev evaluation

Omit `--adapter-dir` to run the frozen base-model control with the same
two-turn workflow.

```bash
python -m rl_finetune.evaluate_two_turn_base \
  --baseline-config overall_pipeline/configs/evaluate_dev.json \
  --adapter-dir rl_finetune/outputs/agentic-grpo-smoke-001/adapter \
  --gpus 0,1,2,3,4,5,6,7 \
  --selection all \
  --output-dir rl_finetune/outputs/evaluation \
  --run-name agentic-grpo-dev-all
```
