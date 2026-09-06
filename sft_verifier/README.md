# Verifier SFT

The detailed design and acceptance criteria are in `PLAN.md`. Commands below are run from the repository root with the project `rl` conda environment.

```bash
export PYTHONPATH=sft_verifier/src:overall_pipeline/src
```

## End-to-end 실행

후보 수집부터 LoRA 학습, merged verifier와 multi-agent 설정 생성까지 한 번에 실행하려면 다음 명령을 사용한다. 기본값은 후보 수집에 물리 GPU 2,3을 사용하고 학습에 GPU 2–7을 사용한다.

```bash
bash sft_verifier/run_end_to_end.sh
```

중단 후 같은 명령을 다시 실행하면 후보 수집은 기존 JSONL을 기준으로 resume한다. 나머지 파생 산출물은 다시 생성한다. 새 실험을 분리하거나 설정을 바꾸려면 환경변수를 사용한다.

```bash
RUN_TAG=v2 \
SAMPLES_PER_QUESTION=2 \
EPOCHS=3 \
EFFECTIVE_BATCH_SIZE=24 \
bash sft_verifier/run_end_to_end.sh
```

기본 coder는 `sft/models/qwen2.5_0.5b_sql_coder`이다. 다른 checkpoint를 사용할 때는 `CODER_MODEL=/absolute/path`로 지정한다. 전체 train 후보 수집은 순차 생성이므로 오래 걸린다. 먼저 동작만 확인하고 싶다면 아래의 smoke-run 명령을 사용한다.

```bash
RUN_TAG=smoke MAX_EXAMPLES=10 SAMPLES_PER_QUESTION=2 EPOCHS=1 \
bash sft_verifier/run_end_to_end.sh
```

`MAX_EXAMPLES`는 train과 validation에서 각각 적용된다. Smoke run은 코드와 GPU 메모리 검증용이며, 만들어진 모델의 성능을 판단하는 용도로는 사용하지 않는다.

## 1. Prepare DB-disjoint splits

```bash
conda run -n rl python -m sft_verifier.prepare_splits \
  --output sft_verifier/data/manifests/spider_train_splits.json
```

## 2. Collect actual planner/coder candidates

Use the exact coder checkpoint that will be used in the multi-agent pipeline. The example below uses the existing merged SFT coder and two GPUs.

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run -n rl python -m sft_verifier.collect_candidates \
  --split-manifest sft_verifier/data/manifests/spider_train_splits.json \
  --split train \
  --planner-model Qwen/Qwen2.5-1.5B-Instruct \
  --planner-revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --planner-device cuda:0 \
  --coder-model sft/models/qwen2.5_0.5b_sql_coder \
  --coder-device cuda:1 \
  --samples-per-question 4 \
  --resume \
  --output sft_verifier/data/candidates/train.jsonl
```

Repeat with `--split validation` and `--output .../validation.jsonl`. Add `--allow-model-download` only if the pinned base model is not already cached.

## 3. Create hard negatives

```bash
conda run -n rl python -m sft_verifier.mutate_gold \
  --candidates sft_verifier/data/candidates/train.jsonl \
  --output sft_verifier/data/candidates/train_mutations.jsonl
```

Repeat for validation.

## 4. Label with the official test-suite evaluator

```bash
conda run -n rl python -m sft_verifier.label_candidates \
  --candidates \
    sft_verifier/data/candidates/train.jsonl \
    sft_verifier/data/candidates/train_mutations.jsonl \
  --output sft_verifier/data/private_labels/train.jsonl
```

The labeled file is private because it contains gold SQL. Do not pass it directly to training.

## 5. Build balanced, gold-free SFT files

```bash
conda run -n rl python -m sft_verifier.build_sft_dataset \
  --labeled \
    sft_verifier/data/private_labels/train.jsonl \
    sft_verifier/data/private_labels/validation.jsonl \
  --output-dir sft_verifier/data/sft/v1
```

## 6. Train LoRA and save a merged pipeline model

For GPUs 2 through 7:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 conda run -n rl torchrun \
  --standalone --nproc_per_node=6 \
  -m sft_verifier.train_lora \
  --train-file sft_verifier/data/sft/v1/train.jsonl \
  --validation-file sft_verifier/data/sft/v1/validation.jsonl \
  --output-dir sft_verifier/models/verifier_lora_v1 \
  --merge-output sft_verifier/models/verifier_merged_v1
```

The adapter is stored under `verifier_lora_v1/adapter`; the merged checkpoint is usable by the current multi-agent local-model backend.

## 7. Offline evaluation

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n rl python -m sft_verifier.evaluate_offline \
  --dataset sft_verifier/data/sft/v1/test.jsonl \
  --adapter sft_verifier/models/verifier_lora_v1/adapter \
  --device cuda:0 \
  --output-dir sft_verifier/outputs/offline_v1
```

Run the same command without `--adapter` into another output directory for the frozen-base comparison.

## 8. Generate a multi-agent config

```bash
conda run -n rl python -m sft_verifier.make_pipeline_config \
  --base-config overall_pipeline/configs/multi_turn_multi_agent_zero_shot.json \
  --merged-verifier sft_verifier/models/verifier_merged_v1 \
  --output sft_verifier/configs/multi_turn_sft_verifier_v1.json
```

Then run `python -m text2sql agent-evaluate` with the generated config and a fresh run name.
