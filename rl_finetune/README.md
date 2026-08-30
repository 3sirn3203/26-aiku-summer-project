# RL Fine-Tuning Extension

이 디렉터리는 기존 single-turn / multi-turn zero-shot baseline을 손상시키지 않고
RL fine-tuning 실험을 추가하기 위한 격리 작업 공간이다. 기존 `single_turn/`,
`multi_turn_agent/`, `core/`의 동작 계약은 가능한 한 변경하지 않고, 필요한 공통
기능은 import해서 재사용한다.

## Scope

- Spider train split에서 RL fine-tuning용 rollout을 생성한다.
- LoRA/adapter 기반 fine-tuning을 우선 고려한다.
- Fine-tuned adapter는 기존 평가 파이프라인과 동일한 dev split metric으로 비교한다.
- 주요 비교 지표는 Spider Test Suite Accuracy, Original Exact Set Match,
  prediction `query_elapsed_ns`이다.
- Reward 설계와 학습 로그는 이 디렉터리 아래에서 추적한다.

## Reward Contract

Reward의 1순위는 SQL의 성공 여부다. 여기서 성공은 단순 실행 성공이 아니라
gold SQL과 같은 결과를 반환하는 것을 의미한다. 실행시간은 성공한 SQL들 사이의
부가 비교 기준으로만 사용한다.

기본 원칙은 다음과 같다.

- Parse 실패, unsafe SQL, 실행 실패는 모두 정답 SQL보다 낮은 reward를 받는다.
- 실행은 성공했지만 gold 결과와 다르면 정답 SQL보다 낮은 reward를 받는다.
- 정답 SQL은 아무리 느려도 실패한 SQL보다 높은 reward를 받는다.
- 실행시간 보너스/패널티는 정답 SQL에만 적용한다.
- Latency 관측값은 기존 pipeline과 같은 `predicted_execution.query_elapsed_ns`를 쓴다.
- Gold 실행 오류는 모델 실패가 아니라 infrastructure 문제로 보고 학습 샘플에서
  제외할 수 있게 `reward=None`으로 표현한다.

현재 기본 reward 형태:

```text
parse / unsafe failure: -1.0
prediction execution failure: -0.7
execution success but result mismatch: -0.2
correct result: 1.0 + latency_bonus

latency_bonus = 0.1 * clip(log((gold_ns + eps) / (pred_ns + eps)), -1, 1)
```

따라서 기본 설정에서는 가장 느린 정답 SQL도 `0.9` 이상이고, 가장 좋은 실패
SQL의 reward인 `-0.2`보다 항상 높다.

## Implementation Plan

1. Reward 계산기를 독립 모듈로 만든다. 완료.
2. Spider train prompt dataset builder를 추가한다. 완료.
3. TRL GRPO + PEFT LoRA 학습 스크립트를 추가한다. 완료.
4. 학습된 adapter를 기존 분산 평가에서 로드하는 PEFT backend를 추가한다. 완료.
5. Two-turn adapter/base를 평가하는 분산 runner와 trajectory artifact를 추가한다. 완료.
6. 서버에서 1-step 학습, checkpoint resume, 2-GPU smoke를 검증한다. 실행 필요.

## Files

- `rewards.py`: parse/execute/compare 결과를 correctness-first scalar reward로 변환한다.
- `dataset.py`: Spider examples를 TRL conversational `prompt` dataset row로 변환한다.
- `trl_reward.py`: TRL `GRPOTrainer`가 호출할 수 있는 reward function adapter다.
- `train_grpo_lora.py`: GRPO + LoRA 학습 entrypoint다. 기존 project CLI에는 연결하지 않는다.
- `adapter_backend.py`: 현재 pipeline PEFT backend의 compatibility import다.
- `evaluate_adapter.py`: fine-tuned adapter를 현재 분산 Spider 평가로 전달하는
  compatibility entrypoint다.
- `tests/test_rewards.py`: reward contract 단위 테스트다.

## Smoke Commands

아래 명령은 사용자가 직접 실행한다. Codex는 이 프로젝트에서 학습/검증 명령을
직접 실행하지 않는다.

학습 환경에는 TRL 계열 패키지가 필요하다. 기존 zero-shot 재현 환경과 RL 학습
환경을 분리하고, `base` conda 환경에는 설치하지 않는다.

현재 RL stack은 TITAN Xp에서 검증한 `torch==2.5.1+cu121` 계약에 맞춰 다음
세대를 고정한다.

- `transformers==4.46.3`
- `trl==0.14.0`
- `peft==0.14.0`
- `datasets>=2.21.0,<4`
- `accelerate>=0.34.0,<2`

새 conda 환경을 만든다.

```bash
cd /home/aikusrv02/aiku/spider-env
conda env create -f rl_finetune/environment-rl-finetune.yml
conda activate rl
```

서버 호환 PyTorch는 프로젝트 정책상 별도 설치한다. 현재 TITAN Xp 계약은
`torch==2.5.1` CUDA 12.1 wheel이다.

```bash
python -m pip install torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

그 다음 RL fine-tuning과 official evaluator dependency를 설치한다.

```bash
cd /home/aikusrv02/aiku/spider-env
python -m pip install -e './overall_pipeline[official-eval]'
python -m pip install -e ./rl_finetune
```

이미 최신 TRL을 설치했다면 torch 2.5.1과 맞지 않을 수 있으므로 강제로
재설치하지 말고, mismatched package만 제거한 뒤 다시 설치한다. `--force-reinstall`은
torch까지 다시 잡으려 할 수 있어 피한다.

```bash
conda activate rl
python -m pip uninstall -y transformers trl peft datasets accelerate
cd /home/aikusrv02/aiku/spider-env
python -m pip install -e './overall_pipeline[official-eval]'
python -m pip install -e ./rl_finetune
```

Official evaluator까지 같은 환경에서 돌릴 예정이면 NLTK tokenizer resource도
준비한다.

```bash
cd /home/aikusrv02/aiku/spider-env
python -m nltk.downloader -d data/nltk_data punkt punkt_tab
```

설치 확인:

```bash
cd /home/aikusrv02/aiku/spider-env
PYTHONDONTWRITEBYTECODE=1 \
python - <<'PY'
from rl_finetune.train_grpo_lora import _check_training_dependencies
print(_check_training_dependencies())
PY
```

먼저 새 모듈 import와 reward contract만 확인한다.

```bash
cd /home/aikusrv02/aiku/spider-env
PYTHONDONTWRITEBYTECODE=1 \
python -m unittest discover -s rl_finetune/tests -v
```

그다음 train prompt dataset 생성만 확인한다. 이 명령은 모델이나 TRL을 import하지
않고 Spider train file만 읽는다. 사용하는 Spider archive에 train file 이름이
다르면 `--examples-file`을 바꾼다.

```bash
cd /home/aikusrv02/aiku/spider-env
PYTHONDONTWRITEBYTECODE=1 \
python -m rl_finetune.train_grpo_lora \
  --baseline-config overall_pipeline/configs/evaluate_dev.json \
  --examples-file train_spider.json \
  --split train \
  --output-dir rl_finetune/outputs \
  --run-name dataset-smoke-001 \
  --limit 8 \
  --dry-run-dataset
```

학습 smoke run은 TRL/PEFT/datasets/accelerate/torch/transformers가 준비된 서버
환경에서만 실행한다. 처음에는 아주 작은 subset으로 checkpoint round-trip만 본다.

```bash
cd /home/aikusrv02/aiku/spider-env
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 \
python -m rl_finetune.train_grpo_lora \
  --baseline-config overall_pipeline/configs/evaluate_dev.json \
  --examples-file train_spider.json \
  --split train \
  --output-dir rl_finetune/outputs \
  --run-name grpo-lora-smoke-001 \
  --limit 16 \
  --learning-rate 5e-6 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-generations 4 \
  --temperature 0.9 \
  --reward-trace
```

학습이 끝난 뒤 adapter smoke evaluation을 돌린다. 이 명령은 모델을 로드하므로
사용자가 서버에서 직접 실행한다.

```bash
cd /home/aikusrv02/aiku/spider-env
PYTHONDONTWRITEBYTECODE=1 \
python -m rl_finetune.evaluate_adapter \
  --baseline-config overall_pipeline/configs/evaluate_dev.json \
  --adapter-dir rl_finetune/outputs/grpo-lora-smoke-001/adapter \
  --gpus 0,1 \
  --output-dir rl_finetune/outputs/evaluation \
  --run-name grpo-lora-smoke-001-dev-smoke \
  --selection smoke
```

같은 adapter를 전체 dev에서 평가하려면 GPU 목록과 selection만 바꾼다.

```bash
cd /home/aikusrv02/aiku/spider-env
PYTHONDONTWRITEBYTECODE=1 \
python -m text2sql evaluate \
  --config overall_pipeline/configs/evaluate_dev.json \
  --backend peft \
  --adapter-dir rl_finetune/outputs/grpo-lora-smoke-001/adapter \
  --gpus 0,1,2,3,4,5,6,7 \
  --output-dir rl_finetune/outputs/evaluation \
  --run-name grpo-lora-dev-all \
  --selection all
```

Agentic adapter의 two-turn 평가는 별도 분산 entrypoint를 사용한다.

```bash
python -m rl_finetune.evaluate_two_turn_base \
  --baseline-config overall_pipeline/configs/evaluate_dev.json \
  --adapter-dir rl_finetune/outputs/agentic-grpo-smoke-001/adapter \
  --gpus 0,1,2,3,4,5,6,7 \
  --output-dir rl_finetune/outputs/evaluation \
  --run-name agentic-grpo-dev-all \
  --selection all
```

## Progress Log

- 2026-08-21: `rl_finetune` 작업 공간을 만들고 correctness-first reward contract를
  문서화했다.
- 2026-08-21: 기존 baseline code path를 변경하지 않는 독립 reward module을 추가했다.
- 2026-08-21: Spider prompt dataset builder, TRL reward adapter, GRPO+LoRA 학습
  entrypoint 초안을 추가했다.
- 2026-08-30: PEFT adapter와 two-turn workflow를 현재 분산 평가, 공식 Spider
  metric, query timing, VM-step artifact 계약에 연결했다.
- 2026-08-21: 학습 환경에 `datasets`가 없어 `ModuleNotFoundError`가 발생했다.
  `rl-finetune` optional dependency extra와 학습 전 dependency preflight를 추가했다.
- 2026-08-21: `base` 환경 오염을 피하기 위해 conda environment file과 RL 전용
  requirements file을 추가했다. PyTorch CUDA wheel은 별도 설치 단계로 유지한다.
- 2026-08-21: `trl==1.10.0`이 `torch==2.5.1`에 없는 `FSDPModule`을 요구해
  `GRPOTrainer` import가 실패했다. RL dependency를 `trl==0.14.0`,
  `transformers==4.46.3`, `peft==0.14.0` 세대로 pinning하고 preflight에서
  version mismatch를 잡도록 했다.
