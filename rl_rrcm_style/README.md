# Adaptive Text-to-SQL: SFT + multi-step GRPO

`codex.md`의 두-action 에이전트를 학습하는 독립 패키지입니다. `overall_pipeline`의 모델/runner/config를 import하지 않습니다. 기본 예제에서만 이미 저장소에 있는 공개 Spider evaluator 경로를 사용하며, 별도 checkout 경로로 교체할 수 있습니다.

RRCM의 outcome-based adaptive retrieval을 SQL 실행으로 옮긴 구조입니다. 원 논문과 달리 자연어 reasoning은 출력하지 않고, 정답 trajectory에만 intermediate 효율성 보너스를 줍니다. `rollout.efficiency_beta=0`으로 outcome-only ablation을 실행할 수 있습니다.

## 설치

저장소 루트에서 실행합니다. 모든 JSON 설정의 상대 경로는 **실행 디렉터리 기준**입니다.

```bash
pip install -e './rl_rrcm_style[peft,spider]'
# QLoRA를 사용할 때만:
pip install -e './rl_rrcm_style[qlora,spider]'
```

특정 Transformers/모델 버전에 고정하지 않습니다. 새로운 backbone에 필요한 Transformers와 PEFT 버전은 사용 환경에서 설치하세요. 구현 검증 환경은 Python 3.11, PyTorch 2.5.1, Transformers 4.46.3, PEFT 0.14.0입니다.

설치 없이 기존 환경에서 사용하려면:

```bash
export PYTHONPATH="$PWD/rl_rrcm_style/src"
python -m rrcm_sql --help
```

## 데이터 준비

```bash
rrcm-sql prepare \
  --config rl_rrcm_style/configs/lora.json \
  --output rl_rrcm_style/prepared-official
```

예제 config는 `data.split_mode=official`입니다. `train_spider.json` 전체로 학습하고, `dev.json`으로 checkpoint와 하이퍼파라미터를 선택하며, `test.json`으로 최종 성능을 확인합니다. Train/dev는 `database/`, `tables.json`을 공유하고 test는 `test_database/`, `test_tables.json`을 사용합니다. `data.train_json`, `dev_json`, `test_json`, `test_database_dir`, `test_tables`로 경로를 변경할 수 있습니다. `train_others_json`은 기본 null이며 명시적으로 지정할 때만 학습에 추가됩니다.

`prepare --config`는 원본을 나누거나 복사하지 않고 split 간 DB 중복, schema와 SQLite 파일 존재를 검증해 SHA256/DB 목록/예제 수를 manifest에 저장합니다. 학습 시에도 다시 검증해 run의 `data_manifest.json`에 기록합니다. 현재 로컬 데이터는 train 7,000개/140 DB, dev 1,034개/20 DB, test 2,147개/40 DB입니다. 임의의 JSON의 공식 출처까지 인증하는 기능은 아닙니다.

기존 설정 호환을 위해 split_mode를 생략하면 `legacy_internal`로 해석합니다. 이전 `prepare --train-json ... --validation-fraction 0.1`과 내부 분할도 이 모드에서만 지원합니다. 새 실험에는 예제 official config를 사용하세요.

DB는 `{database_dir}/{db_id}/{db_id}.sqlite` 구조, schema는 Spider `tables.json` 형식입니다. Full schema는 원래 table/column 이름, type, PK, FK를 일정한 순서로 직렬화합니다. 초기 입력에 sample row나 gold SQL을 넣지 않습니다. Gold SQL은 RL reward 및 평가, 명시적 SFT warm-start의 target에서만 사용합니다.

## 모델 선택

`configs/lora.json`을 복사한 뒤 `model` 설정과 `train.output_dir`을 변경하세요. 예제 모델은 기본값일 뿐, 모델 이름/architecture whitelist는 없습니다. `AutoModelForCausalLM`의 `forward`와 `generate`를 지원하는 decoder-only text 모델을 대상으로 합니다.

| 모델 형태 | 설정 |
|---|---|
| Hugging Face 모델 | `name_or_path: "organization/model"` |
| SFT 후 전체 가중치를 저장한 디렉터리 | `name_or_path: "/path/to/sft-model"` |
| 로컬 PEFT adapter 디렉터리 | `name_or_path: "/path/to/sft-adapter"`; `adapter_config.json`에서 base 경로 확인 |
| base와 adapter를 별도로 지정 | `name_or_path: "/path/to/base"`, `adapter_name_or_path: "/path/or/hf-id/adapter"` |
| tokenizer가 다른 위치에 저장됨 | `tokenizer_name_or_path: "/path/to/tokenizer"` |
| Full fine-tuning | `mode: "full"` |
| LoRA | `mode: "lora"`, `target_modules: "all-linear"` 또는 module 목록/정규식 |
| QLoRA | `mode: "lora"`, `quantization: "4bit"` 또는 `"8bit"`, CUDA 필요 |

로컬 adapter로 시작하면 **기존 SFT adapter를 이어서 학습**합니다. Reference도 동일한 SFT adapter를 로드해서 고정합니다. `mode=full`이면 adapter를 base에 merge한 뒤 전체 가중치를 학습합니다. Adapter의 base 경로는 실제 SFT에 사용한 base를 가리켜야 합니다. Tokenizer vocab과 모델 embedding 크기를 변경했던 모델은 이에 맞게 저장된 전체 모델/embedding이 필요합니다.

추가 옵션:

- `revision`, `local_files_only`, `trust_remote_code` (기본 false).
- `model_kwargs`: `attn_implementation`, `low_cpu_mem_usage` 등 `from_pretrained` 옵션.
- `tokenizer_kwargs`: tokenizer loader 옵션.
- `chat_format`: `auto`는 저장된 chat template을 사용하고, 없으면 plain prompt로 fallback. `chat`/`plain`으로 강제 가능.
- `chat_template`: 사용자 Jinja 파일 경로. `chat_template_kwargs`는 `apply_chat_template`에 전달. 예: 지원되는 reasoning 모델에서 `{"enable_thinking": false}`. SQL action 외 reasoning 출력은 format failure이므로 모델의 template/모드를 확인하세요.
- `dtype`: `auto`, `float32`, `float16`, `bfloat16`. GPU가 BF16을 지원하는지 확인하세요. 예제는 호환성을 위해 FP32이며 메모리/속도에 맞춰 변경할 수 있습니다. Full tuning은 FP32 master weights와 선택한 autocast dtype을 사용합니다.
- `device`: `auto`, `cpu`, `cuda:0` 등. `reference_device`로 별도 GPU/CPU에 고정 reference를 둘 수 있습니다. CPU reference는 느릴 수 있습니다. Quantized reference도 CUDA가 필요합니다.

기본 updater는 policy 한 device에서 실행합니다. `runtime.rollout_devices`를 지정하면 GPU별 Hugging Face worker가 complete trajectory를 분산 생성하고, 매 update 뒤 policy version과 LoRA/full state를 동기화합니다. 단일 updater 구성에서 `runtime.validation_devices`에 updater/reference/rollout device를 함께 지정하면 기존 updater policy와 rollout worker를 재사용합니다. Reference GPU는 validation 동안 reference를 CPU로 옮긴 뒤 현재 policy replica를 임시로 올리며, validation 직후 replica를 해제하고 reference를 복구합니다. 임시 replica를 올릴 수 없으면 해당 GPU를 제외하고 validation을 계속합니다. `runtime.update_backend=fsdp`는 두 GPU에 parameter/gradient/optimizer state를 `FULL_SHARD`하며 별도 rollout device가 필요합니다. DDP와 inference용 `device_map=auto` 학습은 사용하지 않습니다. Reference는 별도 model copy이므로 KL을 사용하면 해당 가중치 메모리가 추가되고, `kl_coefficient=0`이면 로드하지 않습니다. QLoRA는 단일 updater에서 지원하며 FSDP 조합은 검증 전이라 거부합니다.

## 학습

```bash
rrcm-sql train --config rl_rrcm_style/configs/lora.json

# Qwen3-1.7B, trajectory 6개(자율 3 + 확률적 prompt 3), 단일 updater:
CUDA_VISIBLE_DEVICES=0,1 rrcm-sql train \
  --config rl_rrcm_style/configs/qwen3_1_7b_mixed.json

# updater GPU 1개 + reference 1개 + rollout worker GPU 2개:
CUDA_VISIBLE_DEVICES=0,1,2,3 rrcm-sql train \
  --config rl_rrcm_style/configs/qwen3_1_7b_mixed_rollout2.json

# updater를 GPU 2개에 FSDP, reference 1개, rollout worker 2개:
CUDA_VISIBLE_DEVICES=0,1,2,3,4 rrcm-sql train \
  --config rl_rrcm_style/configs/qwen3_1_7b_mixed_fsdp2.json

# 저장된 SFT 모델/adapter에서 시작:
rrcm-sql train --config rl_rrcm_style/configs/lora.json \
  --model /path/to/sft-checkpoint \
  --output rl_rrcm_style/runs/from-sft
```

미리 SFT하지 않은 모델에서 correct sample이 적다면, 동일 prompt와 gold `<answer>`를 사용하는 warm-start를 먼저 실행할 수 있습니다.

```bash
rrcm-sql sft --config rl_rrcm_style/configs/lora.json \
  --output rl_rrcm_style/runs/sft
rrcm-sql train --config rl_rrcm_style/configs/lora.json \
  --model rl_rrcm_style/runs/sft/sft-final \
  --output rl_rrcm_style/runs/rl-after-sft
```

SFT는 `train.sft_epochs` 및 `train.max_steps` 중 먼저 도달한 조건에서 종료합니다. 이 warm-start는 direct-answer만 학습하므로 intermediate 사용을 보장하지 않습니다. Rollout의 correct 비율과 direct-answer 비율을 함께 확인하세요.

학습 과정:

1. Train question을 sampling하고 동일 초기 prompt에서 `group_size`개 trajectory 생성.
2. `<intermediate>` SQL 실행 결과/오류를 user observation으로 누적. 최종 `<answer>` 실행 결과는 모델에게 돌려주지 않음.
3. Trajectory 최종 reward를 group 안에서 평균 0, population standard deviation 1로 정규화. All-equal group은 update에서 제외.
4. 전체 trajectory의 action token 수로 정규화한 token-level clipped GRPO loss + reference KL. 각 trajectory의 advantage를 모든 생성 action token에 적용.
5. 한 turn씩 backward하여 group 전체의 activation graph를 동시에 유지하지 않음.

Mixed 설정에서는 group 6개 중 앞의 3개가 모델의 자율 action이고 나머지 3개가 `prompted_random`입니다. 확률적 trajectory는 매 turn `answer_probability`로 `<answer>` 지시를 고르고, 나머지는 `<intermediate>`를 지시합니다. intermediate를 3회 실행한 뒤에는 `<answer>`를 지시합니다. 지시는 해당 turn의 마지막 user message에만 추가되고, 모델이 지시를 어기면 재생성하지 않고 `action_instruction_violation`으로 기록합니다. 평가는 항상 자율 action만 사용합니다.

`runtime.update_devices`와 `model.device`는 단일 updater에서 같아야 합니다. 장치 번호는 `CUDA_VISIBLE_DEVICES` 적용 후의 논리 번호입니다. 명시적인 runtime topology에서는 updater/reference/rollout GPU를 겹쳐 지정할 수 없습니다. FSDP backend는 현재 `groups_per_update=1`, `update_epochs=1`, 두 update GPU를 요구합니다. 각 rank가 같은 trajectory로 collective에 참여하므로 업데이트 메모리는 shard되지만 계산량은 데이터 병렬처럼 나뉘지 않습니다. FSDP도 activation과 vocabulary logits는 각 GPU에 남으므로 12GB OOM 제거를 보장하지 않습니다.

단일 updater에서 OOM이 발생하면 오류 메시지가 FSDP 재실행 경로를 안내하고, update 단계의 allocated/reserved/peak VRAM을 `train_metrics.jsonl`에 기록합니다. FSDP config로 새 run을 시작하며 기존 complete single-GPU checkpoint를 모델 warm-start로 사용하려면 `--model /path/to/checkpoint --output /new/output`을 함께 지정합니다. optimizer까지 이어가는 exact resume는 backend가 같은 checkpoint에서만 허용됩니다. 프로세스 내부 자동 전환은 부분 gradient·optimizer 상태의 안전한 복구를 보장하기 어려워 제공하지 않습니다.

Prompt/schema/DB response token은 **loss에서 제외**합니다. Sampling에 사용한 temperature를 old/current/reference log-probability에도 동일하게 적용합니다. Top-k/top-p 및 checkpoint의 generation processors를 초기화하여 likelihood와 실제 sampling 분포를 맞춥니다. Dropout을 꺼서 재계산한 old log probability가 일관되게 유지되도록 합니다. `groups_per_update`는 유효 group 누적 수이고, `update_epochs`는 같은 rollout batch의 재사용 횟수입니다. `max_steps`는 optimizer update 횟수, `max_groups`는 all-equal을 포함한 sampling group 예산입니다. 예산 소진 시 미완료 상태를 `status.json`에 명시하고 현재 checkpoint를 저장합니다.

RL의 FP16 AMP에서 nonfinite gradient가 검출되면 `GradScaler`가 해당 optimizer update를 건너뛰고 loss scale을 낮춥니다. 실제 적용된 update만 `step`에 포함하고, skip 여부와 scale 전후 값 및 누적/연속 횟수를 `train_metrics.jsonl`에 기록합니다. `max_consecutive_amp_skips`회 연속 skip되면 지속적인 수치 문제로 판단해 중단하며 기본값은 5입니다. 이 정책은 RL `train` 명령에만 적용되고 SFT optimizer 처리는 변경하지 않습니다.

Reward는 execution 정답일 때 `1 + alpha * I[exact_match] - beta * m/N_max`, 실행 가능한 오답 `0`, 실행 불가능/format failure `-lambda`입니다. 기본 `alpha=1.5`이며 설정 키는 `exact_match_alpha`, `efficiency_beta`, `non_executable_penalty`입니다. Exact-match reward를 사용하려면 `sql.evaluator_path`가 필요합니다. `N_max=0`이면 intermediate penalty는 0입니다. 실패한 intermediate도 횟수에 포함됩니다. 한도를 넘은 intermediate는 실행하지 않고 실패 처리합니다. Context가 가득 차면 schema/trajectory를 잘라내지 않고 해당 trajectory를 `context_limit` 실패로 처리합니다.

## SQL 실행 및 정답 판정

SQL은 별도 CPU subprocess에서 실행합니다. SQLite `mode=ro`, `query_only`, authorizer allowlist, extension 차단, statement/길이 검사, progress timeout과 parent process hard timeout을 적용합니다. DDL/DML/ATTACH/PRAGMA는 action으로 실행할 수 없습니다. Observation에는 column 이름, 제한된 row, 전체 row 수를 포함하며 `<`/`>`를 escape합니다. 전체 row 수를 얻기 위한 scan 자체가 timeout에 걸릴 수도 있습니다.

`sql.evaluator_path`에 공개 [Spider test-suite evaluator](https://github.com/taoyds/test-suite-sql-eval) checkout을 지정합니다. 예제는 저장소의 기존 vendor 경로를 사용하지만 경로를 다른 checkout으로 바꾸면 `overall_pipeline` 없이 사용할 수 있습니다.

- `reward_metric=execution`: 원본 DB에서 전체 결과를 비교. `exec_eval.result_eq`로 column permutation 및 duplicate multiplicity를 처리. Spider와 같은 `order by` substring 규칙으로 row order 민감도를 결정.
- `reward_metric=test_suite`: `suite_database_dir`의 DB 변형 전체를 추가 비교. `{suite_database_dir}/{db_id}/*.sqlite`에 **2개 이상**의 generated suite DB가 있어야 하며 원본 DB만으로 Test-Suite Accuracy라고 보고하지 않음.
- `evaluator_path=null`: 의존성 최소화를 위한 strict execution proxy. Column 순서까지 같아야 하므로 Spider official 지표와 다릅니다. 실험 결과에 backend 이름을 기록하며 공식 성능 보고에는 evaluator를 설정하세요.

비교에는 observation 미리보기를 사용하지 않습니다. 예측 SQL의 full result가 `max_result_rows`/`max_result_bytes`를 넘으면 해당 샘플을 `non_executable`로 기록하고 계속 진행하며, `final_execution.kind`에 `result_limit`을 남깁니다. Gold 결과의 한도 초과나 gold/evaluator 실패는 인프라 오류로 중단합니다. Test-suite는 안전한 executor와 공식 `result_eq`를 결합하고, value plugging/DISTINCT 제거와 같은 SQL 보정은 하지 않습니다.

## 평가

RL/SFT 예제 config는 시작 전, 실제 optimizer update 25회마다, 종료 시 dev 전체를 평가합니다. `evaluation.enabled`, `every_steps`, `at_start`, `at_end`, `limit`으로 조절합니다. Greedy/free inference이며 평가 동안 RNG와 모델 모드를 보존합니다. Rollout pool이 있으면 최신 policy를 worker에 동기화한 뒤 평가합니다. FSDP는 rank 0이 평가를 조율하고 결과/오류를 전체 rank에 전달합니다. `evaluation.timeout_seconds`(기본 86,400초)는 평가 예산 및 분산 대기 timeout입니다. 단일 policy 평가 timeout은 질문 사이에서 확인합니다.

`evaluation.selection_metric` 기본값은 `execution_accuracy`입니다. 동점은 이전 best를 유지하고 step 0 baseline도 후보에 포함합니다. 개선된 모델은 `best-step-N/`에 저장하고 `best_checkpoint.json`에서 경로/점수/step을 찾습니다. `dev_metrics.jsonl`과 `evaluations/dev-step-N/`에 평가 이력을 남깁니다. 종료 step을 이미 평가했다면 반복 평가하지 않습니다. `limit`은 seed로 고정 추출한 subset 크기이며 기본 null은 전체 dev입니다. Subset 결과는 `is_subset=true`로 표시됩니다.

```bash
# Dev validation (checkpoint의 run_config.json 사용 권장)
rrcm-sql evaluate --config rl_rrcm_style/configs/lora.json \
  --split dev --checkpoint /path/to/best-step-N \
  --output rl_rrcm_style/runs/validation

# GPU별 model replica로 dev 샘플을 동적 분산
CUDA_VISIBLE_DEVICES=0,1,2,3 rrcm-sql evaluate \
  --config /path/to/checkpoint/run_config.json \
  --split dev --checkpoint /path/to/checkpoint \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
  --output rl_rrcm_style/runs/validation-4gpu

# 설정과 checkpoint를 dev로 선택한 뒤 전체 test 평가
rrcm-sql evaluate --config rl_rrcm_style/configs/lora.json \
  --split test --checkpoint /path/to/best-step-N \
  --output rl_rrcm_style/runs/test-final
```

Checkpoint 실제 경로는 학습 결과의 `status.json`을 확인하세요. 평가에는 checkpoint 학습 시 저장된 `run_config.json`을 사용하면 model/prompt/reward 설정을 그대로 재현할 수 있습니다. Greedy decoding을 사용하고 seed를 고정합니다. 같은 환경에서의 재현을 목표로 하며 CUDA kernel/라이브러리/하드웨어 변경 간 bitwise 재현까지 보장하지 않습니다.

`--devices`를 지정하면 장치마다 동일 checkpoint를 한 번씩 로드하고, 완료된 worker에 다음 샘플을 보내 동적으로 분산합니다. `CUDA_VISIBLE_DEVICES`를 사용한 경우 `cuda:N`은 그 안의 논리 번호입니다. 최종 trajectory와 prediction은 원래 평가 데이터 순서로 저장하며 metric은 전체 샘플에서 한 번 계산합니다. 옵션을 생략하면 기존 단일-device 평가를 사용합니다.

`evaluation.dev_suite_database_dir` 또는 `test_suite_database_dir`을 설정하면 해당 split에서 execution과 test-suite 정확도를 모두 계산합니다. 미설정 시 test-suite 값은 `null`입니다. Test용 원본 `test_database/`와 generated test-suite DB는 별개입니다. 학습 reward용 suite 경로는 기존 `sql.suite_database_dir`입니다. `evaluator_path`가 있으면 공개 evaluator로 structural Exact Match와 난이도를 계산해 `difficulty.csv`로 저장합니다. `sql.nltk_data`는 NLTK tokenizer 데이터 경로이며 예제는 저장소의 `data/nltk_data`를 사용합니다. 평가 파일의 gold는 trajectory 생성 prompt에 전달하지 않습니다.

출력:

- 학습: `trajectories.jsonl`, `rollout_metrics.jsonl`, `train_metrics.jsonl`, `runtime.json`, config, checkpoint, `status.json`.
- 평가: trajectory, `summary.json`, `summary.csv`, 공식 evaluator 사용 시 `difficulty.csv`/`structural_metrics.json`, 외부 평가용 `predictions.sql`/`gold.sql`.
- Trajectory: 모든 prompt/action token ID, question/schema, exploration mode와 requested/actual action, intermediate SQL/결과/오류, final SQL/outcome/reward, token 수 및 실행 시간. Worker 내부 old/reference log probability는 학습에 전달하지만 JSON에는 중복 저장하지 않습니다.

## 재시작

Official split로 전환하면 학습 데이터가 달라지므로 기존 internal-split checkpoint에서 exact resume할 수 없습니다. `--model /path/to/checkpoint`로 가중치를 가져와 새 실험을 시작하세요. 새 checkpoint는 마지막 dev 평가 step, best 점수/경로, W&B run ID도 저장합니다. 이전 best 디렉터리는 재시작 후에도 유지해야 합니다.

```bash
rrcm-sql train --config /path/to/checkpoint/run_config.json \
  --resume /path/to/checkpoint \
  --output rl_rrcm_style/runs/resumed
```

Checkpoint에 모델/adapter, tokenizer, optimizer, AMP scaler, Python/PyTorch/CUDA RNG, update/group counters, dataset/schema hash를 저장합니다. `complete.json`이 없는 중단된 저장은 재시작하지 않습니다. Resume에서는 output directory, max steps/groups, save interval과 tracking 설정만 변경할 수 있습니다. 평가 설정과 데이터는 고정합니다. Reference는 최초 설정의 모델/SFT adapter를 다시 로드하므로 원본 경로를 유지하세요. HF 모델의 장기 재현에는 commit revision을 고정하세요. `training_state.pt`는 자신의 신뢰할 수 있는 checkpoint만 로드해야 합니다.

## W&B 실험 관리

`pip install -e 'rl_rrcm_style[wandb]'`로 선택 의존성을 설치하고 `wandb login` 또는 환경변수 `WANDB_API_KEY`로 인증합니다. 예제 config의 `tracking.backend`를 `wandb`로 바꾸세요. 기본 `none`은 기존 로컬 기록만 사용합니다.

```json
"tracking": {
  "backend": "wandb", "project": "rl-rrcm-style", "entity": null,
  "run_name": "qwen3-mixed-seed42", "group": "qwen3-mixed",
  "tags": ["grpo", "spider"], "mode": "online",
  "log_trajectories": false, "upload_checkpoints": false,
  "resume_run": false
}
```

`train/*`, `dev/*`는 `optimizer_step`, `rollout/*`는 `group_step`을 축으로 기록합니다. Config, 데이터 manifest, git revision/dirty 여부, best 점수와 경로를 함께 남깁니다. 같은 비교 실험은 group, 모델/설정은 tags, 각 seed는 별도 run으로 관리하세요. 자동 Sweeps 실행은 포함하지 않습니다. FSDP는 rank 0만 W&B를 기록합니다.

`mode=offline`이면 로컬 W&B 파일을 만든 뒤 `wandb sync <offline-run-directory>`로 나중에 업로드할 수 있습니다. `mode=disabled`는 W&B 초기화를 생략합니다. 온라인 모드의 초기화 오류는 명시적으로 실패하므로 오프라인 서버에서는 offline을 선택하세요. API key는 config에 넣지 않습니다.

Resume는 기본적으로 이전 run ID를 parent로 기록한 새 run을 만듭니다. 동일 online run을 이어 기록하려면 `resume_run=true`를 지정합니다. W&B에 기록된 최신 optimizer step보다 오래된 checkpoint이면 분기 run을 요구합니다. Offline resume도 별도 run으로 연결합니다. Test 평가는 별도 `test-evaluation` run에 학습 run ID와 checkpoint 경로를 기록합니다.

`upload_checkpoints=true`이면 best/final 모델을 artifact로 업로드합니다(optimizer pickle 제외). `log_trajectories=true`는 SQL·질문·실행 결과가 포함된 평가 파일 업로드를 활성화합니다. 기본은 두 옵션 모두 false입니다.

## 검증

```bash
PYTHONPATH=rl_rrcm_style/src python -m unittest discover -s rl_rrcm_style/tests -v
```

SQL 경계 사례, timeout/읽기 전용/결과 truncation/오류 복구, split 누수, reward, action masking, GRPO clipping/gradient update, all-equal skip, checkpoint resume 일치, 로컬 SFT adapter 재학습, GPT-2/Llama 로더를 CPU에서 검증합니다. Tiny 모델은 테스트 중 로컬에 생성하므로 모델 다운로드가 필요 없습니다. 학습 update 테스트는 통제된 SQL action을 제공하여 positive/negative reward를 만들고 실제 Transformer backward/optimizer/checkpoint를 검증합니다. 별도 테스트가 실제 HF sampling 경로도 실행합니다.

## 파일 구성

`config.py` 설정, `data.py` Spider/split/schema, `sql.py`/`sql_worker.py` 안전한 실행과 reward, `model.py` 범용 HF/PEFT 로딩과 token likelihood, `exploration.py` 혼합 action 지시, `rollout.py` action loop, `runtime/rollout_pool.py` 분산 생성, `runtime/fsdp.py` sharded update, `train.py` SFT/GRPO 조율, `evaluate.py`/`spider_metrics.py` 평가, `cli.py` 실행 명령입니다.

참고: [RRCM 논문](https://arxiv.org/abs/2605.07129), [HF chat templates](https://huggingface.co/docs/transformers/chat_templating), [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization).
