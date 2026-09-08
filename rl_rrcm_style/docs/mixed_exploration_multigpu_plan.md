# 혼합 탐색 rollout 및 선택적 GPU 분산 구현 계획

작성일: 2026-09-06. 상태: 구현 전 계획. 이 문서만 추가하며 현재 실행 코드와 학습 설정은 변경하지 않는다.

## 1. 범위와 기본값

- 모든 학습 구현은 `rl_rrcm_style/src/rrcm_sql/`에 둔다. SFT 경로는 변경하지 않는다.
- 한 question/schema에 대해 trajectory 6개: 자율 선택 3개, 확률적 프롬프트 지시 3개.
- 확률적 trajectory는 매 turn마다 action을 다시 샘플링한다. 기본 `answer_probability=0.5`로 제안하며 설정 가능하게 한다. 앞서 언급한 answer:intermediate=2:1은 `2/3`으로 설정한다.
- intermediate 최대 3회. 성공/실패와 관계없이 실제 intermediate 실행 시도마다 횟수를 센다. 3회 이후 다음 turn에는 answer 지시만 넣는다. answer가 나오면 즉시 종료하므로 별도 연속 횟수와 누적 횟수는 같다.
- 첫 action의 구성을 고정하거나 강제 비율을 자동으로 줄이는 curriculum은 이번 기본 동작에 추가하지 않는다.
- 기본 파라미터 업데이트는 GPU 1개. rollout 분산과 업데이트 분산을 독립적으로 설정한다.
- 추가 옵션으로 GPU별 rollout replica와 2-GPU FSDP FULL_SHARD 업데이트를 제공한다.
- 학습 재시작이나 GPU 프로세스 실행은 별도 구현/실행 단계의 작업이다.

## 2. 현재 코드에서 바뀌어야 할 지점

`train.py:train`은 list comprehension으로 group을 순차 생성하고, 전체 group reward를 정규화한 뒤 turn별 backward를 수행한다. `model.py:load_model`은 WORLD_SIZE>1을 거부하며 모델 전체를 한 device로 이동한다. `action_log_probs`는 전체 sequence/vocabulary logits를 만든 뒤 생성 구간을 FP32로 변환한다. 따라서 분산 런처만 추가해서는 메모리 문제가 해결되지 않는다.

현재 Qwen 설정은 policy `cuda:0`, reference `cuda:1`이다. 이는 업데이트 GPU가 2개인 설정이 아니다. 또한 `sql.evaluator_path`는 `overall_pipeline/vendor/spider_test_suite_eval`을 가리킨다. 기존 evaluator 경로 의존성은 명시적으로 유지하되 새 학습 코드는 다른 프로젝트 학습 모듈을 import하지 않는다.

## 3. 파일 배치

아래 경로는 모두 `rl_rrcm_style/` 기준이다. 세부 모듈은 각 구현 단계에서 추가한다.

| 경로 | 역할 |
|---|---|
| `docs/mixed_exploration_multigpu_plan.md` | 이 계획과 구현 완료 기준 |
| `src/rrcm_sql/exploration.py` | group의 3+3 구성, turn별 action RNG, 임시 지시 조립 |
| `src/rrcm_sql/rollout.py` | action 지시 적용·검증, 종료 조건, 탐색 metadata |
| `src/rrcm_sql/model.py` | HF/local 모델 로딩 유지, backend별 placement, likelihood 메모리 개선 |
| `src/rrcm_sql/config.py` | 탐색 및 runtime 설정·검증 |
| `src/rrcm_sql/runtime/rollout_pool.py` | spawn worker, trajectory 작업 큐, 결과 수집, 정책 동기화 |
| `src/rrcm_sql/runtime/single.py` | 기존 단일 GPU RL updater 캡슐화 |
| `src/rrcm_sql/runtime/fsdp.py` | 2-rank sharding, 동기 backward·AMP·clipping |
| `src/rrcm_sql/runtime/checkpoint.py` | 단일/FSDP checkpoint, CPU export, 재시작 상태 |
| `src/rrcm_sql/runtime/launcher.py` | GPU 역할 검증, 선택적 실패 복구와 프로세스 재실행 |
| `src/rrcm_sql/train.py` | sample → rollout → reward → update 조율 |
| `src/rrcm_sql/cli.py` | runtime 선택, resume 및 warm-start 구분 |
| `configs/qwen3_1_7b_mixed.json` | 순차 rollout + 1-GPU update |
| `configs/qwen3_1_7b_mixed_rollout2.json` | 2-GPU rollout + 1-GPU update |
| `configs/qwen3_1_7b_mixed_fsdp2.json` | 2-GPU FSDP update + 전용 rollout GPU |
| `tests/test_exploration.py` | 재샘플링, 지시 불이행, cap, 평가 분리 |
| `tests/test_rollout_pool.py` | 순서·seed·정책 버전·실패 처리 |
| `tests/test_distributed_training.py` | loss 등가성, 가변 turn, skip, checkpoint |

기존 Qwen config는 보존하고 새 실험 config/output_dir를 만든다. 이전 4개 group checkpoint를 새 6개 혼합 탐색 학습의 exact resume로 취급하지 않는다. 모델 가중치만 가져오는 명시적 warm-start에서는 optimizer/counter를 새로 시작하고 초기 reference 정의도 저장한다.

## 4. 탐색과 prompt 조립

1. coordinator가 sample을 고르고 trajectory ID 0~5를 만든다. ID 0~2는 free, 3~5는 prompted_random이다.
2. 각 trajectory에 독립 RNG를 부여한다. seed는 run seed/group ID/trajectory ID에서 안정적으로 도출하고 Python hash는 사용하지 않는다. action RNG와 텍스트 sampling RNG를 분리한다.
3. prompted_random의 각 turn에서 answer/intermediate를 Bernoulli sampling한다. intermediate가 이미 3회 실행되었으면 RNG를 소비하지 않고 answer를 선택한다.
4. canonical history를 복사한 뒤 마지막 user message에 이번 turn만의 지시를 덧붙인다. 예: `For this turn, output exactly one <intermediate>...</intermediate> action.` 이전 turn의 일회성 지시가 누적되지 않게 한다. assistant action과 DB observation은 canonical history에 남긴다.
5. 실제 전달한 prompt token IDs, 생성 token IDs, 요청 action, 실제 action, cap 강제 여부, seed, 정책 버전을 저장한다.
6. 프롬프트는 hard decoding constraint가 아니다. 요청과 다른 action이 나오면 `action_instruction_violation`으로 종료하고 기존 non-executable penalty를 적용한다. 조용히 태그를 바꾸거나 성공할 때까지 재생성하지 않는다. cap 이후 intermediate는 실행하지 않는다.
7. free도 기존 intermediate cap=3을 적용한다. evaluation은 항상 free로 호출하고 탐색 설정이 자동 적용되지 않게 명시적인 rollout mode를 받는다.

## 5. RL objective와 앞선 설명의 정정

기본은 6개 reward를 함께 정규화하는 mixed-group GRPO-style 목적함수다. 다만 초기 question은 같아도 강제 지시가 다른 조건에서 생성한 샘플이다. 이를 자율 정책 하나의 엄밀한 on-policy GRPO라고 표현하지 않는다. 외부 action 제어 분포가 포함된 실험적 학습 목적이며 free-only validation으로 효과를 판단한다.

- old/current/reference likelihood는 각 turn을 실제 생성한 동일한 prompt IDs와 temperature를 사용한다. forced 출력을 free prompt로 바꿔서 재계산하면 안 된다.
- 이번 구현은 prompt 지시만 추가한다. 모델이 생성한 action tag도 조건부 정책의 출력이므로 기본 loss에 포함한다. 앞서 제안한 '강제 태그는 반드시 마스킹'은 필수 조건이 아니다.
- 향후 controller가 태그 token을 직접 삽입하거나 decoding으로 고정한다면 그 token은 별도로 mask하고 실제 sampling 분포를 다시 정의해야 한다. 이번 범위에는 포함하지 않는다.
- 강제 trajectory는 SQL/지시 따르기를 학습한다. 이것만으로 자율 action 선택을 학습했다고 간주하지 않는다. free 3개의 행동 지표를 따로 기록한다.
- 기존 trajectory 평균 → group 평균의 가중치를 보존한다. 긴 trajectory가 token 수만으로 더 큰 가중치를 받지 않게 한다.
- mixed 전체 및 free/forced 하위 집합의 reward variance를 함께 기록한다. 필요 시 free/forced 별도 advantage 계산은 후속 비교 실험으로 둔다.

## 6. Multi-GPU rollout

기본 backend는 현재 HF generate를 재사용한다. vLLM을 필수로 만들지 않아 HF ID, 로컬 full/SFT 모델, PEFT adapter 및 chat/plain template 지원을 유지한다.

- GPU마다 독립 프로세스와 policy replica 하나를 두고 complete trajectory 단위로 작업을 배분한다. worker 내부는 우선 순차 generate하여 KV cache 피크를 제한한다.
- 한 trajectory의 turn들은 같은 worker에서 처리하고 같은 정책 버전을 사용한다. SQL executor의 읽기 전용·timeout 제한을 그대로 유지한다.
- 동적 작업 큐로 긴 trajectory의 편중을 줄이되 결과는 trajectory ID 순서로 정렬한다. worker 완료 순서가 3+3 구성이나 reward 정규화에 영향을 주면 안 된다.
- 모든 worker는 해당 rollout batch 시작 전 동일한 policy version을 로드하고 ACK한다. 업데이트 중에는 다음 batch를 생성하지 않는 동기 방식으로 시작한다.
- LoRA는 동일 base와 adapter 구조를 확인한 뒤 trainable adapter state를 동기화한다. full finetuning은 전체 state가 필요하며 전송·복제 비용을 측정한다. 초기 로컬 SFT adapter를 포함한 모든 고정 state의 일치도 확인한다.
- old log-prob는 worker의 생성 정책으로 실제 prompt에 대해 재계산해 CPU에 반환한다. reference도 동일 prompt로 별도 scoring한다. 모델 버전/설정 hash가 다르면 batch를 거부한다.
- worker 사망/timeout은 SQL non-executable reward로 위장하지 않는다. 같은 seed/version으로 제한적 재시도 후 run 오류로 처리한다.
- 재현성은 작업 seed/배정 기록을 기본으로 보장한다. GPU kernel이나 topology가 바뀐 경우 bitwise 일치까지 약속하지 않는다.

### GPU 배치 예시

모든 번호는 CUDA_VISIBLE_DEVICES 적용 후 논리 번호다. 시작 시 실제 UUID/메모리와 대응을 출력한다.

| 모드 | GPU 역할 |
|---|---|
| 기본: 2개 visible | GPU0 policy rollout/update, GPU1 frozen reference |
| rollout 분산: 4개 visible | GPU0 updater, GPU1 reference, GPU2·3 rollout replicas |
| FSDP: 5개 visible | GPU0·1 sharded updater, GPU2 reference, GPU3·4 rollout replicas |
| 2개만으로 FSDP | GPU0·1을 rollout/scoring/update phase별로 재사용; CPU state와 별도 프로세스 수명 관리 필요 |

업데이트 GPU 1개는 총 GPU 사용량이 1개라는 뜻이 아니다. 진짜 총 1개 모드는 reference CPU scoring 또는 phase별 offload로 지원하고 속도 비용을 명시한다. GPU overlap은 전용 phase-sharing 구현 외에는 validation에서 거부한다.

2개만 있는 FSDP 모드는 후순위로 둔다. 먼저 inference worker를 종료해 VRAM 해제를 확인한 후 FSDP updater를 실행하고, 다음 rollout에는 CPU checkpoint에서 replica를 재구성한다. 초기 버전은 느려도 프로세스 종료로 메모리 소유권을 명확히 한다. optimizer는 CPU로 보존하며 매 phase 재설정하지 않는다. FSDP model 전체를 GPU에 gather해서 generate하는 접근은 12GB 회피 목적과 충돌하므로 사용하지 않는다.

## 7. 12GB 업데이트 OOM 대응

### 7.1 단일 GPU를 기본으로 유지

현재처럼 한 turn씩 backward하고 gradient checkpointing을 사용한다. rollout/ref 모델이 updater GPU에 중복 상주하지 않게 한다. logits 계산은 생성 구간의 log-softmax를 chunk 처리하고, 모델이 지원하면 필요한 logits만 반환하는 기능을 사용한다. 모델별 지원 여부를 검사하고 일반 forward fallback을 제공한다. 전체 logits가 생성되는 fallback에서는 chunk 처리만으로 모든 peak가 사라지지 않음을 측정한다.

OOM 로그는 rollout/KV cache, old/reference scoring, policy forward/backward, optimizer step, save 단계를 구분하고 rank별 allocated/reserved peak와 sequence 길이를 남긴다. FSDP가 해결하는 state 메모리와 여전히 각 GPU에 남는 activation/logits 메모리를 구분한다. GPU 두 개가 24GB 연속 메모리처럼 동작하거나 OOM이 반드시 사라진다고 보장하지 않는다.

### 7.2 선택 가능한 2-GPU FSDP updater

DDP는 모델을 복제하므로 이번 메모리 fallback으로 선택하지 않는다. PyTorch FSDP FULL_SHARD로 parameter/gradient/optimizer state를 나눈다. 기존 설치 버전과 호환되는 FSDP API를 고정해 검증하고 선택적 dependency 범위를 문서화한다.

- distributed 환경 초기화 → CPU/meta 기반 로딩과 shard 배치 → wrapping → optimizer 생성 순서를 따른다. 기존 `model.to(cuda)` 후 wrapping 경로는 큰 모델 로딩 중 OOM이 날 수 있어 분리한다.
- transformer block wrap은 모델 metadata에서 추론하고 class-name override를 허용한다. 모델명 whitelist는 두지 않되 지원 불가능한 구조는 이유를 명시한다.
- LoRA의 frozen/trainable parameter 혼합을 처리하는 PEFT wrap policy와 `use_orig_params` 조합을 검증한다. 무조건 모든 모델에 같은 wrapping을 적용하지 않는다.
- full finetuning도 backend 인터페이스로 지원하고 별도 memory smoke test를 한다. QLoRA+FSDP는 quantized storage/dtype 조합 검증 후 opt-in으로 활성화한다. 기존 single-GPU QLoRA 지원을 제한하지 않는다.
- reference는 초기 정책으로 고정한다. 전용 GPU 또는 CPU scoring을 사용하며 updater rank마다 reference replica를 암묵적으로 만들지 않는다. 초기 SFT adapter가 있다면 adapter disable만으로 reference를 대체하지 않는다.
- runtime 입력 device와 autocast는 FSDP wrapper를 인식하도록 한다. sharded embedding submodule을 직접 호출하는 우회 경로를 만들지 않는다.

### 7.3 가변 trajectory의 distributed backward

단순히 trajectory 3개씩 나누면 turn 수 차이로 rank별 collective 호출 횟수가 달라져 hang할 수 있다. coordinator가 전역 turn schedule을 만들고 모든 rank가 같은 횟수의 forward/backward를 실행하도록 한다. 없는 slot은 유효 dummy 입력의 loss를 0으로 곱해 graph/collective 참여를 유지한다. rank별 token 길이 padding은 slot 내에서만 한다.

FSDP gradient 평균을 고려해 각 rank의 local loss에 world_size를 반영하여 기존 전역 trajectory 평균과 같게 한다. advantage는 전역 6개에서 한 번만 계산한다. all-equal skip도 모든 rank가 함께 결정한다. 초기 버전은 매 backward 동기화하여 no_sync 중 full gradient 누적에 따른 메모리 증가를 피한다.

global gradient clipping은 FSDP의 shard-aware API를 사용한다. AMP overflow/skip과 scaler 갱신은 모든 rank에서 동일해야 하며 한 rank에서만 overflow가 발생하는 테스트를 넣는다. 기존 연속 AMP skip 상한 5와 성공 update만 step에 반영하는 규칙을 유지한다.

### 7.4 OOM 시 전환 정책

기본 `oom_fallback=error`: 진단과 FSDP config 재실행 경로를 제시한다. 선택적으로 `restart_fsdp2`를 설정한 경우 supervisor가 GPU 수·배치 가능 여부를 사전 확인하고 한 번만 재시작한다. 살아 있는 단일 GPU 프로세스를 즉석에서 FSDP로 바꾸지 않는다.

자동 전환을 켠 run은 첫 update 전에도 완전한 복구 checkpoint를 남긴다. 실패한 batch와 부분 gradient/부분 optimizer 변경은 버리고 마지막 complete checkpoint부터 재생성한다. rank hang/NCCL 오류는 전체 worker 종료 후 복구하며 무조건 OOM으로 재분류하지 않는다.

checkpoint에는 모델, optimizer, scaler, scheduler가 있다면 그 state, step/group counter, reference identity, 탐색 RNG, policy version, backend/topology, config/data hash를 보존한다. 단일↔FSDP optimizer 변환을 검증한 경우에만 exact resume를 허용한다. 미지원 조합은 optimizer를 조용히 초기화하지 않고 명시적으로 중단한다. 저장은 임시 디렉토리와 complete marker를 사용하고 committed checkpoint를 덮어쓰지 않는다.

## 8. 설정 인터페이스 초안

아래는 구현 예정 필드이며 현재 CLI에서 사용할 수 없다. 기존 config가 그대로 로드되도록 새 기능은 기본 비활성화하고 새 mixed config에서 활성화한다.

```json
{
  "rollout": {
    "group_size": 6,
    "max_intermediate": 3,
    "exploration": {
      "free_trajectories": 3,
      "prompted_trajectories": 3,
      "answer_probability": 0.5
    }
  },
  "runtime": {
    "rollout_backend": "local",
    "rollout_devices": [],
    "update_backend": "single",
    "update_devices": ["cuda:0"],
    "reference_device": "cuda:1",
    "phase_sharing": false
  }
}
```

`rollout_backend=hf_workers`와 `rollout_devices`로 rollout 분산을 활성화한다. FSDP는 `update_backend=fsdp`, `update_devices` 2개로 선택한다. legacy `model.device/reference_device`는 runtime이 없을 때 해석하고, 두 설정이 충돌하면 오류를 낸다. group 합=6, 확률 범위, 장치 존재/중복, backend world size, cap 등을 시작 전에 검사한다.

## 9. 관측과 검증

- 공통/free/prompted별 outcome, reward, intermediate 횟수, token 수, latency, 지시 준수율, cap 도달률을 기록한다. group 전체 및 subset all-equal을 구분한다.
- task ID, requested/actual action sequence, policy version, worker device, phase별 peak VRAM, 성공 update/AMP skip/OOM 복구 횟수를 기록한다.
- action RNG를 제어하는 테스트로 즉시 answer, intermediate→answer, intermediate 3회→강제 answer, 실패 intermediate 횟수 포함, cap 초과 미실행, 이전 지시 미누적을 검사한다.
- 서로 다른 worker 완료 순서에서도 group 구성·task seed·version이 보존되는지 검사한다. 오류 worker의 부분 결과로 group을 업데이트하지 않는다.
- tiny causal LM으로 단일/FSDP 전역 loss 및 한 update 결과를 허용 오차 내 비교한다. 가변 turn·빈 slot·all-equal·한 rank overflow·save/resume를 포함한다.
- CUDA smoke test는 12GB GPU에서 최대 설정 길이 근처의 4-turn trajectory로 forward/backward/첫 Adam step/checkpoint를 모두 실행한다. 실제 peak를 보고하고 테스트하지 않은 조합은 지원 검증 완료로 표시하지 않는다.
- 학습 효과는 동일한 공식 dev의 free inference로 비교한다. baseline free-only 6개 vs mixed 3+3을 같은 rollout/token 예산과 여러 seed로 비교하여 group 증가 효과와 탐색 효과를 구분한다. Dev로 튜닝하고 공개 test로 최종 평가한다.

## 10. 구현 순서와 완료 기준

1. 혼합 탐색: single updater에서 3+3, 매 turn sampling, cap=3, 실제 prompt likelihood, 세부 로그와 새 config를 완성한다.
2. 메모리 진단: 단계별 peak 측정 및 likelihood 계산의 불필요한 메모리를 줄인다. 12GB smoke 결과를 기준으로 판단한다.
3. rollout 분산: 전용 GPU worker pool과 동기 policy version 배포를 구현한다. 순차/분산 결과·비용을 비교한다.
4. FSDP 업데이트: 2-rank 가변 turn schedule, loss 가중치, AMP, clipping, checkpoint까지 검증한다. 우선 Qwen LoRA, 이후 일반 full 모델과 quantized 조합을 검증한다.
5. 복구/최소 GPU 운영: opt-in OOM 재시작과 single↔FSDP state 변환을 검증하고 2-GPU phase sharing을 추가한다.
6. README에 구현된 설정만 실행 명령으로 공개하고, 짧은 학습과 free validation 결과를 함께 기록한다.

완료 기준은 '6개가 생성됨'에 그치지 않는다. 실제 조건과 likelihood의 일치, 정책 버전 일치, rank 간 update/skip 일치, 재시작 상태 보존, measured peak memory, free inference 평가가 모두 확인되어야 한다. 탐색 다양성 증가는 가능성이며 all-equal 감소나 정확도 향상 자체는 사전에 보장하지 않는다.

## 참고 문서

- [PyTorch FSDP 공식 문서](https://docs.pytorch.org/docs/stable/fsdp.html): sharding, frozen parameter 제약, collective와 checkpoint API 확인용.
- [Hugging Face PEFT FSDP 가이드](https://huggingface.co/docs/peft/main/accelerate/fsdp): LoRA wrapping 및 QLoRA 조합 확인용. 설치 버전별 동작은 구현 시 별도 검증한다.
