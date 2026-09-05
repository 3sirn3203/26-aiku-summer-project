# Spider 1.0 Text-to-SQL Evaluation

이 디렉터리는 Spider 1.0에서 다음 두 실험을 서로 독립적으로 수행합니다.

- 고정한 single-turn Text-to-SQL zero-shot baseline
- 최대 3회 반복하는 Planner–Coder–Verifier agentic workflow

두 실험 모두 고정 8개 smoke test와 전체 development split 평가를 지원합니다.
로컬 macOS에서는 실제 모델을 다운로드하거나 실행하지 않고 데이터·prompt·SQL
executor·공식 평가·분산 orchestration을 deterministic mock으로 검증합니다.
실제 Qwen inference는 TITAN Xp 서버에서만 실행합니다.

## Single-turn baseline 계약

- 모델: `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- 모델 revision: `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`
- 입력: 자연어 질문과 `tables.json` 기반 전체 DB schema
- DB row/value, few-shot example, gold SQL: 모델 입력에서 제외
- 문제당 generation 1회
- 실행 feedback, retry, self-correction: 없음
- decoding: pure greedy(`do_sample=false`, `num_beams=1`,
  `repetition_penalty=1.0`), `max_new_tokens=512`, batch size 1
- generation soft limit: 문제당 `max_time=120`초
- smoke 서버 실행: 단일 TITAN Xp, FP32, eager attention
- 전체 dev 추론: 지정한 GPU마다 독립 모델 process 1개, GPU 간 data parallel
- 전체 dev SQL 실행 및 timing: 모든 GPU 추론 종료 후 CPU 부모 process에서 순차 실행
- BF16, quantization, FlashAttention, `torch.compile`, model sharding: 사용하지 않음
- smoke 표본: Spider `dev.json`의 고정 8개
- 정확도: 공식 Test Suite Accuracy와 Original Exact Set Match를 각각 산출
- SQL 실행시간: 예측 SQL의 SQLite query 구간을 질문별로 nanosecond 단위 기록
- Docker: 사용하지 않음

## 디렉터리

```text
code/
├── configs/
│   ├── single_turn_zero_shot.json
│   ├── single_turn_sft_instruct.json
│   ├── single_turn_sft.json
│   ├── single_turn_sft_augmented.json
│   └── multi_turn_multi_agent_zero_shot.json # Planner–Coder–Verifier smoke/full dev
├── src/text2sql/
│   ├── core/                   # 데이터·모델·SQL 실행·공식 평가 공통 인프라
│   ├── single_turn/            # 고정한 single-turn baseline
│   └── multi_turn_agent/       # 독립적인 agentic workflow와 role worker
├── tests/
├── vendor/spider_test_suite_eval/ # 고정한 공식 evaluator 원본
├── outputs/                  # Git 제외
├── pyproject.toml
└── README.md
```

Spider 원본은 코드 밖의 다음 위치를 전제로 합니다.

```text
../data/spider_data/
├── dev.json
├── tables.json
└── database/<db_id>/<db_id>.sqlite

../data/spider_test_suite/
└── database/<db_id>/         # base DB와 generated test-suite DB들

../data/nltk_data/            # 공식 Exact Match tokenizer resource
```

`database/**/*.sql`은 prompt나 실행에 사용하지 않습니다. Prompt schema는 `tables.json`으로 만들고, 실제 실행에는 `.sqlite` 파일만 사용합니다.

현재 고정한 `spider_data.zip`의 SHA-256은 다음과 같습니다. 서버로 원본
archive를 옮기거나 다시 압축 해제할 때 같은 값인지 확인합니다.

```text
00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b  spider_data.zip
```

```bash
# macOS
shasum -a 256 ../data/spider_data.zip

# Linux
sha256sum ../data/spider_data.zip
```

## 공식 Spider evaluator 준비

공식 evaluator는 [`taoyds/test-suite-sql-eval`](https://github.com/taoyds/test-suite-sql-eval)의
commit `e97acc546ecbee8fa27fa8dbf025ef61493a876c`의 공식 소스 파일을
`vendor/spider_test_suite_eval/`에 수정하지 않고 고정했습니다. Apache-2.0
라이선스와 upstream commit 정보도 vendor 디렉터리에 함께 보존합니다.

Test Suite Accuracy에는 Spider 원본 DB 한 개가 아니라 공식 generated
test-suite DB bundle이 필요합니다. 공식
[Google Drive 파일](https://drive.google.com/file/d/1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w/view?usp=sharing)을
`../data/testsuitedatabases.zip`으로 받은 뒤 다음 checksum을 먼저 확인합니다.

```text
9ec24ea8debc6bd04abfe137b5f1a739b5a8836f32c0464e4dfc94eb7f41da96  testsuitedatabases.zip
```

```bash
# macOS
shasum -a 256 ../data/testsuitedatabases.zip

# Linux
sha256sum ../data/testsuitedatabases.zip
```

그다음 `database/` 항목만 별도 root에 풉니다. 압축 파일은 약 1.27 GB,
압축 해제 후 데이터는 약 4.8 GB입니다.

```bash
test ! -e ../data/spider_test_suite/database
unzip -q -n ../data/testsuitedatabases.zip 'database/*' \
  -d ../data/spider_test_suite
```

첫 명령이 실패하면 기존/부분 추출 디렉터리를 자동으로 덮어쓰지 말고 상태를
확인합니다. `-n`도 overwrite 질문 없이 기존 파일을 보존합니다.

`spider_data/test_database`는 hidden test split용 DB이며 generated test-suite
bundle이 아니므로 대체해서는 안 됩니다.

## 로컬 환경과 검증

공식 evaluator 의존성만 설치한 프로젝트 전용 환경을 만듭니다. 이 extra는
`torch`, `transformers` 또는 모델을 설치하지 않습니다.

```bash
cd overall_pipeline
python3 -m venv .venv
.venv/bin/python -m pip install pip==25.1.1
.venv/bin/python -m pip install -e '.[official-eval]'
.venv/bin/python -m nltk.downloader \
  -d ../data/nltk_data punkt punkt_tab
```

준비 이후 아래 명령은 모델 라이브러리나 network를 사용하지 않습니다. 현재
로컬 Python 3.9에서도 동작합니다.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
.venv/bin/python -m text2sql validate-data --config configs/single_turn_zero_shot.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
.venv/bin/python -m text2sql validate-official --config configs/single_turn_zero_shot.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
.venv/bin/python -m unittest discover -s tests -v

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
.venv/bin/python -m text2sql smoke \
  --config configs/single_turn_zero_shot.json \
  --backend mock
```

Mock backend은 고정 표본에 대응하는 gold SQL을 deterministic response로 사용합니다. Gold SQL은 backend 내부 mapping에만 있고 prompt 또는 `GenerationRequest`에는 전달되지 않습니다. Mock 결과의 100% 일치는 모델 성능이 아니라 배관 검증 결과입니다.

로컬 smoke 통과 조건은 다음과 같습니다.

- 고정 8개 모두 record 생성
- generation과 SQL parse 8개 모두 처리
- gold/predicted SQL 8개 모두 읽기 전용 SQLite에서 실행 성공
- 공식 evaluator가 8개 모두 분류하고 Test Suite/Exact Match 결과 생성
- 질문별 prediction query 실행시간과 timing summary 생성
- 선택된 SQLite 파일 hash 불변
- `run_manifest.json`, `records.jsonl`, `summary.json` 생성

## SQL 실행 안전성

생성 SQL은 별도의 `spawn` subprocess에서 실행됩니다. Linux에서도 `fork`를 사용하지 않으므로 부모 프로세스의 CUDA 상태를 복제하지 않습니다.

실행기는 다음 방어를 함께 적용합니다.

- SQLite URI `mode=ro`
- `PRAGMA query_only=ON`
- SQLite authorizer read-only allowlist
- extension loading 비활성화
- 다중 statement 차단
- 내부 progress timeout과 부모 wall-clock timeout
- SQL 길이와 단일 SQLite 값 크기 제한, 메모리에 보관·IPC로 반환하는 결과
  row prefix 및 byte 크기 제한
- Linux worker address-space 1 GiB 제한 및 Python 3.11 SQLite engine 길이 제한
- `randomblob`, `zeroblob`, `printf`, `format` 등 메모리 증폭 함수 차단
- mutation, DDL, ATTACH, PRAGMA, unsafe function 차단

`execution.max_result_rows`와 누적 `execution.max_result_bytes`는 전체 cursor
실행을 중단하는 정확도 상한이 아니라, worker가 메모리에 보관하고 부모
process로 보내는 row prefix의 상한입니다. `max_result_bytes`는 비정상적으로 큰
단일 SQLite 값에 대한 engine-level 상한으로도 사용합니다. Cursor는 timeout
안에서 끝까지 스트리밍하며 전체 `row_count`와 ordered/unordered fingerprint를
계산합니다. 따라서 큰 정상 결과도 비교할 수 있고, artifact와 IPC payload는
계속 제한됩니다.

이 구성은 현재의 단일 사용자 연구 환경을 위한 것입니다. 외부 사용자가 임의 SQL이나 DB 파일을 제출하는 서비스로 확장할 경우에는 별도의 container sandbox가 필요합니다.

공식 test-suite evaluator는 upstream 동작을 보존하되 별도 hard-timeout
subprocess에서 실행합니다. Test Suite Accuracy 실행에는 기존 안전 executor를
통과했는지와 별개로 adapter가 공식 `value`→`1` 정규화 후 다시 검증한 단일
읽기 전용 `SELECT`만 전달하며, 원본 test-suite DB 대신 run별 disposable
copy를 사용합니다. 따옴표로 감싼 함수명과 함수명 뒤 주석도 검사하며,
Linux의 공식 evaluator worker에는 address-space 상한을 fail-closed로
적용합니다. 이 worker에서만 OpenBLAS·OpenMP·MKL·NumExpr thread 수를 1로
강제하여 다중 thread 초기화가 1 GiB address-space 상한을 소진하지 않게
합니다. 따라서 서버 실행 명령에 해당 환경변수를 별도로 붙일 필요가
없습니다. 로컬 time/value-size limit 또는 schema 실행 오류는 공식 점수를
임의로 0으로 만들지 않으며, 쓰기·다중 statement·위험 함수만 실행 대상에서
제외해 오답으로 분류합니다.

## 서버 환경 준비

`base` 환경을 직접 변경하지 말고 별도의 환경을 권장합니다. 서버는 Python
3.11을 기준으로 합니다. 이 버전에서는 Python의 SQLite engine-level length
limit도 적용됩니다. 로컬 파이프라인은 현재 macOS Python 3.9에서도 동작합니다.

```bash
conda create -n spider-smoke python=3.11 -y
conda activate spider-smoke
cd overall_pipeline
python -m pip install pip==25.1.1
```

PyTorch는 서버 driver 535.309.01과 TITAN Xp Pascal에 맞춘 보수적 고정
조합을 사용합니다. 공식 [PyTorch 이전 버전 설치
안내](https://docs.pytorch.org/get-started/previous-versions/)에 있는 2.5.1
CUDA 12.1 wheel에서 이 프로젝트에 필요한 `torch`만 설치합니다.

```bash
python -m pip install torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

이 저장소는 호환되지 않는 최신 CUDA/PyTorch build를 암묵적으로 설치하지
않도록 `torch`를 Python package dependency에 포함하지 않습니다.

PyTorch 설치 후 inference와 공식 evaluator dependency를 함께 설치합니다.

```bash
python -m pip install -e '.[inference,official-eval]'
python -m nltk.downloader -d ../data/nltk_data punkt punkt_tab
```

서버에서도 `code/`와 `data/`가 같은 프로젝트 root 아래 형제 디렉터리가 되게
전송하거나 동일한 절차로 준비합니다.

```text
<project-root>/
├── code/
└── data/
    ├── spider_data/
    ├── spider_test_suite/database/
    └── nltk_data/
```

압축 해제된 세 디렉터리를 전송했다면 zip 파일 자체는 서버 실행에 필요하지
않습니다. 전송 후에는 아래 두 validation을 통과하기 전 모델을 로드하지
않습니다.

설치한 PyTorch build의 실제 호환성을 모델 다운로드 전에 확인합니다.

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader

CUDA_VISIBLE_DEVICES=0 \
python -m text2sql doctor --config configs/single_turn_zero_shot.json
```

Doctor는 모델을 로드하지 않습니다. 최소한 다음이 확인되어야 합니다.

```text
gpu_name: NVIDIA TITAN Xp
gpu_capability: [6, 1]
cuda_available: true
cuda_fp32_probe: passed
gpu_memory_free_bytes: 4 GiB 이상
torch_version: 2.5.1+cu121
transformers_version: 4.46.3
ok: true
model_loaded: false
```

`torch_arch_list`도 진단 정보로 남지만, 호환 여부는 `sm_61` 문자열의 존재가
아니라 실제 FP32 CUDA 연산 성공으로 판정합니다.

그다음 데이터 경로를 검증합니다.

```bash
python -m text2sql validate-data --config configs/single_turn_zero_shot.json
python -m text2sql validate-official --config configs/single_turn_zero_shot.json
```

## 서버 Qwen smoke test

첫 실행에서만 명시적으로 다운로드를 허용합니다.

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m text2sql smoke \
  --config configs/single_turn_zero_shot.json \
  --backend hf \
  --device cuda:0 \
  --allow-model-download
```

각 generation에는 120초 soft limit가 있습니다. GPU driver 자체가 멈추는 경우는
프로세스 내부 limit로 중단할 수 없으므로, 최초 서버 실행은 Linux의 외부
supervisor도 함께 두는 것을 권장합니다. 아래 60분 한도는 8개 generation과
16개 공식 metric worker가 각각 내부 한도까지 사용하는 최악의 정상 경로 및
최초 모델 다운로드 여유를 포함합니다.

```bash
CUDA_VISIBLE_DEVICES=0 \
timeout --signal=TERM --kill-after=30s 60m \
python -m text2sql smoke \
  --config configs/single_turn_zero_shot.json \
  --backend hf \
  --device cuda:0 \
  --allow-model-download
```

모델이 cache에 준비된 뒤에는 다운로드 flag를 제거하여 network나 최신 artifact에 암묵적으로 의존하지 않게 합니다.

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m text2sql smoke \
  --config configs/single_turn_zero_shot.json \
  --backend hf \
  --device cuda:0
```

첫 성공 run의 `run_manifest.json`에는 Hugging Face가 resolve한 model commit hash가 기록됩니다. 이후에는 `configs/single_turn_zero_shot.json`의 `model.revision`을 해당 hash로 바꾸어 재현성을 고정해야 합니다.
Resolved commit hash를 확인할 수 없는 경우에는 pinning 계약을 만족하지 못한 것으로
간주하여 backend 초기화가 실패합니다.

현재 `configs/single_turn_zero_shot.json`은
`Qwen/Qwen2.5-Coder-0.5B-Instruct`의
`ea3f2471cf1b1f0db85067f1ef93848e38e88c25` commit으로 고정되어 있습니다.

### 로컬 SFT checkpoint 사용

Hub에 업로드하지 않은 SFT 모델은 기존 config를 복사한 뒤 `model`만 다음처럼
바꿉니다. `id`의 상대경로는 config 파일이 있는 디렉터리를 기준으로 해석됩니다.
서버에서 실행할 때 모든 GPU worker가 같은 경로를 읽을 수 있어야 합니다.

```json
{
  "source": "local",
  "id": "../checkpoints/qwen-coder-sft-step-1000",
  "revision": "local",
  "dtype": "float32",
  "device": "cuda:0",
  "attention_implementation": "eager",
  "trust_remote_code": false,
  "cache_dir": null
}
```

초기 지원 범위는 model weights, `config.json`, `tokenizer_config.json`과 tokenizer
vocabulary를 함께 저장한 self-contained Hugging Face `save_pretrained()` 전체
checkpoint입니다. PEFT/LoRA adapter만 있는 디렉터리는 지원하지 않으므로 먼저
base model에 merge하거나 전체 모델로 저장해야 합니다.

실행 명령은 기존과 같으며 로컬 checkpoint에서는 `--allow-model-download`를
사용하지 않습니다. `doctor`는 모델을 GPU에 올리지 않고 필수 파일과 checkpoint
fingerprint를 확인합니다. 실행 시 inference 관련 파일 전체의 SHA-256 기반
`local-sha256:...` identity를 한 번 계산하여 manifest에 기록하며, checkpoint
내용이 바뀐 경우 기존 run의 resume을 거부합니다.

서버 pipeline 통과 조건은 다음과 같습니다.

- 모델/tokenizer 로드 성공
- 8개 모두 generation 종료 및 raw output 저장
- 각 출력을 SQL 또는 명시적인 parse error로 분류
- gold SQL 8개 모두 실행 성공
- 예측 SQL 최소 1개 실행 성공
- 공식 Test Suite Accuracy와 Original Exact Set Match를 8개 모두 산출
- 성공한 예측 SQL의 질문별 query 실행시간을 기록
- timeout과 오류를 포함한 모든 결과가 구조적으로 기록됨
- SQLite 파일 hash 불변

`test_suite_accuracy`가 현재 Spider 공식 정확도이며,
`exact_set_match_accuracy`는 기존 구조 기반 지표입니다. 원본 DB 한 개에서
직접 비교하는 `local_result_match_rate`는 파이프라인 진단용 보조값으로서 두
공식 evaluator 지표와 구분합니다.

예측 SQL의 문법·실행 오류나 안전성 탈락은 해당 예제의 0점으로 분모에
포함합니다. Test Suite evaluator timeout도 해당 예측의 실행 실패로 간주하여
그 예제만 0점으로 처리하고 전체 Test Suite Accuracy는 계속 계산합니다. 반면
gold 오류, evaluator 자체 오류 또는 SQL을 실행하지 않는 Exact Match의 evaluator
timeout이 한 건이라도 있으면 모델 오답으로 합산하지 않고 영향받은 metric의
`valid=false`, accuracy=`null`로 기록하며 pipeline을 실패시킵니다. 두 공식
metric의 유효성은 서로 독립적으로 보존합니다.

## RL adapter 평가 연동

루트 `rl_finetune/`에서 생성한 PEFT adapter는 기존 분산 평가와 동일한 SQL
실행, 공식 Spider metric, query timing, VM-step 계약으로 평가합니다.

```bash
python -m text2sql evaluate \
  --config configs/single_turn_zero_shot.json \
  --backend peft \
  --adapter-dir ../rl_finetune/outputs/<run>/adapter \
  --gpus 0,1,2,3,4,5,6,7 \
  --selection all \
  --output-dir ../rl_finetune/outputs/evaluation \
  --run-name <evaluation-run>
```

`--adapter-dir`은 `--backend peft`에서만 허용됩니다. Adapter content hash와
base model revision이 run contract에 포함되므로 adapter가 바뀐 기존 run은
resume하지 않습니다. Two-turn RL 평가는 `rl_finetune`의 별도 entrypoint를
사용하지만 final SQL 평가는 이 파이프라인의 부모 평가 단계를 공유합니다.

## 전체 dev zero-shot 평가

전체 평가는 train split을 사용하거나 fine-tuning하지 않고 `dev.json` 1,034개를
모두 평가합니다. `test.json`은 현재 준비된 generated test-suite DB bundle로
공식 Test Suite Accuracy를 산출할 수 없으므로 이 명령의 대상이 아닙니다.

전체 dev에서 사용하는 20개 DB의 generated suite를 먼저 모두 확인합니다.

```bash
python -m text2sql validate-data --config configs/single_turn_zero_shot.json
python -m text2sql validate-official \
  --config configs/single_turn_zero_shot.json \
  --all-examples
```

분산 방식은 model parallel, DDP 또는 NCCL이 아닙니다. 각 physical GPU에
Qwen 전체 모델을 하나씩 올리고 질문 index를 round-robin으로 나누는 독립
process data parallel입니다. 부모 process는 `torch`를 import하지 않으며 각
worker에 `CUDA_VISIBLE_DEVICES=<physical index>`를 따로 설정합니다. 따라서 모든
worker 내부 device는 항상 `cuda:0`입니다.
Host RAM과 model cache I/O가 동시에 급증하지 않도록 각 worker의 model load가
완료되어 generation-ready 상태가 된 뒤 다음 GPU worker를 시작합니다. 이후
generation 자체는 모든 준비된 worker에서 병렬로 진행됩니다.

먼저 사용할 GPU 각각에 대해 doctor를 통과시키고 4 GiB 이상의 free VRAM을
확인합니다. 아래 예시는 physical GPU 1~7을 사용합니다.

```bash
for gpu in 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$gpu" \
  python -m text2sql doctor --config configs/single_turn_zero_shot.json || exit 1
done
```

모델이 smoke 단계에서 이미 Hugging Face cache에 준비되었으므로 full run에는
download flag를 붙이지 않습니다. 외부 `CUDA_VISIBLE_DEVICES`도 붙이지 않습니다.
평가 부모가 `--gpus`의 physical index를 worker별로 직접 매핑합니다.

처음에는 free GPU 두 개로 고정 8개를 새 분산 경로에서 확인합니다.

```bash
timeout --signal=TERM --kill-after=30s 60m \
python -m text2sql evaluate \
  --config configs/single_turn_zero_shot.json \
  --backend hf \
  --selection smoke \
  --gpus 1,2 \
  --run-name server-distributed-smoke
```

이 run이 `pipeline_pass=true`, `total_examples=8`로 끝나면 전체 평가를 시작합니다.

실행 중 progress는 최종 JSON과 섞이지 않도록 stderr에 표시됩니다. Interactive
terminal에서는 `tqdm` bar를 사용하고, scheduler log나 redirect된 stderr에서는
기본 5초 간격의 일반 텍스트를 남깁니다. GPU worker model 준비, 전체 generation,
순차 SQL 실행 및 공식 평가의 현재 건수·비율·속도·ETA를 확인할 수 있습니다.
SQL progress는 측정된 query가 끝난 뒤에만 갱신하고 tqdm background monitor도
비활성화하므로 `query_elapsed_ns` 측정 구간에는 UI 작업을 포함하지 않습니다.

```text
[progress] worker-00 is ready on physical GPU 1
LLM generation: 317/1034
Sequential SQL execution: 128/1034
Official Spider evaluation: 64/1034
```

표시를 끄거나 비대화형 로그 간격을 변경할 수도 있습니다.

```bash
--no-progress
--progress-interval-seconds 10
```

```bash
timeout --signal=TERM --kill-after=30s 4h \
python -m text2sql evaluate \
  --config configs/single_turn_zero_shot.json \
  --backend hf \
  --selection all \
  --gpus 1,2,3,4,5,6,7 \
  --run-name qwen25-coder-05b-zero-shot-dev
```

추론 worker는 prompt 생성, generation 및 SQL parsing까지만 수행합니다. 모든
shard가 원래 `dev.json` 순서로 엄격히 병합되고 worker가 GPU memory를 해제한
뒤, 부모 process 하나가 각 질문의 prediction SQL과 gold SQL을 이 순서로
실행합니다. 그러므로 GPU worker 수가 달라도 SQL 실행시간 측정에는 병렬 CPU/DB
부하가 섞이지 않습니다. 공식 Exact Set Match와 Test Suite Accuracy도 그 뒤에
CPU에서 수행합니다.

Worker는 생성한 각 record를 즉시 shard JSONL에 flush합니다. 외부 supervisor나
worker 오류로 중단되면 같은 source/config/model 계약으로 다음처럼 재개합니다.
완료된 example은 다시 생성하지 않고 남은 example만 현재 지정한 GPU들에
재분배합니다. SQL 실행 stage도 완료된 원래 순서 prefix를 건너뛰며, 공식 평가는
8개 단위로 결과를 저장합니다.

```bash
timeout --signal=TERM --kill-after=30s 4h \
python -m text2sql evaluate \
  --config configs/single_turn_zero_shot.json \
  --backend hf \
  --selection all \
  --gpus 1,2,3,4,5,6,7 \
  --resume-run qwen25-coder-05b-zero-shot-dev
```

재개 시 Python source tree, 실효 config, dataset hash 또는 generation contract가
달라졌으면 기존 결과를 섞지 않고 거부합니다. GPU 목록과 순서는 남은 작업을
재배치할 수 있도록 변경할 수 있습니다.

0.5B Coder 모델 계약이 적용된 package version은 `0.4.1`입니다.
Per-example record에서 prompt 본문, token 수, model revision, worker/GPU 정보를
제거했습니다. 이전 version으로 만든 run은 source tree와 artifact 계약이
다르므로 `0.4.1` 코드로 resume하지 않고 새 run name을 사용합니다.

Single-turn run의 source contract은 `config.py`, `core/`, `single_turn/`만
포함합니다. 향후 `multi_turn_agent/`에 실험 코드를 추가하는 것만으로
기존 single-turn run의 source hash가 바뀌지는 않습니다. 단, 공통 `core/`나
single-turn 코드가 바뀌면 기존처럼 resume를 거부합니다.

Planner–Coder–Verifier workflow가 추가된 package version은 `0.5.0`입니다.
Agentic run은 `config.py`, `core/`, `multi_turn_agent/`를 별도 source contract로
사용하므로 `0.4.1` single-turn artifact와 섞거나 상호 resume하지 않습니다.

Test Suite evaluator timeout을 해당 example의 오답으로 집계하는 metric contract는
package version `0.5.1`부터 적용됩니다.

## Multi-turn Planner–Coder–Verifier 평가

Agentic workflow는 single-turn baseline을 확장하거나 대체하지 않습니다.
`multi_turn_agent/`, `configs/multi_turn_multi_agent_zero_shot.json`,
`outputs/multi_turn/`을 사용하는 별도 실험입니다. 두 방식은 공통 Spider
loader, read-only SQL executor와 공식 evaluator만 공유합니다. Agentic run의
source contract은 `config.py`, `core/`, `multi_turn_agent/`만 포함하므로
`single_turn/`만 수정해도 기존 agentic run의 source hash는 바뀌지 않습니다.

### 역할과 반복 계약

한 iteration은 반드시 `planner → coder → SQL 실행 → verifier` 순서로 진행되며,
문제당 최대 3회입니다.

| 역할 | 고정 checkpoint | 기본 physical GPU pool | output 한도 | metadata |
|---|---|---:|---:|---|
| Planner | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | `0,1,2` | 512 tokens | trainable |
| Coder | `Qwen/Qwen2.5-Coder-0.5B-Instruct` @ `ea3f2471cf1b1f0db85067f1ef93848e38e88c25` | `3,4` | 512 tokens | trainable |
| Verifier | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | `5,6,7` | 384 tokens | frozen |

Planner와 Verifier는 같은 초기 checkpoint를 사용하지만 config, process와 GPU
model instance는 분리됩니다. `trainable`/`frozen`은 향후 실험을 위한 역할
metadata일 뿐이며, 현재 코드에는 fine-tuning이나 reward 계산이 없습니다. 다만
각 역할의 `model.source`를 `local`로 바꾸어 외부에서 SFT한 전체 checkpoint로
교체할 수 있습니다. 세 역할 모두 FP32, eager attention, pure greedy decoding,
batch size 1, 입력 8,192 tokens, generation soft limit 120초로 고정됩니다.

예를 들어 Coder만 SFT 모델로 교체하려면 `configs/multi_turn_multi_agent_zero_shot.json`을
복사하고 `roles.coder.model`의 `source`, `id`, `revision`만 각각 `local`, 로컬
checkpoint 경로, `local`로 바꿉니다. Planner와 Verifier는 기존 Hub checkpoint를
그대로 사용할 수 있습니다. CLI에서 모델 경로를 override하지 않으므로 실험마다
사용한 config가 명시적으로 남습니다.

모든 role prompt에는 `Iteration N of 3`이 명시됩니다. Planner는 바로 답할 수
있는 `direct` 문제인지 수정 과정이 필요한 `iterative` 문제인지 판단하지만 SQL
자체를 최종 답으로 내지 않습니다. `direct`여도 Coder와 Verifier는 생략하지
않습니다. 2회차부터는 이전의 모든 plan, SQL, bounded 실행 관측과 verifier
feedback을 Planner에 누적 전달합니다.

Planner는 plain JSON 또는 JSON code fence 하나로 다음 schema를 반환해야 합니다.

```json
{
  "iteration": 1,
  "approach": "direct",
  "plan": ["step"],
  "coder_instruction": "instruction"
}
```

Coder는 원 질문, schema, 현재 Planner JSON과 iteration을 받아 읽기 전용 SQLite
query 하나만 반환합니다. 그 SQL을 기존 안전 executor로 실제 실행한 뒤
Verifier에는 다음처럼 제한한 관측만 전달합니다.

- 실행 status와 오류
- column, 전체 row count와 앞 5개 row
- 최대 4 KiB의 JSON-safe 결과
- tool query 실행시간과 truncation 여부

Verifier도 plain JSON 또는 JSON code fence 하나로 고정 schema를 반환합니다.

```json
{
  "iteration": 1,
  "decision": "continue",
  "reason": "assessment",
  "feedback": "specific feedback for the next planner"
}
```

`decision`은 `stop|continue`만 허용하며 `continue`이면 feedback이 비어 있으면
안 됩니다. 실행 오류가 있어도 Verifier가 `stop`을 선택하면 orchestration이
결정을 덮어쓰지 않습니다. 해당 SQL은 loop가 끝난 뒤 오류 또는 0점으로
자연스럽게 평가됩니다. 3회차 `continue`는 `max_iterations_reached`로 종료하고
마지막 후보를 평가합니다. Planner/Verifier JSON schema 오류에는 별도 repair
generation을 호출하지 않으며, 가능한 마지막 후보를 최종 평가합니다. Coder의
SQL parse/실행 오류는 retry하지 않고 그대로 Verifier에게 전달합니다.

Gold SQL과 공식 evaluator 결과는 role prompt와 role worker에 전달되지 않습니다.
Agent loop가 끝난 뒤에만 부모 평가 단계가 final SQL, gold SQL과 두 공식 metric을
처리합니다.

### GPU process와 doctor

부모 coordinator는 model library를 import하지 않습니다. 기본 설정은 8개의
long-lived role subprocess를 만들고, 각 worker에 physical GPU 하나를
`CUDA_VISIBLE_DEVICES`로 격리합니다. 따라서 worker 내부 device는 항상
`cuda:0`입니다. Role pool 사이 GPU 중복은 config load와 CLI override 단계에서
거부합니다. Model은 GPU별로 한 worker씩 순차 로드하고, 준비된 worker들은 여러
episode를 비동기 stage queue로 처리합니다. Worker당 outstanding role task는
하나이며 SQL tool 실행은 최대 2개까지만 동시에 수행합니다.

모델을 다운로드하거나 로드하기 전에 8개 GPU를 한 번에 진단합니다.

```bash
python -m text2sql agent-doctor \
  --config configs/multi_turn_multi_agent_zero_shot.json
```

Planner/Verifier GPU에는 8 GiB, Coder GPU에는 4 GiB 이상의 free VRAM이 필요합니다.
Doctor는 role별 model source/ID, Hub revision 또는 local fingerprint, physical
GPU, logical `cuda:0`, FP32 CUDA probe와 free VRAM을 보고하지만 모델 자체는
로드하지 않습니다. 다른 GPU 배치를
시험할 때만 세 pool을 함께 override합니다.

```bash
python -m text2sql agent-doctor \
  --config configs/multi_turn_multi_agent_zero_shot.json \
  --planner-gpus 0,1,2 \
  --coder-gpus 3,4 \
  --verifier-gpus 5,6,7
```

명령 앞에 전체 GPU를 노출하거나 감추는 별도 `CUDA_VISIBLE_DEVICES`를 붙이지
않습니다. 부모가 각 physical index를 worker 환경에 직접 매핑합니다.

### 로컬 mock smoke

로컬에서는 다음 명령으로 모델·CUDA·network 없이 8개 전체 orchestration과
checkpoint, SQL 안전 실행, 공식 평가 artifact를 검증합니다.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
.venv/bin/python -m text2sql agent-evaluate \
  --config configs/multi_turn_multi_agent_zero_shot.json \
  --backend mock \
  --selection smoke \
  --run-name local-agent-smoke
```

Mock은 고정된 Planner JSON, Coder SQL과 Verifier 결정을 반환하는 배관
검증용입니다. Gold SQL은 mock role response에도 들어가지 않습니다. 따라서 mock
정확도를 모델 성능으로 해석하지 않으며, agentic smoke에는 최소 정확도 통과
기준을 두지 않습니다. 8개 모두 terminal trajectory가 되고 두 공식 metric이
유효하게 분류되며 timing과 DB hash 불변이 확인되는지가 핵심입니다.

### 서버 smoke와 전체 dev 실행

첫 서버 실행에서만 Hugging Face 다운로드를 명시적으로 허용합니다. 아래 예시는
설정 파일의 기본 3/2/3 GPU pool을 사용합니다.

```bash
timeout --signal=TERM --kill-after=30s 4h \
python -m text2sql agent-evaluate \
  --config configs/multi_turn_multi_agent_zero_shot.json \
  --backend hf \
  --selection smoke \
  --run-name qwen-agent-server-smoke \
  --allow-model-download
```

8개 모두 최대 3 iteration 안에서 terminal 상태가 되고 `pipeline_pass=true`, 공식
Exact/Test Suite metric `valid=true`, final SQL timing 생성, DB hash 불변이
확인되어야 합니다. 정확도 값 자체에는 smoke 최소 기준이 없습니다. Model cache가
준비된 다음 run부터는 `--allow-model-download`를 제거합니다.

전체 dev 1,034개는 train이나 fine-tuning 없이 다음처럼 평가합니다. TITAN Xp의
실제 처리량에 맞춰 외부 supervisor 시간은 조정할 수 있습니다.

```bash
timeout --signal=TERM --kill-after=30s 24h \
python -m text2sql agent-evaluate \
  --config configs/multi_turn_multi_agent_zero_shot.json \
  --backend hf \
  --selection all \
  --run-name qwen-agent-zero-shot-dev
```

Role subprocess/process 장애는 해당 stage에서 worker를 한 번만 재시작하고 같은
task를 infrastructure retry합니다. 모델이 반환한 JSON/SQL 오류는 infrastructure
장애가 아니므로 retry하지 않습니다. 각 stage 완료 직후 episode checkpoint를
atomic하게 기록하며, 중단된 run은 다음처럼 재개합니다.

Coder generation·SQL parsing과 tool SQL 실행은 서로 다른 checkpoint stage입니다.
따라서 tool 실행 중 중단되어도 완료된 Coder 호출은 반복하지 않습니다. 부모가
`SIGTERM`/`SIGINT`를 받으면 모든 role worker를 먼저 종료하며, Linux worker에는
parent-death signal도 설정되어 부모가 강제로 사라진 경우 GPU process가 남지 않게
합니다.

```bash
timeout --signal=TERM --kill-after=30s 24h \
python -m text2sql agent-evaluate \
  --config configs/multi_turn_multi_agent_zero_shot.json \
  --backend hf \
  --selection all \
  --resume-run qwen-agent-zero-shot-dev
```

재개는 완료된 LLM 호출을 반복하지 않습니다. Source tree, 실효 config, prompt와
schema contract 또는 dataset hash가 달라졌으면 서로 다른 실험 결과를 섞지 않고
거부합니다. 비동기 완료 순서와 관계없이 최종 trajectory와 record는 원래 Spider
index 순서로 병합됩니다.

### Agentic artifact와 실행시간

Agentic run은 `outputs/multi_turn/<run_id>/`에 다음을 저장합니다.

- `trajectories.jsonl`: iteration별 raw/parsed role output, candidate SQL, bounded
  실행 관측, Verifier 결정과 role prompt hash
- `records.jsonl`: final SQL, 공식 Exact/Test Suite 판정, 원본 DB 로컬 진단 실행과
  최종 SQL 실행시간
- `run_manifest.json`: 실효 config, contract hash, model revision, GPU/worker 상태,
  source·dataset hash와 실행 상태. 원본 config나 상세 worker metadata처럼 다른
  artifact와 중복되는 내용은 저장하지 않습니다.
- `summary.json`: 공통 최종 metric 5개와 multi-turn 전용 비용 metric
  `mean_iterations_used`, `mean_cumulative_tool_vm_steps_lower_bound`,
  `mean_cumulative_tool_query_latency_ms`를 저장
- run 내부 episode checkpoint: stage-level resume를 위한 원자적 중간 상태

전체 prompt message 본문과 per-record worker/GPU 정보는 저장하지 않습니다. GPU
배치와 model revision은 run-level manifest에서 확인합니다.

각 iteration에서 Verifier가 보는 tool 실행의 `query_elapsed_ns`와 loop 종료 뒤
final SQL을 새로 실행한 `predicted_execution.query_elapsed_ns`를 모두 보존합니다.
Single-turn과 비교하거나 향후 RL 입력 후보로 사용할 primary timing은 후자입니다.
Multi-turn 전용 누적 metric은 문제마다 iteration 내부 tool 실행값을 먼저 합산한
뒤 전체 문제에 대해 평균을 냅니다. 평가용 final SQL 재실행, gold SQL 실행과 공식
evaluator 시간은 이 누적값에 포함하지 않습니다.
다만 agent loop가 같은 원본 DB에서 후보 SQL을 먼저 실행하므로 OS/SQLite cache가
final 재실행 전에 warm-up될 수 있습니다. 이 cache 정책을 적용하므로,
이 값은 엄격한 cold-cache latency로 해석하지 않습니다. 현재 범위에는 이 원시
SQL 실행시간을 reward로 결합하는 공식이나 가중치가 포함되지 않습니다.

## Single-turn Artifact

Single-turn smoke run과 전체 평가는 모두 `outputs/single_turn/<run_id>/`에
저장됩니다. Multi-turn agent 평가는 `outputs/multi_turn/<run_id>/`에 저장됩니다.

- `run_manifest.json`: 실효 설정, 소스 tree/data hash, runtime, model revision과
  실행 상태. 중복되는 원본 config, per-file source hash와 상세 preflight 내용은
  제외합니다.
- `records.jsonl`: 문제 식별 정보, LLM raw output, 추출 SQL, 두 공식 정확도
  판정, 로컬 진단 결과, SQL 실행 결과와 timing
- `summary.json`: `test_suite_accuracy`, `exact_set_match_accuracy`,
  `result_match_accuracy`, `mean_vm_steps`, `mean_latency_ms`만 저장

전체 평가에는 다음 중간 artifact도 있습니다.

- `shards/attempt-NNN/`: worker assignment, 즉시 flush되는 compact generation JSONL,
  worker status와 log. Worker/GPU 배치는 이 run-level status에만 남습니다.
- `generation_records.jsonl`: shard를 원래 Spider index 순서로 병합한 결과.
  각 record에는 `example_id`, split/index, `db_id`, question, prompt hash,
  generation 상태·소요시간·raw output·오류, SQL parsing 결과만 저장합니다.
  Prompt message 본문, input/output token 수, model/revision, worker/GPU는 중복
  저장하지 않습니다. Model/revision과 worker/GPU 정보는
  `run_manifest.json`과 worker status에서 run 단위로 확인할 수 있습니다.
- `records.jsonl`: 순차 SQL 실행과 공식 평가 결과까지 결합한 최종 record

Model backend 오류를 빈 결과로 대체하거나 실패한 SQL을 자동 수정하지 않습니다.
초기 model/tokenizer load가 실패해도 빈 디렉터리만 남기지 않고 failed manifest,
빈 `records.jsonl`, 모든 metric이 `null`인 실패 `summary.json`을 기록합니다.
Manifest의 `config`는 CLI override까지 반영한 유일한 실효 설정 snapshot이며,
원본 설정은 본문 대신 경로와 SHA-256만 보존합니다. 재실행에 필요한 비민감 CLI
인자는 정규화된 `invocation` 항목에 기록합니다.

실행시간 필드는 다음 경계를 구분합니다.

- `query_elapsed_ns`: SQLite `execute()` 직전부터 결과 fetch 완료까지
- `worker_elapsed_ns`: 격리 worker 진입부터 결과 구성 완료까지
- `parent_elapsed_ns`: subprocess 시작부터 IPC 수신과 종료 확인까지

같은 실행에서 SQLite VM 작업량도 기록합니다. 현재 표준 `sqlite3`가
`sqlite3_stmt_status()`를 노출하지 않으므로 기존 timeout progress handler를
재사용한 1,000-step 단위의 범위 측정입니다.

- `vm_steps_lower_bound`: 관찰한 최소 VM operation 수
- `vm_steps_upper_bound_exclusive`: 완료된 SQL의 배타적 상한
- `vm_step_progress_interval`: 현재 `1000`
- `vm_step_measurement_complete`: SQL이 정상 완료되어 상한도 유효한지 여부

따라서 이 값은 exact `SQLITE_STMTSTATUS_VM_STEP`이 아니며 `mean_vm_steps`도 정상
완료된 prediction들의 하한값 평균입니다. Timeout과 engine-level result-limit은 중단 시점까지의 하한만
남깁니다. Agent
trajectory에는 측정값을 보존하지만 기존 실험의 입력 조건을 바꾸지 않도록
Planner와 Verifier prompt에는 전달하지 않습니다.

각 질문에서는 prediction을 먼저 실행한 뒤 gold를 실행합니다. 별도의 cache
reset이나 warm-up은 수행하지 않으며, gold 실행이 prediction 시간을 미리
warm-up하지 않게 순서를 고정합니다. 성공한 prediction의 `query_elapsed_ns`만
`mean_latency_ms`에 포함합니다. Timeout처럼 worker가
결과를 돌려준 경우에는 해당 질문 record에 관찰된 query 시간과 parent 시간을
남기되 성공 latency 통계에는 포함하지 않습니다.

향후 RL에 사용할 원시 실행시간 관측값은 prediction의 `query_elapsed_ns`입니다.
모델 생성시간, process 기동 오버헤드, gold SQL 시간 및 공식 evaluator
runtime은 이 값에 섞지 않습니다. 이 단계에서는 reward 결합식이나 가중치를
정의하지 않습니다.

`run_manifest.json`은 Python·OS·CPU architecture·CPU count·SQLite 버전을
기록합니다. 상세 backend/worker metadata는 single-turn의 `shards/` 또는
multi-turn의 `worker_runtime/`에 남깁니다. 실행시간을 서로 비교할 때에는 같은 서버와 가능한 한 유사한
시스템 부하 조건을 사용하고, 서버 준비 단계의 `nvidia-smi` 출력도 run과 함께
보존합니다.
