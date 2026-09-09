# Qwen3-1.7B zero-shot Spider test baselines

이 패키지는 tuning하지 않은 `Qwen/Qwen3-1.7B`를 Spider **test split만** 대상으로 평가합니다.
모델에는 question과 full schema만 전달하며 gold SQL은 generation이 끝난 뒤 CPU 평가에서만 사용합니다.

두 설정은 `rl_rrcm_style`의 prompt builder, action parser, chat template, SQL sandbox 및 Spider metric을
공유합니다. 차이는 intermediate budget뿐입니다.

- `single_turn`: `max_intermediate=0`, 모든 샘플에서 LLM을 정확히 한 번 호출하고 `<answer>`만 허용합니다.
- `multi_turn`: `max_intermediate=3`, base model이 `<answer>` 또는 `<intermediate>`를 자율적으로 선택합니다.

## 설치

저장소 루트에서 동일 환경에 두 패키지를 설치합니다.

```bash
pip install -e './rl_rrcm_style[spider]' -e ./baseline
```

기본 설정은 `local_files_only=true`이므로 Qwen3-1.7B가 로컬 Hugging Face cache에 있어야 합니다.
`model.adapter_name_or_path`에 PEFT LoRA adapter 디렉터리를 지정할 수 있습니다. 또는
`model.name_or_path` 자체를 adapter 디렉터리로 지정하면 `adapter_config.json`에서 base model을
찾습니다. adapter와 함께 저장된 tokenizer가 있으면 이를 우선 사용합니다.

## 실행

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 baseline-sql evaluate \
  --config baseline/configs/qwen3_1_7b_single_turn_test.json \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
  --output baseline/outputs/qwen3_1_7b_single_turn_test

CUDA_VISIBLE_DEVICES=4,5,6,7 baseline-sql evaluate \
  --config baseline/configs/qwen3_1_7b_multi_turn_test.json \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
  --output baseline/outputs/qwen3_1_7b_multi_turn_test
```

`cuda:N`은 `CUDA_VISIBLE_DEVICES` 안의 논리 번호입니다. `--limit N`은 고정 seed로 뽑은 test-only
smoke subset을 평가합니다. GPU별 generation shard는 즉시 저장되며 같은 명령을 다시 실행하면 누락된
shard만 생성합니다. 설정·데이터가 달라진 output directory는 재사용할 수 없습니다.

결과는 `summary.json`, `summary.csv`, `trajectories.jsonl`, `predictions.sql`, `gold.sql`,
`difficulty.csv`, `structural_metrics.json`에 저장됩니다. 원본 DB execution accuracy와 structural exact
match를 구분해 기록합니다.

`evaluation.test_suite_database_dir`은 기본적으로 null입니다. 경로를 지정하면 선택된 test DB마다
두 개 이상의 generated suite DB가 있는지 먼저 검사하며, 하나라도 빠지면 평가를 중단합니다. 현재
`data/spider_test_suite/database`는 full test DB를 포함하지 않으므로 test TSA 경로로 사용하면 안 됩니다.

이미 generation을 끝낸 결과는 모델을 다시 띄우지 않고 test-suite DB만 연결해 재채점할 수 있습니다.

```bash
baseline-sql rescore \
  --config baseline/configs/qwen3_1_7b_single_turn_test.json \
  --output baseline/outputs/qwen3_1_7b_single_turn_test \
  --test-suite-database-dir /path/to/full-test-suite/database
```

`multi_turn` 결과도 해당 config와 output 경로로 같은 명령을 실행합니다. 재채점은 기존 shard를 보존하고
리포트만 갱신하며, 실제 적용된 설정은 `rescore_config.json`에 기록합니다. 지정한 suite는 해당 run에
포함된 모든 DB에 대해 원본을 포함한 SQLite 파일이 두 개 이상 있어야 합니다.
