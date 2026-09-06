# Verifier SFT 구현 계획

## 1. 목표와 고정 조건

목표는 현재 multi-agent 파이프라인의 verifier를 다음 두 가지 능력에 맞춰 SFT하는 것이다.

1. 후보 SQL이 질문을 충족하는지 판단하여 `stop` 또는 `continue`를 안정적으로 출력한다.
2. `continue`인 경우 다음 planner가 실제로 사용할 수 있는 구체적인 수정 방향을 제공한다.

학습된 verifier의 출력 계약은 현재 파이프라인과 동일하게 유지한다.

```json
{"decision":"continue","feedback":"후보 쿼리는 전체 직원을 세고 있으므로 활성 직원만 포함하도록 조건을 추가해야 한다."}
```

- 출력 필드는 `decision`, `feedback` 두 개뿐이다.
- `feedback`은 두 decision 모두에서 비어 있지 않아야 한다.
- iteration 번호는 runner가 관리하며 모델 출력에 포함하지 않는다.
- coder와 planner는 verifier 학습 중 고정한다.
- gold SQL과 공식 평가 결과는 라벨 생성에만 사용하고 verifier 입력에는 절대 포함하지 않는다.

첫 대상 모델은 현재 verifier와 같은 `Qwen/Qwen2.5-1.5B-Instruct`로 한다. 실제 배포 시 사용하는 planner 및 SFT coder 체크포인트의 경로와 revision을 데이터 manifest에 고정한다.

## 2. 데이터 원천

학습에는 Spider train만 사용한다.

- `data/spider_data/train_spider.json`
- `data/spider_data/train_others.json`
- `data/spider_data/tables.json`
- `data/spider_data/database/`
- `data/spider_test_suite/database/`

Spider dev 1,034개는 최종 비교 평가 전용으로 남겨 둔다. 데이터 분할은 질문 단위가 아니라 `db_id` 단위로 수행하여 같은 스키마가 train과 validation에 동시에 들어가지 않게 한다. 고정 seed와 DB 목록을 manifest에 기록한다.

각 학습 질문에 대해 다음 후보를 수집한다.

### 2.1 실제 pipeline 후보

배포 환경과 동일한 planner → SFT coder 경로로 후보 SQL을 만든다. verifier가 실제로 보게 될 분포를 재현하기 위해 각 레코드에 다음을 보존한다.

- question과 serialized schema
- planner output
- coder의 raw output과 추출된 SQL
- SQL parse 상태
- bounded execution observation
- 모델 ID, revision, generation config, prompt hash

질문마다 최소한 greedy 후보 1개를 생성한다. 데이터 다양성이 부족하면 sampling 후보를 3~4개 추가하되, temperature와 seed를 manifest에 기록한다. 완전히 동일한 SQL은 정규화 후 제거한다.

### 2.2 Gold positive

각 질문의 gold SQL을 `stop` positive로 포함한다. Gold만으로 positive를 구성하면 실제 coder 출력 형식과 지나치게 달라질 수 있으므로, test-suite를 통과한 coder 후보도 별도의 positive로 유지한다.

### 2.3 실행 가능한 hard negative

Gold SQL에 다음과 같은 한 가지 오류를 주입하여 hard negative를 만든다.

- WHERE 조건 제거 또는 비교 연산 변경
- JOIN 조건이나 JOIN 대상 변경
- 집계 함수 변경 또는 집계 누락
- GROUP BY, DISTINCT, ORDER BY, LIMIT 중 하나 제거 또는 변경
- SELECT 대상 컬럼 변경
- 중첩 질의의 범위나 연결 조건 변경

변형 결과가 실제로 test-suite에서 실패한 경우에만 negative로 채택한다. 문자열 치환만으로 SQL을 훼손하지 않고 Spider SQL parser가 식별할 수 있는 clause 단위 변형부터 구현한다.

문법 오류와 존재하지 않는 테이블·컬럼 오류도 포함하되 전체 negative의 20% 이하로 제한한다. verifier의 핵심 학습 대상은 실행 가능한 의미적 오답이다.

## 3. 정답 판정

후보의 `decision` 라벨은 고정된 공식 Spider test-suite evaluator로 생성한다.

1. 읽기 전용 단일 SQL인지 검사한다.
2. 후보 SQL과 gold SQL을 test-suite DB들에서 평가한다.
3. 모든 공식 판정이 일치하면 `stop`으로 라벨링한다.
4. test-suite mismatch, parse 오류 또는 실행 오류이면 `continue`로 라벨링한다.
5. evaluator infrastructure error, gold 실행 오류, 지원 DB 누락처럼 신뢰할 수 없는 경우는 학습 데이터에서 제외한다.

원본 DB 한 개에서 결과가 같다는 이유만으로 `stop` 라벨을 만들지 않는다. 우연히 같은 결과가 나온 잘못된 SQL이 positive에 섞일 수 있기 때문이다. Exact-set-match는 진단 정보로 저장하되, 의미적으로 동등한 다른 SQL을 오답 처리할 수 있으므로 주 라벨로 사용하지 않는다.

각 레코드에는 라벨 근거를 저장한다.

```json
{
  "example_id": "train:42:candidate:2",
  "db_id": "example_db",
  "source": "coder_sample|gold|gold_mutation",
  "question": "...",
  "serialized_schema": "...",
  "planner_output": {"plan": "..."},
  "candidate_raw_output": "...",
  "candidate_sql": "...",
  "sql_parsing": {"status": "success"},
  "execution_observation": {"status": "success", "rows": [], "truncated": false},
  "label": {"decision": "continue", "feedback": "..."},
  "label_evidence": {
    "test_suite_match": false,
    "exact_set_match": false,
    "error_category": "missing_filter"
  }
}
```

`gold_sql`은 별도의 private labeling artifact에만 저장한다. 최종 학습 JSONL과 렌더링된 user prompt에는 정답 참조 필드로 넣지 않는다. 단, gold-positive 레코드에서는 검증 대상인 candidate SQL 자체가 gold SQL과 같은 것이 정상이다.

## 4. Feedback 생성

초기 버전은 신뢰도가 높은 feedback만 사용한다.

### 4.1 규칙으로 확정할 수 있는 경우

- SQL parse 실패: 한 개의 완전한 읽기 전용 SQL을 만들도록 안내
- 존재하지 않는 테이블·컬럼: 실행 오류에 나타난 대상과 스키마를 근거로 수정 안내
- 명확한 단일 clause 변형: 제거하거나 변경한 필터, JOIN, 집계, 정렬 등을 복구하도록 안내
- 정답 후보: 질문의 필터·집계·출력 조건을 충족하므로 승인한다는 짧은 설명

### 4.2 의미적 차이가 복잡한 경우

Gold SQL과 후보 SQL의 parsed structure를 비교하여 차이를 요약한다. 하나의 명확한 오류 범주로 설명할 수 있을 때만 template feedback을 만든다. 여러 clause가 동시에 다르거나 equivalent rewrite 가능성이 있으면 다음 중 하나를 택한다.

- 더 큰 teacher 모델에 question, schema, candidate, gold를 제공하여 feedback 초안을 만든다.
- teacher 출력이 gold 구조와 일치하는 수정 사항을 언급하는지 규칙으로 검증한다.
- 검증할 수 없으면 해당 레코드는 decision-only 평가 데이터로 남기고 SFT에서는 제외한다.

Teacher가 본 gold SQL은 target feedback을 만드는 데만 사용한다. Feedback에는 완성된 gold SQL을 그대로 복사하지 않고, 누락되거나 잘못된 요구사항과 수정할 clause를 자연어로 기술한다.

## 5. 데이터 품질과 균형

최종 SFT 데이터는 `stop:continue`를 약 1:1로 맞춘다. `continue` 내부에서도 다음 범주가 한 종류에 편중되지 않게 제한한다.

- executable semantic error
- schema/execution error
- SQL parse/format error
- aggregation/grouping error
- filtering/value error
- join/subquery error
- projection/order/limit error

동일 질문에서 생성된 유사 후보가 데이터의 대부분을 차지하지 않도록 질문별·오류 범주별 최대 개수를 둔다. 완전 중복 prompt/target과 정규화 SQL 중복을 제거한다.

빌드 결과에는 다음 통계를 포함한다.

- split별 DB, 질문, 후보 수
- decision 비율
- candidate source별 비율
- error category별 비율
- parse/execution 상태
- test-suite 판정 상태
- prompt와 target token 길이 분포
- 제외된 레코드 수와 이유

## 6. SFT 방식

Titan Xp 12GB 환경을 기준으로 LoRA SFT를 우선 사용한다.

- base: `Qwen/Qwen2.5-1.5B-Instruct`
- dtype: FP16
- LoRA target: attention projection과 MLP projection
- 시작값: rank 16, alpha 32, dropout 0.05
- assistant 응답 토큰에만 loss 적용
- tokenizer chat template로 실제 system/user/assistant 메시지를 렌더링
- validation DB의 JSON contract validity와 decision balanced accuracy로 checkpoint 선택
- 2~3 epoch부터 시작하고 validation 악화 시 early stopping

입력 메시지는 `build_verifier_messages`와 하나의 공통 builder를 사용하여 학습과 실제 pipeline 사이의 prompt drift를 막는다. 학습 dataset에 이미 완성된 문자열 prompt를 영구 저장하기보다, 구조화 필드와 prompt-builder version/hash를 저장하고 학습 시 렌더링한다.

## 7. 평가 계획

### 7.1 고정 후보 offline 평가

Base verifier와 SFT verifier에 완전히 동일한 후보를 제공한다.

- JSON contract validity
- stop precision/recall
- continue precision/recall
- balanced accuracy와 macro F1
- 오답을 stop하는 false-stop rate
- 정답을 continue하는 false-continue rate
- 실행 가능한 오답만 대상으로 한 continue recall
- error category별 recall
- decision과 feedback의 모순율

현재 v2처럼 모든 후보를 `stop`하는 collapse가 발생하면 balanced accuracy와 false-stop rate에서 바로 드러나야 한다.

### 7.2 End-to-end multi-turn 평가

Spider dev에서 planner와 coder를 고정하고 verifier만 교체하여 비교한다.

- 공식 test-suite accuracy
- exact-set-match
- iteration 분포와 평균 iteration
- verifier 형식 오류 및 형식 재시도 횟수
- 1차 오답 → 최종 정답 전환 수
- 1차 정답 → 최종 오답 전환 수
- verifier `continue` 이후 coder 수정 성공률
- 평균 생성 시간과 SQL 실행 비용

Base와 SFT run은 동일한 예제 순서, 모델 revision, decoding 설정을 사용한다. 최종 판단은 정확도 향상을 우선하며 iteration 증가 자체를 성공으로 보지 않는다.

## 8. 구현 순서와 산출물

다음 순서로 구현한다.

1. `prepare_splits`: Spider train DB 단위 분할과 manifest 생성
2. `collect_candidates`: 실제 planner/coder 후보 및 gold positive 수집
3. `mutate_gold`: clause 단위 hard negative 생성
4. `label_candidates`: 공식 test-suite 판정과 불확실 레코드 제외
5. `build_feedback`: 규칙 기반 feedback 및 선택적 teacher feedback 생성
6. `build_sft_dataset`: 균형 조정, 중복 제거, 누수 검사, 통계 생성
7. `train_lora`: assistant-only LoRA SFT와 checkpoint 저장
8. `evaluate_offline`: 고정 후보 decision/feedback 평가
9. `evaluate_pipeline`: 기존 multi-agent runner에 adapter를 연결한 dev 평가

예상 디렉터리 구조는 다음과 같다.

```text
sft_verifier/
  PLAN.md
  configs/
  data/
    manifests/
    candidates/
    private_labels/
    sft/
  src/sft_verifier/
    prepare_splits.py
    collect_candidates.py
    mutate_gold.py
    label_candidates.py
    build_feedback.py
    build_sft_dataset.py
    train_lora.py
    evaluate_offline.py
  models/
  outputs/
  tests/
```

첫 milestone은 teacher 없이 만들 수 있는 고신뢰 subset으로 끝낸다. Gold positive, test-suite를 통과한 coder positive, 실행 오류 negative, 단일 clause mutation negative만 사용해 base verifier 대비 offline 성능을 확인한다. 이 단계에서 개선이 확인된 뒤 복잡한 semantic feedback과 teacher 생성 데이터를 추가한다.

## 9. 완료 기준

첫 SFT 모델은 다음 조건을 모두 만족해야 end-to-end 실험으로 진행한다.

- JSON contract validity 99% 이상
- validation에서 stop과 continue recall 모두 80% 이상
- executable semantic negative의 continue recall이 base보다 개선
- false-stop rate가 base보다 감소
- decision–feedback 모순율 2% 이하
- DB-disjoint validation에서 개선

최종 채택은 Spider dev test-suite accuracy가 base verifier보다 높고, 정답→오답 전환 증가 없이 오답→정답 수정이 늘어났을 때 결정한다.
