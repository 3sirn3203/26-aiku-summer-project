# Adaptive Multi-step Text-to-SQL Agent on Spider

## 1. 연구 목표

Spider benchmark에서 하나의 LLM이 현재 상태만으로 최종 SQL을 생성할 수 있는지 스스로 판단하도록 학습한다.

모델은 매 단계에서 다음 두 행동 중 하나만 선택한다.

- `<answer>`: 최종 제출 SQL 생성
- `<intermediate>`: 추가 정보를 얻거나 가설을 검증하기 위한 중간 SQL 생성

핵심 가설은 모든 문제에서 고정된 횟수만큼 중간 추론을 수행하는 것보다, 문제별로 필요한 만큼만 intermediate SQL을 실행하는 adaptive agent가 정확도와 추론 효율성 사이에서 더 좋은 균형을 보인다는 것이다.

## 2. 기본 설정

- Benchmark: Spider 1.0
- Architecture: Single-LLM, multi-step agent
- 초기 입력:
  - Natural-language question
  - 해당 question과 연결된 database의 full schema
- Full schema에 포함할 정보:
  - Table names
  - Column names
  - Column types
  - Primary keys
  - Foreign-key relationships
- 초기 prompt에는 gold SQL, gold schema linking 정보와 sample row를 포함하지 않는다.
- Intermediate SQL을 실행하는 경우에만 실제 SQLite database content에 접근한다.

## 3. Action 형식

LLM은 매 단계에서 반드시 다음 두 형식 중 하나만 출력한다.

```text
<answer>
SELECT ...
</answer>
```

또는

```text
<intermediate>
SELECT ...
</intermediate>
```

자연어 reasoning이나 추가 설명은 출력하지 않는다. 각 action에는 정확히 하나의 SQL statement만 포함한다.

`<intermediate>`가 생성되면 SQL을 실행하고 다음 내용을 기존 trajectory에 추가한다.

```text
<intermediate>
{generated SQL}
</intermediate>
<response>
{column names and execution result}
</response>
```

이후 question, full schema와 누적된 trajectory를 입력으로 동일한 LLM을 다시 호출한다.

`<answer>`가 생성되면 trajectory를 종료하고 해당 SQL을 최종 예측으로 평가한다. `<answer>` SQL은 reward 계산을 위해 실행하지만, 실행 결과를 다시 모델에 제공하지 않는다.

## 4. 종료 조건

- `<answer>`가 생성되면 trajectory를 종료하고 해당 SQL을 최종 예측으로 평가한다.
- Intermediate SQL은 최대 `N_max`회까지 허용한다.
- `N_max = 3`을 기본값으로 사용한다.
- `N_max`회 이후에는 `<answer>`만 생성할 수 있도록 prompt 또는 decoding constraint를 적용한다.
- 최대 횟수 이후에도 `<intermediate>`를 생성하면 최종 SQL이 없는 실행 불가능 trajectory로 처리한다.
- 동일하거나 의미상 같은 intermediate SQL을 반복하더라도 호출 횟수에 포함한다.
- `<answer>`가 출력되는 즉시 trajectory를 종료한다.

## 5. SQL 실행 환경

Intermediate SQL과 final SQL에는 다음 제한을 적용한다.

- `SELECT` 또는 읽기 전용 `WITH ... SELECT`만 허용
- 한 action당 하나의 statement만 허용
- 실행 시간 제한 적용
- 반환 row 수와 전체 문자 수 제한
- 결과가 길면 일부 row와 전체 row 수만 반환
- SQL 오류가 발생하면 정규화된 오류 메시지를 `<response>`로 제공
- Intermediate SQL의 실행 성공 여부와 관계없이 호출 횟수에 포함

Intermediate 결과에는 column name을 포함한다. DB의 문자열 값이 prompt의 태그 구조를 깨뜨리지 않도록 escape 또는 structured serialization을 적용한다.

## 6. Trajectory 결과 분류

완성된 trajectory는 최종 SQL을 기준으로 다음 세 가지로 분류한다.

### 6.1 Correct

최종 SQL이 정상적으로 실행되고 Spider evaluator에서 정답으로 판정된 경우다.

가능하면 Test-Suite Accuracy를 이용해 정답 여부를 결정한다. 학습 비용 문제로 Test-Suite Accuracy 적용이 어렵다면 학습 중에는 execution correctness를 사용하고, validation 및 최종 평가에서는 Test-Suite Accuracy를 사용한다.

### 6.2 Executable but Incorrect

최종 SQL이 다음 조건을 만족하지만 정답으로 판정되지 않은 경우다.

- SQL parsing 성공
- 금지된 statement가 없음
- 제한 시간 안에 실행 완료
- DB에서 결과를 정상적으로 반환
- Spider evaluator에서는 오답으로 판정

결과가 빈 테이블이더라도 SQL 자체가 오류 없이 실행되었다면 이 범주에 포함한다. 단, gold SQL과 의미적으로 동일하여 정답으로 판정되면 Correct에 포함한다.

### 6.3 Non-executable

다음 중 하나라도 해당하면 실행 불가능한 SQL로 분류한다.

- SQL syntax error
- 존재하지 않는 table 또는 column 참조
- 잘못된 function 사용
- 실행 중 runtime error 발생
- 실행 시간 초과
- 여러 statement 출력
- 금지된 DDL 또는 DML statement 출력
- `<answer>` 태그 누락 또는 파싱 실패
- 최대 intermediate 횟수 이후에도 final SQL을 제출하지 않음

Intermediate SQL의 실행 실패는 trajectory를 즉시 종료시키지 않는다. 오류 메시지를 observation으로 제공하고 모델이 다음 단계에서 수정할 기회를 준다. 세 가지 결과 분류는 최종 `<answer>` SQL을 기준으로 결정한다.

## 7. Reward 설계

Reward는 다음 두 요소로 구성한다.

1. 최종 SQL의 결과 수준
2. Correct trajectory에서 사용한 intermediate 호출 횟수

변수는 다음과 같이 정의한다.

- `m`: 실제 intermediate 호출 횟수
- `N_max`: 최대 intermediate 호출 횟수
- `β`: 정답 trajectory의 효율성 보상 가중치
- `γ`: 실행 불가능한 SQL에 대한 penalty

기본 reward는 다음과 같이 정의한다.

\[
R =
\begin{cases}
1+\beta\left(\frac{N_{\max}-m}{N_{\max}}\right),
& \text{Correct}\\
0,
& \text{Executable but Incorrect}\\
-\gamma,
& \text{Non-executable}
\end{cases}
\]

초기 설정은 다음과 같이 사용한다.

```text
N_max = 3
β = 0.2
γ = 0.25
```

이 설정에서 reward 예시는 다음과 같다.

| 최종 결과 | Intermediate 횟수 | Reward |
|---|---:|---:|
| Correct | 0 | 1.200 |
| Correct | 1 | 1.133 |
| Correct | 2 | 1.067 |
| Correct | 3 | 1.000 |
| Executable but Incorrect | 무관 | 0.000 |
| Non-executable | 무관 | -0.250 |

Reward는 다음 우선순위를 보장한다.

1. 모든 Correct trajectory는 오답 trajectory보다 높은 reward를 받는다.
2. Correct trajectory 사이에서는 intermediate 호출이 적을수록 높은 reward를 받는다.
3. Executable but Incorrect trajectory는 Non-executable trajectory보다 높은 reward를 받는다.
4. 오답 trajectory 사이에서는 intermediate 횟수를 직접 최적화하지 않는다.

Executable but Incorrect trajectory에 양의 reward를 부여하지 않는 이유는 문법적으로 실행되지만 의미적으로 잘못된 SQL을 강화할 수 있기 때문이다. 대신 0점을 부여하여 Non-executable SQL보다는 우선하되 Correct SQL과는 명확히 구분한다.

Non-executable trajectory에는 음의 reward를 부여하여 SQL 문법, schema grounding과 기본 실행 가능성을 학습할 수 있도록 한다.

Intermediate 횟수에 대한 효율성 보상은 Correct trajectory에만 적용한다. 오답 trajectory에도 단계 penalty를 적용하면 학습 초기에 모델이 intermediate SQL을 통한 탐색을 포기하고 바로 실행 가능한 오답을 제출하는 premature stopping이 발생할 수 있다.

## 8. Reward 계산 절차

각 trajectory의 reward는 다음 순서로 계산한다.

1. 출력 형식과 `<answer>` 태그를 검사한다.
2. Final SQL이 읽기 전용 단일 statement인지 검사한다.
3. Final SQL을 제한된 SQLite 환경에서 실행한다.
4. 실행 실패 시 Non-executable로 분류한다.
5. 실행 성공 시 Spider evaluator로 정답 여부를 확인한다.
6. 정답이면 Correct로 분류하고 intermediate 횟수에 따른 효율성 보상을 계산한다.
7. 실행됐지만 정답이 아니면 Executable but Incorrect로 분류한다.

Reward 계산 코드는 모델 학습 코드와 분리하여 독립적으로 테스트할 수 있도록 구현한다.

다음 경계 사례에 대한 unit test를 작성한다.

- 정답 SQL
- 의미는 같지만 표면 형태가 다른 SQL
- 실행되지만 결과가 잘못된 SQL
- 빈 결과를 반환하는 SQL
- 존재하지 않는 column을 사용하는 SQL
- syntax error가 있는 SQL
- timeout이 발생하는 SQL
- 여러 SQL statement를 출력한 경우
- answer 태그가 없는 경우
- 최대 intermediate 횟수를 초과한 경우

## 9. Trajectory 생성

각 training question에 대해 동일한 초기 prompt에서 `G`개의 trajectory를 sampling한다.

초기 설정:

```text
G ∈ {4, 8}
temperature ∈ {0.7, 1.0}
N_max = 3
```

GPU 메모리와 학습 시간을 고려하여 group size와 최대 생성 길이를 조정할 수 있다.

각 trajectory에 대해 다음 정보를 저장한다.

- Question
- Database ID
- Full schema prompt
- 모든 action
- 모든 intermediate SQL
- Intermediate 실행 결과와 오류
- Final SQL
- Final SQL 결과 분류
- Intermediate 호출 횟수
- Reward
- 입력 및 출력 token 수
- SQL 실행 시간

하나의 group에 Correct trajectory가 없다면 다음 두 종류의 학습 신호만 발생할 수 있다.

- Executable but Incorrect와 Non-executable이 함께 존재하면 실행 가능성을 구분하는 신호가 발생한다.
- 모든 trajectory가 Executable but Incorrect이거나 모두 Non-executable이면 reward variance가 없어 group-relative advantage가 생성되지 않는다.

Reward가 모두 동일한 group은 policy update에서 제외한다. Correct sample이 지나치게 적으면 다음 방법을 적용한다.

- RL 이전에 gold SQL을 이용한 SFT warm-start 수행
- 제한된 횟수만큼 trajectory 추가 sampling
- 낮은 난이도의 문제부터 시작하는 curriculum 구성
- 초기에 높은 sampling temperature 사용

## 10. GRPO-style Fine-tuning

동일 question에서 생성된 trajectory를 하나의 group으로 구성한다.

- 각 trajectory의 reward 계산
- Group 내부 reward 정규화
- Group-relative advantage 계산
- Policy objective 계산
- Reference model에 대한 KL regularization 적용
- LoRA 또는 QLoRA 방식으로 parameter update
- Accuracy, reward, intermediate 횟수와 outcome category 비율 기록

학습 로그에는 최소한 다음 값을 포함한다.

- Correct trajectory 비율
- Executable but Incorrect 비율
- Non-executable 비율
- 평균 reward
- Correct trajectory의 평균 intermediate 횟수
- 전체 평균 intermediate 횟수
- Direct-answer 비율
- KL divergence
- All-equal reward group 비율

Optimizer, learning rate, batch size, rollout group size와 KL coefficient는 사용하는 base model과 GPU 환경을 확인한 뒤 결정한다.

## 11. 데이터 분할 및 누수 방지

- Spider train split만 RL 학습에 사용한다.
- 공식 train 전체로 학습하고 dev로 checkpoint 및 hyperparameter를 선택한다.
- Train/dev/test database가 서로 겹치지 않는지 검증한다.
- 공개된 Spider test set은 설정을 확정한 뒤 최종 성능 평가에 사용한다.
- Dev/test gold SQL은 평가 채점에만 사용하며 학습 reward나 생성 prompt에 사용하지 않는다.
- Training gold SQL은 reward 계산에만 사용하고 prompt에는 포함하지 않는다.
- Gold schema linking 정보도 모델 입력에 포함하지 않는다.
- 모든 schema serialization 순서와 형식을 학습과 평가에서 동일하게 유지한다.

## 12. 평가 지표

### 정확도 및 실행 가능성

- Test-Suite Accuracy
- Execution Accuracy
- Exact Match
- Correct 비율
- Executable but Incorrect 비율
- Non-executable 비율
- Spider difficulty별 정확도
  - Easy
  - Medium
  - Hard
  - Extra-hard

### 효율성

- 평균 intermediate 호출 횟수
- Correct trajectory의 평균 intermediate 횟수
- Intermediate를 사용하지 않은 비율
- 평균 LLM 호출 횟수
- 평균 token 사용량
- 평균 SQL 실행 시간
- Invalid intermediate SQL 비율
- Duplicate intermediate SQL 비율
- 최대 intermediate 횟수 도달 비율

### Adaptive behavior 분석

Spider difficulty별로 intermediate 사용 분포를 분석한다.

바람직한 결과는 다음과 같다.

- Easy 문제에서는 `<answer>`를 바로 출력하는 비율이 높아야 한다.
- Hard 및 Extra-hard 문제에서는 intermediate 사용 비율이 상대적으로 높아야 한다.
- Intermediate를 많이 사용하는 것 자체보다, 필요한 문제에서 사용했을 때 Correct로 전환되는지가 중요하다.
- Correct trajectory의 평균 intermediate 횟수가 불필요하게 증가하지 않아야 한다.

## 13. 주요 위험과 대응

### Premature stopping

단계 penalty로 인해 모델이 바로 실행 가능한 오답을 제출할 수 있다.

- Intermediate 효율성 보상은 Correct trajectory에만 적용한다.
- Executable but Incorrect는 단계 수와 관계없이 동일한 reward를 부여한다.
- SFT warm-start를 적용한다.

### Executability reward에 대한 과적합

모델이 의미적으로 올바른 SQL보다 단순히 실행되는 SQL을 생성하는 데 집중할 수 있다.

- Executable but Incorrect에는 양의 reward를 부여하지 않는다.
- Correct와 Executable but Incorrect 사이의 reward 차이를 충분히 크게 유지한다.
- 학습 로그에서 실행 가능성 증가와 정답률 증가를 별도로 확인한다.

### Reward hacking

한 번의 intermediate SQL에 지나치게 복잡한 탐색을 포함할 수 있다.

- 한 action당 하나의 statement만 허용한다.
- SQL 길이, 반환 row 수와 실행 시간을 기록한다.
- 실행 시간 및 결과 크기 제한을 적용한다.
- 단계 수 외 실제 계산 비용도 평가 지표로 보고한다.

### Spurious correctness

잘못된 SQL이 우연히 gold SQL과 같은 결과를 반환할 수 있다.

- 가능한 경우 Test-Suite Accuracy를 사용한다.
- Execution Accuracy와 Exact Match를 함께 보고한다.

### Repetitive query loop

동일하거나 의미 없는 intermediate SQL을 반복할 수 있다.

- 반복 호출도 intermediate 횟수에 포함한다.
- Duplicate-query rate를 기록한다.
- 최대 호출 횟수로 무한 반복을 방지한다.

### Context growth

Intermediate 결과가 누적되면서 context가 지나치게 길어질 수 있다.

- 반환 row 및 문자 수를 제한한다.
- 일관된 결과 serialization을 사용한다.
- 최대 context 길이를 초과하는 trajectory의 처리 규칙을 명시한다.

## 14. 구현 순서

1. Spider loader와 full-schema serializer 구현
2. 안전한 SQLite executor 구현
3. 두-action parser와 multi-step inference loop 구현
4. Final SQL outcome classifier 구현
5. Reward 함수와 경계 사례 unit test 작성
6. Trajectory sampling 및 logging 구현
7. SFT warm-start 구성
8. GRPO-style LoRA fine-tuning 구현
9. 공식 dev validation을 이용한 reward parameter 및 checkpoint 선택
10. Spider dev set 평가
11. Difficulty별 행동 및 효율성 분석
12. 실행 방법과 결과를 README에 문서화

## 15. 완료 조건

- 두 action만으로 end-to-end trajectory 생성이 가능해야 한다.
- 최대 intermediate 횟수가 확실하게 적용되어야 한다.
- DB를 변경하는 SQL이 실행되지 않아야 한다.
- Final SQL을 세 가지 outcome으로 안정적으로 분류해야 한다.
- Reward 함수가 각 outcome과 intermediate 횟수에 맞게 계산되어야 한다.
- 동일한 seed와 checkpoint로 평가 결과를 재현할 수 있어야 한다.
- Spider dev 결과와 difficulty별 행동 분석이 표 형태로 저장되어야 한다.
- Correct, Executable but Incorrect, Non-executable의 비율과 변화를 학습 과정에서 추적할 수 있어야 한다.
