# 소형 언어 모델을 활용한 Agentic Text-to-SQL

## 1. 프로젝트 배경

복잡한 질의에 답하기 위해서는 한 번의 언어 모델 호출만으로 결과를 생성하는 방식보다 여러 단계의 중간 추론을 거치는 **multi-hop reasoning**이 효과적일 수 있습니다. Multi-hop reasoning에서는 각 단계의 결과가 다음 단계의 입력으로 활용되며, 최종 답변은 이러한 연속적인 추론의 결과로 생성됩니다.

이 과정에 **agentic workflow**를 적용하면, 사용자가 모든 추론 경로를 사전에 고정하지 않아도 언어 모델이 현재 상태를 분석하고 다음 행동을 선택하며 추론을 이어갈 수 있습니다. 현재 상태를 바탕으로 행동을 선택하고 그 결과에 따라 보상을 받는 구조는 강화학습의 패러다임과도 연결됩니다. 이에 따라 최근에는 agent의 행동 경로를 강화학습으로 fine-tuning하여 성능을 개선하려는 연구가 활발히 이루어지고 있습니다.

본 프로젝트는 이러한 두 가지 흐름, 즉 agentic workflow와 강화학습 기반 fine-tuning을 Text-to-SQL 과제에 적용하고 그 효과를 살펴보는 것을 목표로 합니다.

## 2. 문제 정의

Text-to-SQL은 사용자의 자연어 질의를 SQL로 변환하는 과제입니다. 모델은 질의를 이해하는 것뿐만 아니라 데이터베이스 스키마에서 관련 테이블과 컬럼을 식별하고, 적절한 조인 관계와 조건을 구성한 뒤, 문법적·의미적으로 올바른 SQL을 생성해야 합니다. 특히 여러 테이블의 관계를 파악하거나 중첩 질의, 집계, 정렬 등이 필요한 문제에서는 여러 단계의 추론과 오류 수정이 요구됩니다.

기존의 single-turn 방식에서는 질문 해석, 관련 테이블과 컬럼 탐색, 조인 경로 결정, SQL 작성, 결과 검증을 한 번의 응답 안에서 모두 수행해야 합니다. 또한 잘못된 SQL을 생성하더라도 이후의 상호작용을 통해 이를 검토하고 수정하기 어렵습니다. 이러한 한계는 단순한 질의보다 복잡한 추론이 필요한 질의에서 더욱 크게 나타납니다.

Text-to-SQL의 성능은 생성한 SQL의 정확성뿐만 아니라 실행 효율의 관점에서도 살펴볼 필요가 있습니다. 같은 질문에 대해 올바른 결과를 반환하는 SQL이라도 작성 방식에 따라 실제 실행시간이 달라질 수 있기 때문입니다. 따라서 본 프로젝트에서는 생성 SQL의 정확도와 실제 실행시간을 함께 측정하여 모델과 workflow의 성능을 평가하고자 합니다.

최근에는 agentic workflow를 활용하여 비교적 작은 언어 모델로도 복잡한 과제를 처리하려는 연구가 주목받고 있습니다. 본 프로젝트 역시 제한된 연산 자원을 고려하여 약 2B 규모의 소형 언어 모델과 adapter fine-tuning을 활용할 수 있는 문제를 탐색하며, 소형 모델에서도 일정 수준의 성능이 확인된 Text-to-SQL을 대상 과제로 삼습니다.

## 3. 프로젝트 목적

본 프로젝트의 핵심 목적은 복잡한 multi-hop reasoning이 필요한 Text-to-SQL 문제에서 소형 언어 모델 기반의 agentic workflow를 구성하고, 강화학습을 통해 그 성능을 개선할 수 있는지 확인하는 것입니다.

구체적으로 다음 세 요소가 Text-to-SQL 성능에 미치는 영향을 구분하여 분석하고자 합니다.

- Multi-turn reasoning
- Agentic workflow
- 강화학습 기반 fine-tuning

각 요소에 대한 ablation을 통해 성능 변화를 비교하며, zero-shot 언어 모델의 single-turn reasoning과 강화학습을 적용하지 않은 agentic workflow 등의 비교 기준보다 통계적으로 유의한 성능 향상을 달성하는 것을 목표로 합니다. 이러한 비교 기준을 넘어서는 성능을 프로젝트의 기본적인 성공 기준으로 삼습니다.

정확도는 Spider의 현재 공식 평가 기준인 **Test Suite Accuracy**와 기존의 **Original Exact Set Match**를 구분하여 확인하고, 생성 SQL의 실제 실행시간도 별도의 성능 지표로 함께 기록합니다. 향후 강화학습 기반 fine-tuning에서는 이렇게 측정한 정확도와 실행시간을 학습에 활용하는 것을 목표로 합니다.

이 프로젝트는 완전히 새로운 문제를 정의하기보다는 최근의 agentic workflow 및 강화학습 연구 흐름을 직접 탐구하고, 관련 방법론에 대한 이해와 경험을 쌓는 데에도 의의가 있습니다.

## 4. 데이터 및 평가 환경

주요 데이터셋 및 벤치마크로는 **Spider 1.0**을 고려합니다. Spider 1.0은 자연어 질문과 관계형 데이터베이스 스키마가 주어졌을 때 질문에 대응하는 SQL을 생성하는 대표적인 Text-to-SQL 벤치마크입니다.

Spider 1.0의 주요 특징은 다음과 같습니다.

- 10,181개의 자연어 질문, 5,693개의 고유 SQL 쿼리, 200개의 데이터베이스, 138개의 도메인으로 구성됩니다.
- 학습과 평가에 서로 다른 데이터베이스를 사용하는 cross-domain 설정을 채택하여, 모델이 특정 데이터베이스를 암기하는 방식으로 성능을 높이는 것을 방지합니다.
- 실제 SQLite 데이터베이스가 제공되므로, 생성한 SQL을 실행하고 그 결과를 평가하거나 피드백으로 활용할 수 있습니다.
- 문제별 난이도와 여러 평가 지표가 제공되어 모델의 성능을 다양한 관점에서 분석할 수 있습니다.

평가에서는 Spider의 Test Suite Accuracy와 Original Exact Set Match를 각각 산출하여 정확도를 확인합니다. 이와 함께 각 자연어 질문에 대해 생성된 SQL을 실제로 실행하고, 그 실행시간을 정확도와 함께 측정합니다. 두 정확도 지표와 실행시간은 서로 구분하여 기록하며, 정확도와 실행 효율을 함께 분석할 수 있는 평가 결과를 구성합니다.

데이터 분할은 다음과 같습니다.

| Split | 질문 수 |
| --- | ---: |
| Train | 7,000 |
| Development | 1,034 |
| Test | 2,147 |
| **합계** | **10,181** |

## 5. 관련 연구

### 연구 동기

**RRCM: Ranking-Driven Retrieval over Collaborative and Meta Memories for LLM Recommendation**은 필요한 외부 context를 검색하는 과정을 고정된 규칙으로 설계하는 대신, agentic workflow와 강화학습을 활용한다는 점에서 본 프로젝트의 주요 동기가 되었습니다. 해당 연구의 접근을 통해 retrieval뿐만 아니라 다른 복합 추론 과제에서도 agent의 행동 경로를 학습하는 방법을 탐구할 수 있다는 가능성에 주목했습니다.

### Text-to-SQL 관련 선행연구

현재 검토 대상으로 정리한 선행연구는 크게 agentic workflow를 도입한 연구와 강화학습으로 fine-tuning한 연구로 구분할 수 있습니다.

#### Agentic workflow 관련 연구

- MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL
- CHESS: Contextual Harnessing for Efficient SQL Synthesis
- Enhancing Text-to-SQL with Question Classification and Multi-Agent Collaboration
- SQL-of-Thought: Multi-agentic Text-to-SQL with Guided Error Correction
- Beyond Static Pipelines: Learning Dynamic Workflows for Text-to-SQL

#### 강화학습 관련 연구

- MARS-SQL: A Multi-Agent Reinforcement Learning Framework for Text-to-SQL
- MTIR-SQL: Multi-Turn Tool-Integrated Reasoning Reinforcement Learning for Text-to-SQL
- SQL-Trail: Multi-Turn Reinforcement Learning with Interleaved Feedback for Text-to-SQL
