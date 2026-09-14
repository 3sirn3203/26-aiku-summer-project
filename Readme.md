# Learning Agentic Workflows for text2sql with Reinforcement Learning

📢 2026년 여름학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다.

## 소개

자연어 질의를 SQL로 옮기는 Text-to-SQL task는 자연어처리(NLP, Natural Language Processing) 분야에서 이미 오랫동안 연구되어 온 task이지만, 복잡한 자연어 질의를 작은 언어 모델(SLM, Small Language Model)이 정확하게 SQL로 옮기는 데에는 여전히 한계가 있습니다.
따라서 본 프로젝트에서는 SLM의 Text-to-SQL task 성능을 향상시키기 위해 Agent 구조를 설계하고, 이를 RL 기반 finetuning 하여 baseline 대비 task 성능을 향상시키는 것을 목표로 합니다.
이 프로젝트의 핵심 키워드는 다음과 같습니다.
- Agentic AI
- Text-to-SQL
- Reinforcement Learning

## 방법론

이 프로젝트에서 제안하는 핵심 방법론은 크게 두 가지 축으로 정리할 수 있습니다: (1) Agentic workflow 기반 SQL 생성 (2) Reinforcement Learning 기반 generalization 성능 향상.

### 1. Agentic workflow

복잡한 자연어 질의와 database schema를 이해한 다음, 이를 실행 가능한 SQL query로 한 번에 정확하게 변환하는 task는 SLM에게는 여전히 어려운 과제입니다. [[Overall Performance](#2-overall-performance)]
이를 개선하기 위해 본 프로젝트에서는 자연어 질의를 단계적으로 SQL로 변환하는 Agentic workflow를 설계합니다.
각 iteration에서 LLM agent는 현재 context만으로 정답 SQL을 생성할 수 있다고 확신하는지 판단하고, 다음 두 가지 action 중 하나를 선택합니다.
- `<Answer>`: Agent가 정답 SQL을 생성할 수 있다고 판단하면 최종 SQL query를 출력합니다. 이후 iteration을 종료하고, 생성된 query를 기준으로 정답 여부를 평가합니다.
- `<Intermediate>`: Agent가 정답 SQL을 생성할 수 있다고 확신하지 못하면 Intermediate query를 출력합니다. 이는 정답 query의 subquery이거나 문법 오류를 확인하기 위한 query일 수 있습니다. Intermediate query는 SQLite 환경에서 실행되며, 실행 결과는 다음 iteration의 context에 추가됩니다.

이를 수학적으로 아래와 같이 나타낼 수 있습니다.

Trajectory의 초기 상태 $s_0$는 사용자의 자연어 질의 $q$와 database schema $\mathcal{S}$로 구성됩니다.

$$s_0=\langle q, \mathcal{S} \rangle.$$

$t$번째 상태 $s_t$에서 policy $\pi_\theta$를 통해 action $a_t$를 샘플링합니다.

$$a_t\sim \pi_\theta(\cdot\mid s_t) \qquad \text{where }a_t\in\left\lbrace\texttt{Answer}(y_t), \texttt{Intermediate}(z_t)\right\rbrace$$

이때 $y_t$와 $z_t$는 각각 실행 가능한 SQL query를 의미합니다.
- $a_t=\texttt{Answer}(y_t)$이면 $y_t$가 최종 SQL query가 되면서 iteration이 종료됩니다.
- $a_t=\texttt{Intermediate}(z_t)$이면 $z_t$를 실행한 다음 iteration의 state $s_{t+1}$을 다음과 같이 업데이트합니다. 

$$s_{t+1}\leftarrow s_t\oplus z_t \oplus e_t \qquad \text{where }e_t \text{ is the execution result of the query }z_t.$$

Iteration을 통해 최종 trajectory $\tau$는 다음과 같이 구성됩니다.

$$\tau = (s_0, a_0, o_0, s_1, a_1, o_1,\cdots s_T, a_T) \qquad \text{where }a_T=\texttt{Answer}(y_T).$$

### 2. GRPO based Reinforcement Learning

위에서 정의한 agentic workflow를 state-action model로 정의하고, reinforcement learning을 수행할 수 있습니다.
이 프로젝트에서는 RL finetuning을 위해 Group Relative Policy Optimization(GRPO)를 사용합니다.

Agent가 생성한 최종 SQL query는 크게 두 가지 축으로 평가할 수 있습니다.
- Execution accuracy: Agent가 생성한 query $\hat{y}$와 정답 query $y^\star$의 실행 결과를 비교합니다. 두 query의 실행 결과가 같으면 정답으로 인정합니다.
- Exact match: Agent가 생성한 query $\hat{y}$와 정답 query $y^\star$가 구조적으로 동일한지 분석합니다. 두 query가 구조적으로 동일하다면 정답으로 인정합니다.

위에서 정의한 query 평가 방식을 통해서 각 trajectory의 reward 계산을 위한 수식을 아래와 같이 정의합니다.

$$
R(\tau)=
\begin{cases}
\mathbb{1}\!\left[\mathcal{E}(\hat{y})=\mathcal{E}(y^*)\right]
+ \alpha \cdot \mathbb{1}\!\left[\operatorname{Match}(\hat{y},y^*)=1\right]
- \beta \dfrac{m_\tau}{N_{\max}},
& \text{if } \hat{y} \text{ is executable}, \\
-\lambda,
& \text{otherwise}.
\end{cases}
$$

$\alpha$는 execution accuracy에 비해 exact match를 얼마나 더 중요하게 평가할지 결정하는 가중치이며, $\beta$는 trajectory 내 Intermediate query 수에 부과되는 penalty의 강도를 조절하는 계수입니다.

또한 GRPO의 objective function $\mathcal{J}_{\mathrm{GRPO}}(\theta)$는 다음과 같이 정의됩니다.
$$
\mathcal{J}_{\mathrm{GRPO}}(\theta)
=
\mathbb{E}_{q\sim\mathcal{D},\,\{\tau_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}}
\left[
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|\tau_i|}\sum_{k=1}^{|\tau_i|}
\left(
\min\!\left(
\rho_{i,k}(\theta)\hat{A}_i,
\operatorname{clip}\!\left(\rho_{i,k}(\theta),1-\epsilon,1+\epsilon\right)\hat{A}_i
\right)
-\eta D_{\mathrm{KL}}\!\left(\pi_\theta\,\|\,\pi_{\mathrm{ref}}\right)
\right)
\right],
$$

여기서 $\rho_{i,k}(\theta)=\dfrac{\pi_\theta(a_{i,k}\mid s_{i,k})}{\pi_{\theta_{\mathrm{old}}}(a_{i,k}\mid s_{i,k})}$이며, $\hat{A}_i$는 동일한 입력에서 샘플링한 group 내 reward를 정규화하여 계산한 advantage입니다. $\epsilon$은 clipping 범위를, $\eta$는 reference policy와의 KL divergence에 적용되는 계수를 나타냅니다.

## 실험 결과

### 1. Experimental Setup

#### Dataset

실험에는 cross-domain text-to-SQL benchmark인 Spider 1.0을 사용했습니다. Spider 1.0은 총 10,181개의 자연어 질의와 200개의 database로 구성됩니다.

| Split | # of Questions | # of Databases |
| --- | ---: | ---: |
| Train | 7,000 | 146 |
| Validation | 1,034 | 20 |
| Test | 2,147 | 40 |

#### Metrics

- **Execution Accuracy**: 생성된 SQL과 정답 SQL을 원본 database에서 실행했을 때, 두 실행 결과가 일치하는 비율입니다.
- **Exact Match**: 생성된 SQL과 정답 SQL의 구조 및 구성 요소가 정확히 일치하는 비율입니다.
- **Mean LLM Calls**: 하나의 자연어 질의를 처리하는 데 필요한 평균 LLM 호출 횟수입니다.
- **Non Executable Ratio**: 생성된 SQL 중 실행할 수 없는 query가 차지하는 비율입니다.

#### Baselines

Agentic workflow와 fine-tuning의 효과를 비교하기 위해 다음 네 가지 baseline을 구성했습니다.

- Zero-shot & Single-turn
- Zero-shot & Multi-turn
- SFT & Single-turn
- SFT & Multi-turn

#### Implementation Details

| Configuration | Value |
| --- | --- |
| Backbone LLM | Qwen3-1.7B |
| Fine-tuning method | LoRA (rank $=16$, $\alpha=32$) |
| GRPO group size | $G=6$ |
| GRPO clip | $\epsilon=0.2$ |
| Sampling temperature | $T=0.7$ |
| Maximum Intermediate calls | $N_{\max}=3$ |
| Training steps | 1,000 |


### 2. Overall Performance

Baseline과 proposed method의 전체 성능은 아래 표와 같습니다.

| Method | Execution Accuracy | Exact Match | Mean LLM Calls | Non Executable Ratio |
| --- | ---: | ---: | ---: | ---: |
| Single-turn zero-shot | 0.556 | 0.264 | 1.0 | 0.127 |
| Multi-turn zero-shot | 0.537 | 0.256 | 2.490 | 0.088 |
| Single-turn SFT | 0.646 | 0.558 | 1.0 | 0.132 |
| Multi-turn SFT | 0.636 | 0.544 | 1.0 | 0.141 |
| Single-turn RL | 0.532 | 0.415 | 1.0 | 0.227 |
| **Multi-turn RL (Ours)** | **0.653** | **0.575** | 2.270 |**0.056** |

표의 결과를 통해 다음과 같은 사실을 확인할 수 있습니다.

- RL fine-tuning은 zero-shot 및 SFT보다 높은 일반화 성능을 보이며, 정확도와 실행 불가능 비율 모두에서 개선된 결과를 나타냅니다.
- RL을 통해 LLM은 최종 답변과 Intermediate query를 출력할 시점을 판단하는 방법을 학습합니다.
    - Multi-turn 방식의 평균 LLM 호출 횟수가 zero-shot의 2.490회에서 RL fine-tuning 후 2.270회로 감소하여 효율성이 향상되었습니다.
    - Multi-turn 적용에 따른 성능 향상은 RL fine-tuning에서만 관찰되었으며, zero-shot과 SFT에서는 오히려 성능이 저하되었습니다.

### 3. Hyperparameter Analysis

Intermediate query 생성에 부과되는 보상 페널티의 강도를 조절하는 하이퍼파라미터 $\beta$를 달리하여 학습한 결과는 다음과 같습니다.

<table>
  <tr>
    <td align="center">
      <img src="figures/beta_exact_match_validation.png" width="100%">
    </td>
    <td align="center">
      <img src="figures/beta_llm_calls_train.png" width="100%">
    </td>
  </tr>
</table>

그래프에서 확인할 수 있는 주요 결과는 다음과 같습니다.

- $\beta=0.0$으로 설정했을 때 전반적으로 더 높은 성능을 보입니다.
- $\beta=0.0$에서는 Intermediate query 수의 변동 폭이 크게 나타납니다. 이는 Intermediate query가 필요하지 않은 상황을 충분히 학습하지 못했음을 시사합니다.
- $\beta=0.2$에서는 학습 초기부터 Intermediate query 수가 빠르게 0으로 수렴합니다. 이는 GRPO 학습 과정에서 Intermediate query의 필요성이 과도하게 낮게 평가되는 경향이 있음을 보여줍니다.


## Contribution and Limitation

본 프로젝트의 주요 contribution은 다음과 같습니다.

- SLM 기반 Text-to-SQL task의 성능 한계를 실험적으로 확인.
- Agentic workflow를 도입하여 Text-to-SQL 성능을 개선.
- Intermediate query의 영향력을 별도로 modeling하지 않고도, RL을 통해 모델이 각 Intermediate query의 유효성을 스스로 학습하도록 learning objective 설계.

위 프로젝트의 limitation 및 future work는 다음과 같습니다.

- GRPO 학습에서 $\beta > 0$으로 설정하면 쉬운 query에 대한 Intermediate query의 유효성이 과소평가되어, Intermediate query 수가 빠르게 0으로 수렴하고 Agentic workflow가 충분히 작동하지 않는 한계.
- Agentic workflow를 도입하여 성능을 개선했지만, 추가적인 LLM 호출로 증가한 latency에 대한 분석은 충분히 이루어지지 않았음.
- 향후에는 Intermediate query가 적절한 수준으로 수렴하도록 reward와 GRPO objective를 개선하고, Agentic workflow의 성능 향상과 latency 증가 사이의 trade-off를 분석할 필요가 있음.

## 사용 방법

### 1. Repository 구조 설명

```text
.
├── data/
├── overall_pipeline/
│   ├── configs/
│   ├── outputs/
│   └── src/
├── rl/
│   ├── configs/
│   ├── outputs/
│   └── src/
└── sft/
    ├── configs/
    ├── outputs/
    └── src/
```

- `data/`: [Spider 1.0 공식 페이지](https://yale-lily.github.io/spider)에서 `spider_data.zip`을 내려받아 `data/`에 저장한 뒤 압축을 해제합니다.
- `overall_pipeline/`: Zero-shot 및 학습된 모델의 single-turn/multi-turn 평가를 실행합니다.
- `rl/`: Agentic workflow를 위한 GRPO 학습을 실행합니다.
- `sft/`: Text-to-SQL 모델의 supervised fine-tuning을 실행합니다.

### 2. 프로젝트 의존성 설명

본 프로젝트는 Python 3.11 이상에서 동작하며, 다음 패키지를 사용합니다.

```text
python >= 3.11
torch >= 2.3
transformers >= 4.51
peft >= 0.14
sqlparse >= 0.5
nltk >= 3.9
accelerate >= 0.34
psutil >= 5.9
wandb >= 0.17
```

- `torch`: SFT 및 GRPO 학습과 GPU 기반 추론을 수행합니다.
- `transformers`: Qwen3-1.7B 모델과 tokenizer를 불러오고 text generation을 수행합니다.
- `peft`: LoRA adapter를 구성하고 fine-tuning에 필요한 parameter를 관리합니다.
- `sqlparse`: 생성된 SQL query를 parsing하고 형식을 처리합니다.
- `nltk`: Spider evaluation 과정에서 SQL tokenization을 지원합니다.
- `accelerate`: multi-GPU 학습과 분산 실행을 지원합니다.
- `psutil`: 학습 및 평가 과정의 system resource를 확인합니다.
- `wandb`: 학습 과정의 metric과 실험 결과를 기록합니다.

저장소의 root directory에서 다음 명령어를 실행하면 프로젝트와 필수 패키지가 함께 설치됩니다.

```bash
pip install -e .
```

### 3. Baseline 및 방법론 실행

본 프로젝트의 핵심 방법론에 대한 학습은 다음 명령어로 실행할 수 있습니다.

```bash
text2sql-rl train  \
--config rl/configs/qwen3_alpha1.0_beta0.0.json
```

Main baseline인 SFT 학습은 다음 명령어로 실행할 수 있습니다.

```bash
torchrun --nproc_per_node=8 -m sft.cli train \
--config sft/configs/qwen3_sft.json
```

Zero-shot LLM 또는 앞서 학습한 모델을 사용하여 Spider 1.0의 test set에 대한 평가를 수행합니다.
Config 파일에서 평가할 LLM과 single-turn 또는 multi-turn 추론 방식을 지정할 수 있습니다.

```bash
text2sql-eval evaluate \
--config overall_pipeline/configs/zero_shot_single_turn.json \
--output overall_pipeline/outputs/zero_shot_single_turn \
--devices cuda:0
```
