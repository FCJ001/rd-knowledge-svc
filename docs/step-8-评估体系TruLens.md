# Step 8：评估体系（TruLens 2.x + 在线/离线评测 + Guardrails）

> **目标**：理解 TruLens 2.x 评估架构，在线/离线双模式评测，以及生产级质量保障体系。

---

## 8.1 架构总览

```
┌────────────────────────────────────────────────────────────────┐
│                         评估体系                                │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │  在线评测      │  │  离线评测     │  │  用户反馈             │ │
│  │  (async eval) │  │  (scripts)   │  │  (Feedback API)      │ │
│  │  采样率 10%    │  │  全量/多通道  │  │  👍 / 👎            │ │
│  │  fire&forget  │  │  TruLens     │  │  trace_id 关联       │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘ │
│         │                 │                      │              │
│         ▼                 ▼                      ▼              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  trulens_eval (PostgreSQL)                 │  │
│  │  online_scores | records | feedbacks | feedback_defs      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │  Guardrails   │  │  Token 追踪   │  │  Dashboard           │ │
│  │  SQL安全检查   │  │  LangChain    │  │  /eval (内建)         │ │
│  │  PII 输出检测  │  │  callbacks    │  │  :8501 (TruLens)    │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

---

## 8.2 文件清单

| 文件 | 职责 |
|------|------|
| `src/rag/evaluation/feedbacks.py` | TruLens 2.x Metric 定义（RAG Triad + NL2SQL） |
| `src/rag/evaluation/async_tracker.py` | 在线异步评估器，采样+打分+入库 |
| `src/rag/evaluation/trulens_config.py` | LiteLLM provider + TruSession + Dashboard 启动 |
| `src/rag/evaluation/tracked_rag.py` | RAG 检索通道追踪包装（离线评测用） |
| `src/rag/evaluation/tracked_nl2sql.py` | NL2SQL 追踪包装（离线评测用） |
| `src/rag/evaluation/nl2sql_metrics.py` | NL2SQL 即时指标（SQL可执行率/LLM裁判） |
| `src/rag/evaluation/guardrails.py` | SQL 安全检查 + PII 检测 |
| `src/rag/evaluation/token_tracker.py` | LangChain callback 累计 Token/费用 |
| `src/api/routers/eval.py` | 评测结果查询 API |
| `src/api/routers/feedback.py` | 用户反馈 CRUD API |
| `src/static/eval.html` | 评测仪表盘前端 |
| `scripts/run_rag_experiments.py` | RAG 离线评测（单/多通道） |
| `scripts/run_nl2sql_eval.py` | NL2SQL 离线评测（20道题） |

---

## 8.3 TruLens 2.x 核心变化

### 从 1.x 到 2.x

| 概念 | 1.x (旧) | 2.x (新) |
|------|----------|----------|
| 入口 | `TruCustomApp` / `TruChain` | `TruApp` |
| 指标 | `Feedback(provider.xxx)` | `Metric(implementation=provider.xxx)` |
| 选择器 | `.on_input().on_output()` | `Selector.select_record_input()` |
| Session | `Tru()` | `TruSession()` |
| Provider | `from trulens_eval import LiteLLM` | `from trulens.providers.litellm import LiteLLM` |

### RAG Triad 指标定义（Metric API）

```python
# src/rag/evaluation/feedbacks.py
from trulens.core.metric import Metric
from trulens.core.metric.selector import Selector
from trulens.providers.litellm import LiteLLM

def build_rag_triad_metrics(provider: LiteLLM) -> list[Metric]:
    m_answer_relevance = Metric(
        implementation=provider.relevance_with_cot_reasons,
        name="答案相关性",
        selectors={
            "prompt": Selector.select_record_input(),
            "response": Selector.select_record_output(),
        },
    )
    m_context_relevance = Metric(
        implementation=provider.context_relevance_with_cot_reasons,
        name="上下文相关性",
        selectors={
            "question": Selector.select_record_input(),
            "context": Selector.select_context(collect_list=False),
        },
        agg=np.mean,
    )
    m_groundedness = Metric(
        implementation=provider.groundedness_measure_with_cot_reasons,
        name="有据性",
        selectors={
            "source": Selector.select_context(collect_list=True),
            "statement": Selector.select_record_output(),
        },
    )
    return [m_answer_relevance, m_context_relevance, m_groundedness]
```

### NL2SQL 三指标

```python
def build_nl2sql_metrics(provider: LiteLLM) -> list[Metric]:
    # 1. SQL 可执行率（客观，不需要 LLM）
    m_sql_valid = Metric(implementation=_nl2sql_sql_valid, name="SQL可执行率", ...)
    # 2. 数据返回率（客观，有数据=1.0, 空结果=0.5, SQL失败=0.0）
    m_has_data = Metric(implementation=_nl2sql_has_data, name="数据返回率", ...)
    # 3. 结果相关性（LLM-as-Judge）
    m_relevance = Metric(implementation=provider.relevance_with_cot_reasons, name="结果相关性", ...)
    return [m_sql_valid, m_has_data, m_relevance]
```

### 标准生命周期

```python
# ★ 顺序很重要！
with tru_app as recording:
    output = await tracked.query(question)

session.force_flush()
tru_app.compute_feedbacks()     # ① 先计算反馈
tru_app.stop_evaluator()        # ② 再停止评估线程
session.force_flush()
leaderboard = session.get_leaderboard()
```

---

## 8.4 在线评测（生产环境）

### 设计思路

每次 API 请求返回后，**fire-and-forget** 触发 LLM-as-Judge 评分，不阻塞用户。按采样率随机触发，平衡成本与覆盖。

```
用户请求 → API返回 → 用户收到结果
                 ↘ (fire&forget, 采样率)
                   后台调 LLM 打分 → 写入 online_scores
```

### 配置

```python
# src/core/config.py
TRULENS_ENABLED: bool = True
EVAL_SAMPLE_RATE: float = 0.1    # 10% 采样率
```

设置 `EVAL_SAMPLE_RATE=1.0` 可 100% 采样（调试用）。设 `TRULENS_ENABLED=false` 完全关闭。

### AsyncEvaluator

```python
# src/rag/evaluation/async_tracker.py

class AsyncEvaluator:
    """单例，fire-and-forget 模式"""

    def evaluate(self, question, sql, data, summary, error, token_usage):
        """NL2SQL 三指标：SQL可执行率 + 数据返回率 + 结果相关性"""
        if random.random() > self.sample_rate:
            return  # 不采样
        asyncio.create_task(self._run(ctx))

    def evaluate_knowledge(self, question, answer, contexts, token_usage):
        """RAG Triad：答案相关性 + 上下文相关性 + 有据性"""
        ...
```

### API 接入

两个路由都接了评估器：

**知识检索**（`knowledge.py`）：
```python
# TokenTracker 包裹 LLM
token_tracker = TokenTracker()
llm = llm.with_config({"callbacks": [token_tracker]})

# 检索 + 融合
result = await multi_channel_search(...)
answer = result["answer"]
contexts = result["contexts"]

# Guardrails + 异步评估
check_output(answer)
evaluator.evaluate_knowledge(question, answer, contexts, token_tracker.usage)
```

**BI 查询**（`bi.py`）：同上模式，在 SSE 流结束后触发。
```python
evaluator.evaluate(question, sql, data, summary, error, token_tracker.usage)
```

---

## 8.5 离线评测

### NL2SQL 评测（20 道题）

```bash
python scripts/run_nl2sql_eval.py --list       # 列出题目
python scripts/run_nl2sql_eval.py --limit 3     # 快速验证
python scripts/run_nl2sql_eval.py               # 跑全部 20 题
```

三指标 × 20 题，TruLens 录制 + Dashboard 可查看。

### RAG 多通道对比

```bash
python scripts/run_rag_experiments.py --channel doc_rag   # 单通道
python scripts/run_rag_experiments.py                      # 全通道对比
python scripts/run_rag_experiments.py --dashboard          # Dashboard
```

每个通道作为独立 `app_version`，Leaderboard 直接对比 `doc_rag` vs `graph_rag` vs `fusion`。

---

## 8.6 Guardrails（输出安全保障）

**文件**：`src/rag/evaluation/guardrails.py`

### SQL 安全检查

```python
def check_sql(sql: str) -> tuple[bool, str]:
    # DROP TABLE / TRUNCATE / ALTER → BLOCK
    # DELETE / UPDATE 没有 WHERE → BLOCK
    # pg_read_file / dblink → BLOCK
```

在 BI 查询的 `execute_sql` 结果阶段调用，违反时 SSE 推送 guard 事件。

### 输出 PII 检测

```python
def check_output(text: str) -> tuple[bool, str]:
    # 手机号 / 身份证 / 邮箱
    # ≥5 个匹配才告警（避免误报）
```

### 配置

```python
# src/core/config.py
GUARDRAILS_ENABLED: bool = True
GUARDRAILS_BLOCK_DDL: bool = True
GUARDRAILS_BLOCK_DML_WITHOUT_WHERE: bool = True
```

---

## 8.7 Token 用量 & 费用追踪

**文件**：`src/rag/evaluation/token_tracker.py`

```python
class TokenTracker(BaseCallbackHandler):
    """LangChain 回调，累计每次 LLM 调用的 token 和费用"""

    def on_llm_end(self, response: LLMResult, **kwargs):
        # 优先读 response.llm_output["token_usage"]
        # 兜底读 usage_metadata

    @property
    def usage(self) -> dict:
        return {
            "input_tokens": ...,
            "output_tokens": ...,
            "total_tokens": ...,
            "cost_usd": ...,    # 按 MODEL_PRICING 计算
            "calls": ...,       # LLM 调用次数
        }
```

费用公式：
```
cost_usd = input_tokens × (MODEL_PRICING_INPUT / 1,000,000)
         + output_tokens × (MODEL_PRICING_OUTPUT / 1,000,000)
```

配置（qwen-max 价格）：
```python
MODEL_PRICING_INPUT: float = 0.4    # $0.4/1M tokens
MODEL_PRICING_OUTPUT: float = 1.2   # $1.2/1M tokens
```

---

## 8.8 用户反馈闭环

**API**：`src/api/routers/feedback.py`

| 端点 | 说明 |
|------|------|
| `POST /api/v1/feedback` | 提交反馈（rating: 1赞/-1踩/0中性，trace_id 关联） |
| `GET /api/v1/feedback/stats?days=30` | 近 N 天好评率/差评率 |
| `GET /api/v1/feedback/trace/{trace_id}` | 按 trace 查反馈 |

通过 `trace_id` 将用户反馈与评测记录关联，形成"自动 + 人工"双评估闭环。

---

## 8.9 评测仪表盘

**页面**：`http://localhost:8002/eval`

三个 Tab：

| Tab | 数据来源 | 展示内容 |
|-----|---------|---------|
| 在线评测 | `online_scores` 表 | NL2SQL三指标 + 知识检索RAG Triad统计卡片 + 记录表（含Token/费用） |
| 离线评测 | TruLens `records`/`feedbacks` 表 | Leaderboard + 评测记录（含各指标分数） |
| 用户反馈 | `feedback` API | 好评/差评比例、统计 |

**TruLens Dashboard**：`http://localhost:8501`（运行 `--dashboard` 启动）

---

## 8.10 配置总览

```python
# src/core/config.py
TRULENS_ENABLED: bool = True
EVAL_SAMPLE_RATE: float = 0.1           # 在线采样率

GUARDRAILS_ENABLED: bool = True         # 安全检测开关
GUARDRAILS_BLOCK_DDL: bool = True
GUARDRAILS_BLOCK_DML_WITHOUT_WHERE: bool = True

MODEL_PRICING_INPUT: float = 0.4        # $/1M tokens
MODEL_PRICING_OUTPUT: float = 1.2       # $/1M tokens
```

---

## 8.11 面试要点

### 在线 vs 离线评测

- **在线**：每次 API 请求后 fire-and-forget，采样率控制成本，评估实际生产质量
- **离线**：批量运行评测集，多通道/多配置对比，量化每个设计决策的影响

### LLM-as-Judge 的局限性

- **中庸倾向**：LLM 倾向打中间分（如 0.7），需要通过 prompt 工程（细化评分档次、强调使用全范围）来缓解
- **self-enhancement bias**：LLM 可能偏好自己生成的答案
- **一致性**：同一份答案两次打分可能不同
- **成本**：每次评估调一次 LLM

### 为什么"自动 + 人工"双闭环

- TruLens 评估"相对质量"（A 比 B 好）
- 用户反馈提供"绝对质量"（这个答案对不对）
- 两者互补，持续迭代

---

## 8.12 下一步

→ `docs/step-9-面试问答实战.md`
