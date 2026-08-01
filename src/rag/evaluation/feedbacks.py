# ============================================================
# TruLens 2.x 评估指标定义
#
# RAG Triad：答案相关性 / 上下文相关性 / 有据性
# NL2SQL：SQL 可执行率 / 结果相关性
# ============================================================

import json

import numpy as np
from trulens.core.metric import Metric
from trulens.core.metric.selector import Selector
from trulens.providers.litellm import LiteLLM


# ════════════════════════════════════════════════════════════════
# RAG Triad 三指标
# ════════════════════════════════════════════════════════════════

def build_rag_triad_metrics(provider: LiteLLM) -> list[Metric]:
    # 1. 答案相关性 (Question → Answer)
    m_answer_relevance = Metric(
        implementation=provider.relevance_with_cot_reasons,
        name="答案相关性",
        selectors={
            "prompt": Selector.select_record_input(),
            "response": Selector.select_record_output(),
        },
    )

    # 2. 上下文相关性 (Question → Context)
    m_context_relevance = Metric(
        implementation=provider.context_relevance_with_cot_reasons,
        name="上下文相关性",
        selectors={
            "question": Selector.select_record_input(),
            "context": Selector.select_context(collect_list=False),
        },
        agg=np.mean,
    )

    # 3. 有据性 (Context → Answer)
    m_groundedness = Metric(
        implementation=provider.groundedness_measure_with_cot_reasons,
        name="有据性",
        selectors={
            "source": Selector.select_context(collect_list=True),
            "statement": Selector.select_record_output(),
        },
    )

    return [m_answer_relevance, m_context_relevance, m_groundedness]


# ════════════════════════════════════════════════════════════════
# NL2SQL 自定义指标
# ════════════════════════════════════════════════════════════════

def _parse_nl2sql_output(response: str) -> dict:
    """从 TrackedNL2SQL.query() 的 JSON 输出提取字段"""
    try:
        return json.loads(response)
    except Exception:
        return {}


def _nl2sql_sql_valid(response: str) -> float:
    """SQL 可执行率：无 error = 1.0"""
    data = _parse_nl2sql_output(response)
    return 0.0 if data.get("error") else 1.0


def _nl2sql_has_data(response: str) -> float:
    """数据返回率：有数据=1.0，空结果=0.5，失败=0.0"""
    data = _parse_nl2sql_output(response)
    if data.get("error"):
        return 0.0
    return 1.0 if data.get("row_count", 0) > 0 else 0.5


def build_nl2sql_metrics(provider: LiteLLM) -> list[Metric]:
    # 1. SQL 可执行率（客观指标，不需要 LLM）
    m_sql_valid = Metric(
        implementation=_nl2sql_sql_valid,
        name="SQL可执行率",
        selectors={
            "response": Selector.select_record_output(),
        },
    )

    # 2. 结果相关性（LLM-as-Judge，比较问题与查询结果）
    m_relevance = Metric(
        implementation=provider.relevance_with_cot_reasons,
        name="结果相关性",
        selectors={
            "prompt": Selector.select_record_input(),
            "response": Selector.select_record_output(),
        },
    )

    # 3. 数据返回率（客观指标）
    m_has_data = Metric(
        implementation=_nl2sql_has_data,
        name="数据返回率",
        selectors={
            "response": Selector.select_record_output(),
        },
    )

    return [m_sql_valid, m_relevance, m_has_data]
