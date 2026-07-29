# ============================================================
# RAG 评估指标定义
# RAG Triad 三指标 + 汽车安全领域自定义指标
# ★ TruLens 2.x OTEL 模式：用 on_input/on_output
# ============================================================

import numpy as np
from trulens_eval import Feedback, LiteLLM


def build_rag_triad_metrics(provider: LiteLLM) -> list[Feedback]:
    """RAG Triad 三大核心指标"""

    m_answer_relevance = Feedback(
        provider.relevance_with_cot_reasons,
        name="答案相关性",
    ).on_input().on_output()

    m_context_relevance = Feedback(
        provider.context_relevance_with_cot_reasons,
        name="上下文相关性",
    ).on_input().on_output()

    m_groundedness = Feedback(
        provider.groundedness_measure_with_cot_reasons,
        name="有据性",
    ).on_input().on_output()

    return [m_answer_relevance, m_context_relevance, m_groundedness]


def build_safety_metrics(provider: LiteLLM) -> list[Feedback]:
    """汽车领域安全合规指标"""

    m_safety = Feedback(
        provider.harmfulness_with_cot_reasons,
        name="安全性",
    ).on_output()

    return [m_safety]


def build_all_metrics(provider: LiteLLM) -> list[Feedback]:
    """全部指标：RAG Triad + 安全合规"""
    return build_rag_triad_metrics(provider) + build_safety_metrics(provider)
