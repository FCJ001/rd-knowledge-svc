# ============================================================
# NL2SQL 在线评估指标 — LLM-as-Judge 打分
# ============================================================

from __future__ import annotations

import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage


RESULT_RELEVANCE_PROMPT = """你是 NL2SQL 查询质量评审专家。根据用户问题和 SQL 返回的数据摘要，判断查询结果是否与问题相关。
请使用 0.0-1.0 的连续评分，**不要都打同一分数**。

## 用户问题
{question}

## SQL 摘要
{summary}

## 返回数据（前 5 行）
{data_preview}

## 详细评判标准
- 0.9-1.0：结果完全回答了问题，数据维度正确，数量准确，信息充分
- 0.7-0.8：结果基本回答了问题，但缺少一些维度或数据不够全面
- 0.5-0.6：结果部分相关，返回了部分有用数据但有明显遗漏
- 0.3-0.4：结果与问题勉强相关，但关键维度缺失
- 0.1-0.2：只有极少量数据与问题相关，大部分无关
- 0.0：结果完全不相关或 SQL 执行失败

返回纯 JSON（score 可以是任意小数，如 0.85）：
{{
    "score": <0.0-1.0>,
    "reason": "一行理由"
}}"""


async def judge_result_relevance(
    question: str,
    summary: str,
    data: list[dict],
    llm: BaseChatModel,
) -> tuple[float, str]:
    """LLM 评判查询结果与问题的相关性"""
    data_preview = json.dumps(data[:5], ensure_ascii=False, default=str) if data else "[]"
    prompt = RESULT_RELEVANCE_PROMPT.format(
        question=question,
        summary=summary[:500],
        data_preview=data_preview[:2000],
    )
    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        content = response.content.strip()
        match = re.search(r"\{[^{}]*\}", content)
        if match:
            result = json.loads(match.group())
            return float(result.get("score", 0)), result.get("reason", "")
    except Exception:
        pass
    return 0.0, ""


def score_sql_valid(error: str | None) -> float:
    """SQL 可执行率 — 客观指标，不需要 LLM"""
    return 0.0 if error else 1.0



