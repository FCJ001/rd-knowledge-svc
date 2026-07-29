# ============================================================
# 有据性校验（fail-open）：校对 LLM 回答是否基于证据
# ============================================================

from __future__ import annotations

import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from loguru import logger

from src.knowledge.prompts import HALLUCINATION_CHECK_PROMPT


async def check_hallucination(
    question: str,
    evidence: str,
    answer: str,
    llm: BaseChatModel,
    threshold: float = 0.7,
) -> dict:
    """
    幻觉检测。fail-open：异常时不拦截，返回 is_grounded=True。
    """
    prompt = HALLUCINATION_CHECK_PROMPT.format(
        question=question, evidence=evidence, answer=answer,
    )
    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        content = response.content.strip()
        # 容错提取 JSON：LLM 可能在 JSON 后继续输出额外内容
        match = re.search(r"\{[^{}]*\}", content)
        if match:
            content = match.group()
        result = json.loads(content)
        result["is_grounded"] = (
            result.get("is_grounded", False)
            and result.get("confidence", 0) >= threshold
        )
        return result
    except Exception as e:
        logger.warning(f"幻觉检测失败（fail-open，放行）: {e}")
        return {"is_grounded": True, "unsupported_claims": [], "confidence": 1.0}
