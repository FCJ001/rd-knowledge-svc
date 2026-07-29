# ============================================================
# HyDE（Hypothetical Document Embeddings）
# ============================================================

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from loguru import logger

from src.knowledge.prompts import HYDE_PROMPT


async def generate_hyde_embedding(
    question: str,
    llm: BaseChatModel,
    embedding_model: Embeddings,
) -> list[float]:
    """
    HyDE：LLM 生成假设性回答 → 嵌入 → 用该向量检索
    缩小 query-doc 语义鸿沟
    """
    prompt = HYDE_PROMPT.format(question=question)
    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        hypothetical_doc = response.content.strip()
        logger.debug(f"HyDE 假设文档: {hypothetical_doc[:100]}...")
        return await embedding_model.aembed_query(hypothetical_doc)
    except Exception as e:
        logger.warning(f"HyDE 生成失败，回退到原始查询向量: {e}")
        return await embedding_model.aembed_query(question)
