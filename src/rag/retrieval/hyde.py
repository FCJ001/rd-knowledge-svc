# ============================================================
# HyDE 检索增强
# ============================================================

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from loguru import logger

HYDE_PROMPT = """请根据以下问题，生成一段假设性的技术回答作为检索参考：

问题：{question}

假设回答（200字以内）："""


async def generate_hyde_query(
    question: str,
    llm: BaseChatModel,
    embedding_model: Embeddings,
) -> list[float]:
    """LLM 生成假设文档 → embedding"""
    try:
        prompt = HYDE_PROMPT.format(question=question)
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        hypothetical = response.content.strip()
        logger.debug(f"HyDE: {hypothetical[:100]}...")
        return await embedding_model.aembed_query(hypothetical)
    except Exception as e:
        logger.warning(f"HyDE 失败，回退: {e}")
        return await embedding_model.aembed_query(question)
