# ============================================================
# Query 改写：口语 → 专业术语
# ============================================================

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from loguru import logger

REWRITE_PROMPT = """将用户的自然语言问题改写为更精准的技术查询用语：

用户问题：{question}

改写后的查询："""


async def transform_query(question: str, llm: BaseChatModel) -> str:
    """Query 改写"""
    try:
        prompt = REWRITE_PROMPT.format(question=question)
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        logger.warning(f"Query 改写失败: {e}")
        return question
