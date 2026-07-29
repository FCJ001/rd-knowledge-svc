# ============================================================
# LLM 答案生成
# ============================================================

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

PROMPT = """你是一个汽车研发知识助手。
请根据以下检索到的资料回答用户问题。

用户问题：{question}

参考资料：
{context}

请基于资料准确回答。如果资料不足以回答，请明确说明。"""


async def generate_answer(
    question: str,
    contexts: list[str],
    llm: BaseChatModel,
) -> str:
    """LLM 基于检索结果生成回答"""
    if not contexts:
        return "未找到相关信息。"
    context = "\n\n---\n\n".join(contexts)
    prompt = PROMPT.format(question=question, context=context)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.content
