# ============================================================
# DashScope Reranker（qwen3-rerank）— 后检索精排
# 4 层降级：空→[] / len<=k→透传 / ImportError→docs[:k] / 非200→docs[:k]
# ============================================================

from __future__ import annotations

from loguru import logger

from src.core.config import get_settings

settings = get_settings()


async def rerank_docs(
    query: str,
    documents: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """DashScope gte-rerank-v2 精排"""
    if not documents:
        return []

    if len(documents) <= top_k:
        return documents

    try:
        import dashscope
        from dashscope import TextReRank

        dashscope.api_key = settings.DASHSCOPE_API_KEY
        texts = [doc.get("text", "") for doc in documents]

        response = TextReRank.call(
            model="qwen3-rerank",
            query=query,
            documents=texts,
            top_n=top_k,
            return_documents=False,
        )

        if response.status_code != 200:
            logger.warning(f"Reranker 调用失败: {response.message}")
            return documents[:top_k]

        reranked = []
        for item in response.output.results:
            idx = item.index
            doc = documents[idx].copy()
            doc["rerank_score"] = item.relevance_score
            reranked.append(doc)

        return reranked

    except ImportError:
        logger.warning("dashscope 未安装，回退到向量距离排序")
        return documents[:top_k]
    except Exception as e:
        logger.warning(f"Reranker 异常，回退到向量距离排序: {e}")
        return documents[:top_k]
