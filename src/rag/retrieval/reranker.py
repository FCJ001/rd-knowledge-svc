# ============================================================
# DashScope Reranker
# ============================================================

from loguru import logger

from src.core.config import get_settings

settings = get_settings()


async def rerank(query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
    """DashScope qwen3-rerank 精排"""
    if not documents or len(documents) <= top_k:
        return documents

    try:
        import dashscope
        from dashscope import TextReRank

        dashscope.api_key = settings.DASHSCOPE_API_KEY
        texts = [d.get("text", "") for d in documents]

        response = TextReRank.call(
            model="qwen3-rerank",
            query=query,
            documents=texts,
            top_n=top_k,
            return_documents=False,
        )

        if response.status_code != 200:
            logger.warning(f"Reranker 失败: {response.message}")
            return documents[:top_k]

        return [
            {**documents[item.index], "rerank_score": item.relevance_score}
            for item in response.output.results
        ]
    except ImportError:
        return documents[:top_k]
    except Exception as e:
        logger.warning(f"Reranker 异常: {e}")
        return documents[:top_k]
