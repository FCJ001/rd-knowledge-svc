# ============================================================
# Dense 向量嵌入生成器（DashScope text-embedding-v3）
#
# ★ Sparse 向量由 Milvus 2.6 内置 BM25 Function 自动生成
# ============================================================

from langchain_community.embeddings import DashScopeEmbeddings
from loguru import logger

from src.core.config import get_settings

settings = get_settings()


class DenseEmbedder:
    """DashScope dense embedding 生成器，超长文本自动截断"""

    def __init__(self):
        self.model = DashScopeEmbeddings(
            model=settings.EMBEDDING_MODEL,
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """生成 dense 向量。超长文本截断到 6000 chars 避免 DashScope 8192 token 限制"""
        if not texts:
            return []
        truncated = [t[:6000] if len(t) > 6000 else t for t in texts]
        try:
            return await self.model.aembed_documents(truncated)
        except Exception as e:
            logger.error(f"dense embedding 失败: {e}")
            return []
