# ============================================================
# 嵌入向量生成器：dense + sparse
#
# ★ 补齐医疗版空文件，接通 Hybrid Search 的前提
# ============================================================

from langchain_community.embeddings import DashScopeEmbeddings
from loguru import logger

from src.core.config import get_settings

settings = get_settings()


class EmbeddingGenerator:
    """dense (DashScope text-embedding-v3) + sparse (BM25) 双向量生成"""

    def __init__(self):
        self.dense_model = DashScopeEmbeddings(
            model=settings.EMBEDDING_MODEL,
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
        )
        self._sparse_model = None

    @property
    def dim(self) -> int:
        return 1024

    async def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """DashScope text-embedding-v3 → 1024 dim"""
        if not texts:
            return []
        try:
            return await self.dense_model.aembed_documents(texts)
        except Exception as e:
            logger.error(f"dense embedding 失败: {e}")
            return []

    async def embed_sparse(self, texts: list[str]) -> list[dict]:
        """
        BM25 sparse 向量生成。
        用 pymilvus.model.sparse.BM25EmbeddingFunction 在应用层生成，
        不依赖 Milvus Function。
        """
        if not texts:
            return []
        try:
            from pymilvus.model.sparse import BM25EmbeddingFunction
            if self._sparse_model is None:
                self._sparse_model = BM25EmbeddingFunction()
                self._sparse_model.fit(texts)
            return self._sparse_model.encode_documents(texts)
        except ImportError:
            logger.warning("pymilvus.model.sparse 不可用，跳过 sparse embedding")
            return [{}] * len(texts)
        except Exception as e:
            logger.error(f"sparse embedding 失败: {e}")
            return [{}] * len(texts)

    async def generate(self, texts: list[str]) -> tuple[list[list[float]], list[dict]]:
        """同时生成 dense + sparse 向量"""
        dense = await self.embed_dense(texts)
        sparse = await self.embed_sparse(texts)
        return dense, sparse
