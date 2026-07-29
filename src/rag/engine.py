# ============================================================
# RAGEngine — 可配置检索增强生成引擎（消融实验核心）
# ============================================================

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.language_models import BaseChatModel
from loguru import logger
from pymilvus import MilvusClient

from src.core.config import get_settings
from src.rag.config import RAGConfig
from src.rag.retrieval.vector_search import vector_search
from src.rag.retrieval.hybrid_search import hybrid_search
from src.rag.retrieval.reranker import rerank

settings = get_settings()


class RAGEngine:
    def __init__(self, config: RAGConfig, llm: BaseChatModel, milvus: MilvusClient):
        self.config = config
        self.llm = llm
        self.milvus = milvus
        self.embedding = DashScopeEmbeddings(
            model=config.embedding_model,
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
        )

    async def retrieve(self, question: str, filters: dict | None = None) -> list[str]:
        """检索阶段"""
        # HyDE
        if self.config.retrieval.use_hyde:
            from src.rag.retrieval.hyde import generate_hyde_query
            query_vec = await generate_hyde_query(question, self.llm, self.embedding)
        else:
            query_vec = await self.embedding.aembed_query(question)

        # 检索
        if self.config.retrieval.use_hybrid:
            hits = await hybrid_search(
                self.milvus, self.config.collection_name,
                query_vec, question, top_k=self.config.retrieval.top_k, filters=filters,
            )
        else:
            hits = await vector_search(
                self.milvus, self.config.collection_name,
                query_vec, top_k=self.config.retrieval.top_k, filters=filters,
            )

        # Rerank
        if self.config.retrieval.use_rerank and len(hits) > self.config.retrieval.rerank_top_k:
            hits = await rerank(question, hits, top_k=self.config.retrieval.rerank_top_k)

        return [h["text"] for h in hits]

    async def generate(self, question: str, contexts: list[str]) -> str:
        """生成阶段"""
        from src.rag.generation.generator import generate_answer
        return await generate_answer(question, contexts, self.llm)

    async def query(self, question: str, filters: dict | None = None) -> str:
        """完整 query: retrieve → generate"""
        contexts = await self.retrieve(question, filters)
        return await self.generate(question, contexts)
