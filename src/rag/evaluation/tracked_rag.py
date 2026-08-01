# ============================================================
# TrackedRAG — TruLens 追踪包装
# 将检索通路包装为 retrieve → generate → query 三段式
# 通过 @instrument 注入追踪，不修改任何现有检索函数
#
# ★ 从 tiangong-agent 医疗版适配为汽车 ALM 领域
# ============================================================

from __future__ import annotations

import json

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from neo4j import AsyncDriver
from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import AsyncSession
from trulens.core.otel.instrument import SpanAttributes, instrument

from src.rag.evaluation.token_tracker import TokenTracker

SpanType = SpanAttributes.SpanType


class TrackedRAG:
    """RAG 检索通路的 TruLens 追踪包装。每个实例绑定一条检索通路。"""

    def __init__(
        self,
        channel: str,
        llm: BaseChatModel,
        embedding_model: Embeddings,
        milvus_client: MilvusClient,
        neo4j_driver: AsyncDriver,
        db_session: AsyncSession | None = None,
        role: str = "engineer",
    ):
        self.channel = channel
        self.embedding_model = embedding_model
        self.milvus_client = milvus_client
        self.neo4j_driver = neo4j_driver
        self.db_session = db_session
        self.role = role
        self._token_tracker = TokenTracker()
        self.llm = llm.with_config({"callbacks": [self._token_tracker]})

    @property
    def token_usage(self) -> dict:
        return self._token_tracker.usage

    @instrument(
        span_type=SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    async def retrieve(self, query: str) -> list[str]:
        """检索步骤 — 根据 channel 分发到对应的检索函数"""
        if self.channel == "doc_rag":
            from src.knowledge.doc_rag import format_doc_context, search_docs_raw

            hits = await search_docs_raw(
                question=query,
                embedding_model=self.embedding_model,
                milvus_client=self.milvus_client,
                llm=self.llm,
                use_hyde=True,
            )
            return [h["text"] for h in hits] if hits else []

        elif self.channel == "graph_rag":
            from src.knowledge.graph_rag import search_graph_raw

            records = await search_graph_raw(query, self.neo4j_driver, self.llm)
            return [json.dumps(r, ensure_ascii=False) for r in records] if records else []

        elif self.channel == "nl2sql":
            from src.nl2sql.engine import search_sql_raw

            if not self.db_session:
                return ["数据库连接不可用"]
            data, sql = await search_sql_raw(
                query, self.llm, self.db_session, role=self.role,
            )
            contexts = [f"[SQL] {sql}"]
            contexts.extend([json.dumps(d, ensure_ascii=False) for d in data[:10]])
            return contexts

        elif self.channel == "fusion":
            import asyncio

            from src.knowledge.doc_rag import search_docs_raw
            from src.knowledge.graph_rag import search_graph_raw

            doc_task = search_docs_raw(
                question=query,
                embedding_model=self.embedding_model,
                milvus_client=self.milvus_client,
                llm=self.llm,
                use_hyde=True,
            )
            graph_task = search_graph_raw(query, self.neo4j_driver, self.llm)
            doc_hits, graph_records = await asyncio.gather(
                doc_task, graph_task, return_exceptions=True,
            )

            contexts = []
            if isinstance(doc_hits, list):
                contexts.extend([h["text"] for h in doc_hits])
            if isinstance(graph_records, list):
                contexts.extend([json.dumps(r, ensure_ascii=False) for r in graph_records])
            return contexts

        elif self.channel == "change_review":
            import asyncio

            from src.knowledge.doc_rag import search_docs_raw
            from src.knowledge.graph_rag import search_graph_raw

            doc_task = search_docs_raw(
                question=query,
                embedding_model=self.embedding_model,
                milvus_client=self.milvus_client,
                doc_type="spec_doc",
            )
            graph_task = search_graph_raw(query, self.neo4j_driver, self.llm)
            doc_hits, graph_records = await asyncio.gather(
                doc_task, graph_task, return_exceptions=True,
            )

            contexts = []
            if isinstance(doc_hits, list):
                contexts.extend([h["text"] for h in doc_hits])
            if isinstance(graph_records, list):
                contexts.extend([json.dumps(r, ensure_ascii=False) for r in graph_records])
            return contexts

        return []

    @instrument(span_type=SpanType.GENERATION)
    async def generate(self, query: str, contexts: list[str]) -> str:
        """生成步骤 — 基于检索结果调用 LLM 生成回答"""
        if not contexts:
            return "未找到相关信息。"

        context_str = "\n---\n".join(contexts[:10])
        prompt = (
            f"你是 ALM 研发数据平台的知识问答助手。根据以下检索结果回答用户问题。\n"
            f"如果检索结果中没有答案，请明确告知。\n\n"
            f"用户角色：{self.role}\n"
            f"检索结果：\n{context_str}\n\n"
            f"用户问题：{query}"
        )
        response = await self.llm.ainvoke([SystemMessage(content=prompt)])
        return response.content

    @instrument()
    async def query(self, query: str) -> str:
        """完整 RAG 流程入口"""
        contexts = await self.retrieve(query)
        answer = await self.generate(query, contexts)
        return answer
