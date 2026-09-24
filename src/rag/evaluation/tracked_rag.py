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

    # 喂给 LLM 的上下文条数上限。评测侧裁定「上下文相关性 / 有据性」时必须
    # 用同一个窗口：run_triad_direct 原来写死 8 条，比生成侧少看两条 ——
    # 排在列表后面的图谱证据被截掉，两项指标系统性偏低（答案满分却「无据」）。
    CONTEXT_WINDOW = 10

    def __init__(
        self,
        channel: str,
        llm: BaseChatModel,
        embedding_model: Embeddings,
        milvus_client: MilvusClient,
        neo4j_driver: AsyncDriver,
        db_session: AsyncSession | None = None,
        role: str = "engineer",
        use_hyde: bool | None = None,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
    ):
        """消融参数（use_hyde / top_k / rerank_top_k）为 None 时回落到
        全局配置（Settings.RAG_*），使 .env 成为离线调参的单一事实来源。
        """
        from src.core.config import get_settings as _get_settings

        _s = _get_settings()
        self.use_hyde = _s.RAG_HYDE_ENABLED if use_hyde is None else use_hyde
        self.top_k = _s.RAG_TOP_K if top_k is None else top_k
        self.rerank_top_k = _s.RAG_RERANK_TOP_K if rerank_top_k is None else rerank_top_k

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

    async def _retrieve_bi(self, query: str) -> list[str]:
        """问数通道：HTTP 调 rd-chatBI（NL2SQL 已迁出为独立服务）。

        nl2sql 单通道与 fusion 三路共用同一份实现，避免两处各写一份 HTTP 调用。
        """
        import httpx

        from src.core.config import get_settings

        settings = get_settings()
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{settings.CHATBI_URL}/api/v1/bi/query",
                json={"question": query, "session_id": "eval", "with_chart": False},
                headers={
                    "X-User-Id": "knowledge-svc",
                    # 评测口径：用 admin（role_rules 里等于 all，不做行级过滤），
                    # 与 NL2SQL 离线评测（run_nl2sql_eval 用 role="admin"）一致。
                    # 用 engineer 会因缺 owner_domain_id 参数被拒 —— 问数通道恒为空。
                    "X-User-Role": "admin",
                    "X-Project-Id": settings.BI_PROJECT_ID if hasattr(settings, "BI_PROJECT_ID") else settings.CHATBI_PROJECT_ID,
                },
            )
            data = (resp.json().get("data") or {})
            if not data.get("success"):
                return [f"查询失败: {data.get('error', 'unknown')}"]
            contexts = [f"[SQL] {data.get('sql', '')}"]
            contexts.extend(
                json.dumps(d, ensure_ascii=False, default=str)
                for d in (data.get("data") or [])[:10]
            )
            return contexts

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
            from src.knowledge.doc_rag import search_docs_raw

            hits = await search_docs_raw(
                question=query,
                embedding_model=self.embedding_model,
                milvus_client=self.milvus_client,
                role=self.role,
                llm=self.llm,
                use_hyde=self.use_hyde,
                top_k=self.top_k,
                rerank_top_k=self.rerank_top_k,
            )
            return [h["text"] for h in hits] if hits else []

        elif self.channel == "graph_rag":
            from src.knowledge.graph_rag import search_graph_raw

            records = await search_graph_raw(query, self.neo4j_driver, self.llm)
            return [json.dumps(r, ensure_ascii=False) for r in records] if records else []

        elif self.channel == "nl2sql":
            return await self._retrieve_bi(query)

        elif self.channel == "fusion":
            import asyncio

            from src.knowledge.doc_rag import search_docs_raw
            from src.knowledge.graph_rag import search_graph_raw

            doc_task = search_docs_raw(
                question=query,
                embedding_model=self.embedding_model,
                milvus_client=self.milvus_client,
                role=self.role,
                llm=self.llm,
                use_hyde=True,
            )
            graph_task = search_graph_raw(query, self.neo4j_driver, self.llm)
            # ★ 生产融合是「文档 + 图谱 + 问数」三路。评测原来只跑了两路，
            #   业务数据类问题（如「近 3 个月 S1 数量 / 闭环率」）必然判 0 —— 补上第三路。
            bi_task = self._retrieve_bi(query)
            doc_hits, graph_records, bi_contexts = await asyncio.gather(
                doc_task, graph_task, bi_task, return_exceptions=True,
            )

            contexts = []
            # ★ 与线上融合层（src/knowledge/fusion.py）一致的证据组装方式：
            #   三路各取一段证据进入上下文。原来把 N 条文档片段平铺在最前面，
            #   [:CONTEXT_WINDOW] 一截就把图谱/问数证据整段切掉 —— 融合通道
            #   实际退化成纯文档通道，这是评测与线上最大的偏差。
            if isinstance(doc_hits, list) and doc_hits:
                contexts.append("\n\n".join(h["text"] for h in doc_hits)[:1000])
            if isinstance(graph_records, list) and graph_records:
                contexts.append(json.dumps(graph_records, ensure_ascii=False)[:1000])
            if isinstance(bi_contexts, list) and bi_contexts:
                contexts.append("\n".join(bi_contexts)[:1000])
            return contexts

        elif self.channel == "change_review":
            import asyncio

            from src.knowledge.doc_rag import search_docs_raw
            from src.knowledge.graph_rag import search_graph_raw

            doc_task = search_docs_raw(
                question=query,
                embedding_model=self.embedding_model,
                milvus_client=self.milvus_client,
                role=self.role,
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

    async def retrieve_docs(self, query: str, top_k: int | None = None,
                            rerank_top_k: int | None = None) -> list[dict]:
        """检索原始命中（含 doc_name 等元数据），供 HitRate@K / Recall@K / MRR 评测。

        不传参数时用实例配置（默认来自 Settings.RAG_*，见 __init__）。
        调参场景可传 top_k=50 把召回放宽，一次检索即可切 K=5/10/20 三档指标。
        指标只算 doc 路：fusion 不再并发 graph 查询（其结果不参与 doc 级
        ground truth，省掉每题两次 LLM 调用）；change_review 与生产 retrieve
        一致不走 HyDE。graph_rag 返回图谱记录而非文档，无 doc 级
        ground truth，返回 []。
        """
        if self.channel == "graph_rag":
            return []

        from src.knowledge.doc_rag import search_docs_raw

        doc_kwargs: dict = {}
        if self.channel == "change_review":
            doc_kwargs["doc_type"] = "spec_doc"
        else:
            doc_kwargs["llm"] = self.llm
            doc_kwargs["use_hyde"] = self.use_hyde

        return await search_docs_raw(
            question=query,
            embedding_model=self.embedding_model,
            milvus_client=self.milvus_client,
            role=self.role,
            top_k=self.top_k if top_k is None else top_k,
            rerank_top_k=self.rerank_top_k if rerank_top_k is None else rerank_top_k,
            **doc_kwargs,
        )

    @instrument(span_type=SpanType.GENERATION)
    async def generate(self, query: str, contexts: list[str]) -> str:
        """生成步骤 — 基于检索结果调用 LLM 生成回答"""
        if not contexts:
            return "未找到相关信息。"

        context_str = "\n---\n".join(contexts[: self.CONTEXT_WINDOW])
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
