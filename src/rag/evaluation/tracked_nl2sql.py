# ============================================================
# TrackedNL2SQL — TruLens 追踪包装（新 NL2SQL 12 节点流水线）
# 将流水线拆为 retrieve → generate → query 三段式
# 通过 @instrument 注入追踪，用于 TruLens 离线评测
# ============================================================

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from elasticsearch import AsyncElasticsearch
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import AsyncSession
from trulens.core.otel.instrument import SpanAttributes, instrument

from src.nl2sql.repositories import (
    ESValueRepository,
    MilvusColumnRepository,
    MilvusMetricRepository,
    PgMetaRepository,
)
from src.rag.evaluation.token_tracker import TokenTracker

SpanType = SpanAttributes.SpanType


class TrackedNL2SQL:
    """NL2SQL 12 节点流水线的 TruLens 追踪包装。

    三段式拆分：
      retrieve — 关键词提取 → 3路召回 → 合并 → 2路过滤（节点 ①②③④）
      generate  — 注入上下文 → 生成SQL → 校验 → 纠错（节点 ⑤⑥⑦⑧）
      query     — 完整流水线 + 执行（全部节点 ①-⑨）
    """

    def __init__(
        self,
        llm: BaseChatModel,
        embedding_model: Embeddings,
        milvus_client: MilvusClient,
        es_client: AsyncElasticsearch,
        pg_meta_repo: PgMetaRepository,
        dw_db_session: AsyncSession,
        role: str = "engineer",
        owner_domain_id: int | None = None,
        business_line: str | None = None,
    ):
        self._token_tracker = TokenTracker()
        self.llm = llm.with_config({"callbacks": [self._token_tracker]})
        self.embedding_model = embedding_model
        self.milvus_client = milvus_client
        self.es_client = es_client
        self.pg_meta_repo = pg_meta_repo
        self.dw_db_session = dw_db_session
        self.role = role
        self.owner_domain_id = owner_domain_id
        self.business_line = business_line

        # 构建节点函数所需的 DataAgentContext
        self.ctx = {
            "llm": self.llm,
            "embedding_model": embedding_model,
            "milvus_client": milvus_client,
            "milvus_column_repo": MilvusColumnRepository(milvus_client),
            "milvus_metric_repo": MilvusMetricRepository(milvus_client),
            "es_client": es_client,
            "es_value_repo": ESValueRepository(es_client),
            "pg_meta_repo": pg_meta_repo,
            "dw_db_session": dw_db_session,
            # 与流水线一致：execute_sql 节点做角色行级过滤
            "role": role,
            "owner_domain_id": owner_domain_id,
            "business_line": business_line,
        }

        # 流水线中间状态（retrieve → generate 之间传递）
        self._state: dict = {}

    @property
    def token_usage(self) -> dict:
        return self._token_tracker.usage

    # ── 节点函数引用 ────────────────────────────────────────────────

    @staticmethod
    async def _extract_keywords(state: dict, ctx: dict) -> dict:
        from src.nl2sql.nodes.extract_keywords import extract_keywords
        return await extract_keywords(state, ctx)

    @staticmethod
    async def _recall_columns(state: dict, ctx: dict) -> dict:
        from src.nl2sql.nodes.recall_columns import recall_columns
        return await recall_columns(state, ctx)

    @staticmethod
    async def _recall_values(state: dict, ctx: dict) -> dict:
        from src.nl2sql.nodes.recall_values import recall_values
        return await recall_values(state, ctx)

    @staticmethod
    async def _recall_metrics(state: dict, ctx: dict) -> dict:
        from src.nl2sql.nodes.recall_metrics import recall_metrics
        return await recall_metrics(state, ctx)

    @staticmethod
    async def _merge_info(state: dict, ctx: dict) -> dict:
        from src.nl2sql.nodes.merge_info import merge_info
        return await merge_info(state, ctx)

    @staticmethod
    async def _filter_tables(state: dict, ctx: dict) -> dict:
        from src.nl2sql.nodes.filter_tables import filter_tables
        return await filter_tables(state, ctx)

    @staticmethod
    async def _filter_metrics(state: dict, ctx: dict) -> dict:
        from src.nl2sql.nodes.filter_metrics import filter_metrics
        return await filter_metrics(state, ctx)

    @staticmethod
    async def _add_context(state: dict, ctx: dict) -> dict:
        from src.nl2sql.nodes.add_context import add_context
        return await add_context(state, ctx)

    @staticmethod
    async def _generate_sql(state: dict, ctx: dict) -> dict:
        from src.nl2sql.nodes.generate_sql import generate_sql
        return await generate_sql(state, ctx)

    @staticmethod
    async def _validate_sql(state: dict, ctx: dict) -> dict:
        from src.nl2sql.nodes.validate_sql import validate_sql
        return await validate_sql(state, ctx)

    @staticmethod
    async def _correct_sql(state: dict, ctx: dict) -> dict:
        from src.nl2sql.nodes.correct_sql import correct_sql
        return await correct_sql(state, ctx)

    @staticmethod
    async def _execute_sql(state: dict, ctx: dict) -> dict:
        from src.nl2sql.nodes.execute_sql import execute_sql
        return await execute_sql(state, ctx)

    # ── 三段式追踪方法 ──────────────────────────────────────────────

    @instrument(
        span_type=SpanType.RETRIEVAL,
        attributes={
            "query": "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    async def retrieve(self, query: str) -> list[str]:
        """检索阶段：①②③④ — 关键词提取 → 3路召回 → 合并 → 2路过滤"""
        state: dict = {"query": query}

        # ① 关键词提取
        result = await self._extract_keywords(state, self.ctx)
        state.update(result)

        # ② 三路并行召回
        recall_results = await asyncio.gather(
            self._recall_columns(state, self.ctx),
            self._recall_values(state, self.ctx),
            self._recall_metrics(state, self.ctx),
        )
        for r in recall_results:
            state.update(r)

        # ③ 合并检索结果
        result = await self._merge_info(state, self.ctx)
        state.update(result)

        # ④ 两路并行过滤
        filter_results = await asyncio.gather(
            self._filter_tables(state, self.ctx),
            self._filter_metrics(state, self.ctx),
        )
        for r in filter_results:
            state.update(r)

        # 保存状态供 generate 使用
        self._state = state

        # 序列化检索结果为 context 字符串列表
        contexts: list[str] = []
        for t in state.get("table_infos", []):
            col_names = [c.name for c in t.columns] if hasattr(t, "columns") else []
            contexts.append(
                f"[TABLE] {t.name} ({t.role}): {t.description} "
                f"columns=[{', '.join(col_names)}]"
            )
        for m in state.get("metric_infos", []):
            contexts.append(
                f"[METRIC] {m.name}: {m.description} "
                f"relevant_columns={m.relevant_columns}"
            )
        return contexts

    @instrument(
        span_type=SpanType.GENERATION,
        attributes={
            "query": "query",
        },
    )
    async def generate(self, query: str, contexts: list[str]) -> str:
        """生成阶段：⑤⑥⑦⑧ — 注入上下文 → 生成SQL → 校验 → 纠错"""
        state = self._state
        if not state:
            # fallback：如果 retrieve 没调用过，从空状态开始
            state = {"query": query}
            self._state = state

        # ⑤ 注入日期/DB 上下文
        result = await self._add_context(state, self.ctx)
        state.update(result)

        # ⑥ LLM 生成 SQL
        result = await self._generate_sql(state, self.ctx)
        state.update(result)

        # ⑦ EXPLAIN 校验
        result = await self._validate_sql(state, self.ctx)
        state.update(result)

        # ⑧ 条件纠错
        if state.get("error"):
            result = await self._correct_sql(state, self.ctx)
            state.update(result)

        return json.dumps(
            {
                "sql": state.get("sql", ""),
                "error": state.get("error", ""),
            },
            ensure_ascii=False,
        )

    @instrument()
    async def query(self, query: str) -> str:
        """完整流水线：①-⑨ — retrieve → generate → execute"""
        await self.retrieve(query)
        gen_output = await self.generate(query, [])
        gen_data = json.loads(gen_output)

        # ⑨ 执行 SQL
        result = await self._execute_sql(self._state, self.ctx)
        self._state.update(result)

        return json.dumps(
            {
                "sql": gen_data.get("sql", ""),
                "error": gen_data.get("error", ""),
                "columns": self._state.get("result_columns", []),
                "data": self._state.get("result_data", [])[:20],
                "row_count": self._state.get("result_row_count", 0),
                "summary": self._state.get("result_summary", ""),
            },
            ensure_ascii=False,
            default=str,
        )
