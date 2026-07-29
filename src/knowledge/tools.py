# ============================================================
# 知识 Agent 工具集（5 个 @tool）
# ★ 每个 tool 外裹 Timer() + QueryAuditLog.log()
# ============================================================

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from neo4j import AsyncDriver
from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.knowledge.audit import QueryAuditLog, Timer


class KnowledgeDeps:
    """知识检索依赖注入容器"""

    def __init__(
        self,
        llm: BaseChatModel,
        embedding_model: Embeddings,
        milvus_client: MilvusClient,
        neo4j_driver: AsyncDriver,
        db_session: AsyncSession | None = None,
        user_id: str = "anonymous",
        role: str = "engineer",
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.milvus_client = milvus_client
        self.neo4j_driver = neo4j_driver
        self.db_session = db_session
        self.user_id = user_id
        self.role = role


async def _rewrite(question: str, deps: KnowledgeDeps) -> str:
    """Query 改写"""
    from src.knowledge.query_rewriter import rewrite_query
    result = await rewrite_query(question, deps.llm, deps.role)
    return result["queries"][0] if result.get("queries") else question


def build_knowledge_tools(deps: KnowledgeDeps) -> list:
    """构建 5 个知识检索工具"""

    @tool
    async def search_docs(question: str, doc_type: str = "", model_code: str = "") -> str:
        """从知识文档库中检索信息。
        适用：查维修手册、技术规范、TSB、历史问题案例。
        question: 用户问题
        doc_type: 文档类型过滤 repair_manual/spec_doc/tsb/issue_case
        model_code: 车型代码过滤（如 汉EV_2024）"""
        from src.knowledge.doc_rag import search_docs as _search_docs
        rewritten = await _rewrite(question, deps)
        with Timer() as t:
            result = await _search_docs(
                question=rewritten, embedding_model=deps.embedding_model,
                milvus_client=deps.milvus_client, llm=deps.llm,
                doc_type=doc_type or None, model_code=model_code or None,
                role=deps.role,
            )
        QueryAuditLog.log(
            deps.user_id, deps.role, question, "doc_rag",
            ["doc_rag"], result[:80], t.elapsed_ms,
        )
        return result

    @tool
    async def search_graph(question: str) -> str:
        """从追溯图谱中检索实体关系。
        适用：查询现象→根因→配置项→基线→需求的多跳关系链。
        question: 用户问题"""
        from src.knowledge.graph_rag import search_graph as _search_graph
        rewritten = await _rewrite(question, deps)
        with Timer() as t:
            result = await _search_graph(
                question=rewritten, neo4j_driver=deps.neo4j_driver,
                llm=deps.llm, role=deps.role,
            )
        QueryAuditLog.log(
            deps.user_id, deps.role, question, "graph_rag",
            ["graph_rag"], result[:80], t.elapsed_ms,
        )
        return result

    @tool
    async def search_metrics(question: str) -> str:
        """查询研发过程指标和运营数据（NL2SQL）。
        适用：问题闭环率、变更率、索赔成本、OTA成功率等统计数据。
        question: 自然语言数据查询"""
        if deps.db_session is None:
            return "数据库连接不可用，无法执行查询。"
        from src.nl2sql.engine import search_sql
        with Timer() as t:
            result = await search_sql(question=question, llm=deps.llm, db=deps.db_session)
        QueryAuditLog.log(
            deps.user_id, deps.role, question, "nl2sql",
            ["nl2sql"], result[:80], t.elapsed_ms,
        )
        return result

    @tool
    async def search_multi(question: str) -> str:
        """多通道融合检索：同时查文档库和追溯图谱，综合回答。
        适用：复杂问题，需要同时参考手册和实体关系链。
        question: 用户问题"""
        from src.knowledge.fusion import multi_channel_search
        rewritten = await _rewrite(question, deps)
        with Timer() as t:
            result = await multi_channel_search(
                question=rewritten, llm=deps.llm,
                embedding_model=deps.embedding_model,
                milvus_client=deps.milvus_client,
                neo4j_driver=deps.neo4j_driver,
                db_session=deps.db_session,
                role=deps.role,
            )
        QueryAuditLog.log(
            deps.user_id, deps.role, question, "multi",
            ["doc_rag", "graph_rag"], result[:80], t.elapsed_ms,
        )
        return result

    @tool
    async def review_change(question: str) -> str:
        """变更影响预检：分析变更可能的影响范围、风险和验证步骤。
        适用：业务/售后自查变更影响，不需要工程深度分析。
        question: 变更描述"""
        from src.knowledge.review_change import review_change as _review_change
        with Timer() as t:
            result = await _review_change(
                change_info=question, llm=deps.llm,
                embedding_model=deps.embedding_model,
                milvus_client=deps.milvus_client,
                neo4j_driver=deps.neo4j_driver,
            )
        QueryAuditLog.log(
            deps.user_id, deps.role, question, "change_review",
            ["doc_rag", "graph_rag"], result[:80], t.elapsed_ms,
        )
        return result

    return [search_docs, search_graph, search_metrics, search_multi, review_change]
