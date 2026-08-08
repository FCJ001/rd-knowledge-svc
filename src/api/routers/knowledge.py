# ============================================================
# 知识检索 API
#
# POST /api/v1/knowledge/search    多通道检索
# POST /api/v1/knowledge/feedback  用户反馈
# GET  /api/v1/knowledge/docs      文档列表
# ============================================================

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_llm, get_embedding_model
from src.core.base_schema import PageResult, ResponseSchema
from src.core.deps import PageParams, UserContext, get_current_user
from src.core.exceptions import ERR_BAD_REQUEST, ERR_NOT_FOUND, BizException
from src.core.logger import logger
from src.infra.db import get_db
from src.infra.milvus_client import get_milvus_client
from src.infra.neo4j_client import get_neo4j_driver
from src.knowledge.model import KnowledgeDoc, KnowledgeFeedback
from src.rag.evaluation.async_tracker import get_async_evaluator
from src.rag.evaluation.guardrails import check_output
from src.rag.evaluation.token_tracker import TokenTracker

router = APIRouter(prefix="/api/v1/knowledge", tags=["知识检索"])


# ── Request / Response models ────────────────────────────────────────────

class SearchRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    channels: list[str] = Field(
        default=["doc_rag", "graph_rag"],
        description="检索通道: doc_rag / graph_rag / nl2sql"
    )
    doc_type: str = Field(default="", description="文档类型过滤: repair_manual/spec_doc/tsb/issue_case")
    model_code: str = Field(default="", description="车型代码过滤")
    use_hyde: bool = Field(default=False, description="是否启用 HyDE（假设性文档嵌入，默认关）")


class SearchResponse(BaseModel):
    question: str
    answer: str
    channels: list[str]


class FeedbackRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    answer_preview: str = Field(default="", description="答案摘要")
    rating: int = Field(default=0, description="评分: 1 赞 / -1 踩 / 0 中性")
    comment: str = Field(default="", description="评语")
    intent: str = Field(default="", description="意图分类")
    channels: str = Field(default="", description="使用的检索通道")
    trace_id: str = Field(default="", description="关联请求")


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/search", response_model=ResponseSchema[SearchResponse])
async def search_knowledge(
    req: SearchRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """多通道知识检索：文档 + 图谱 + NL2SQL 并行查询"""
    llm = get_llm()
    embedding_model = get_embedding_model()
    milvus_client = get_milvus_client()
    neo4j_driver = get_neo4j_driver()

    # ── Token 追踪 ──
    token_tracker = TokenTracker()
    llm = llm.with_config({"callbacks": [token_tracker]})

    from src.knowledge.audit import QueryAuditLog, Timer
    from src.knowledge.fusion import multi_channel_search

    with Timer() as timer:
        result = await multi_channel_search(
            question=req.question,
            llm=llm,
            embedding_model=embedding_model,
            milvus_client=milvus_client,
            neo4j_driver=neo4j_driver,
            db_session=db if "nl2sql" in req.channels else None,
            channels=req.channels,
            role=user.role,
            use_hyde=req.use_hyde,
        )
    answer = result["answer"] if isinstance(result, dict) else result
    contexts = result.get("contexts", []) if isinstance(result, dict) else []

    # ── 查询审计日志（检索链路在 API 端点落审计）──
    QueryAuditLog.log(
        user_id=user.user_id,
        role=user.role,
        question=req.question,
        intent="multi_channel",
        channels=req.channels,
        answer_preview=answer,
        duration_ms=timer.elapsed_ms,
    )

    # ── Guardrails 输出检查 ──
    safe, reason = check_output(answer)
    if not safe:
        logger.warning(f"[Guardrails] 知识检索输出检查: {reason}")

    # ── 异步评估（fire-and-forget，按采样率触发）──
    evaluator = get_async_evaluator()
    evaluator.evaluate_knowledge(question=req.question, answer=answer, contexts=contexts, token_usage=token_tracker.usage)

    # ── Token 用量 ──
    usage = token_tracker.usage
    logger.info(
        f"[Token] 知识检索 input={usage['input_tokens']} output={usage['output_tokens']} "
        f"total={usage['total_tokens']} calls={usage['calls']} "
        f"cost=${usage['cost_usd']:.6f}"
    )

    return ResponseSchema(data=SearchResponse(
        question=req.question,
        answer=answer,
        channels=req.channels,
    ))


@router.post("/feedback", response_model=ResponseSchema[dict])
async def submit_feedback(
    req: FeedbackRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """提交知识检索反馈"""
    feedback = KnowledgeFeedback(
        user_id=user.user_id,
        question=req.question,
        answer_preview=req.answer_preview or None,
        rating=req.rating,
        comment=req.comment or None,
        intent=req.intent or None,
        channels=req.channels or None,
        trace_id=req.trace_id or None,
    )
    db.add(feedback)
    await db.flush()
    return ResponseSchema(data={"id": str(feedback.id), "status": "ok"})


@router.get("/docs", response_model=ResponseSchema[PageResult[dict]])
async def list_docs(
    page: PageParams = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """文档元数据列表"""
    from src.core.base_repository import BaseRepository
    repo = BaseRepository(KnowledgeDoc, db)
    items, total = await repo.get_page(
        offset=page.offset,
        limit=page.page_size,
        keyword=page.keyword,
        search_fields=["doc_name", "doc_type", "category"],
    )
    docs = [
        {
            "id": str(d.id),
            "doc_id": d.doc_id,
            "doc_name": d.doc_name,
            "doc_type": d.doc_type,
            "category": d.category,
            "business_line": d.business_line,
            "model_code": d.model_code,
            "chunk_count": d.chunk_count,
            "chunk_strategy": d.chunk_strategy,
            "version": d.version,
            "status": d.status,
        }
        for d in items
    ]
    return ResponseSchema(data=PageResult(
        items=docs,
        total=total,
        page=page.page,
        page_size=page.page_size,
    ))
