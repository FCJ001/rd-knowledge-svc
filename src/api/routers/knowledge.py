# ============================================================
# 知识检索 API
#
# POST /api/v1/knowledge/search    多通道检索
# POST /api/v1/knowledge/feedback  用户反馈
# GET  /api/v1/knowledge/docs      文档列表
# ============================================================

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_llm, get_embedding_model
from src.core.base_schema import PageResult, ResponseSchema
from src.core.cache import build_search_cache_key, get_json_cache, set_json_cache
from src.core.deps import PageParams, UserContext, get_current_user
from src.core.exceptions import ERR_BAD_REQUEST, ERR_NOT_FOUND, BizException
from src.core.logger import logger
from src.core.rate_limit import check_rate_limit
from src.infra.db import get_db
from src.infra.db_readonly import get_db_readonly
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
    rate_limit: None = Depends(check_rate_limit),
    user: UserContext = Depends(get_current_user),
    # ★ nl2sql 通道查 ALM 业务库（alm_issues 等），必须用只读 session 而非自有库
    alm_db: AsyncSession = Depends(get_db_readonly),
):
    """多通道知识检索：文档 + 图谱 + NL2SQL 并行查询（命中查询缓存则直接返回）"""
    cache_key = build_search_cache_key(
        question=req.question,
        channels=req.channels,
        doc_type=req.doc_type,
        model_code=req.model_code,
        use_hyde=req.use_hyde,
        role=user.role,
    )
    cached = await get_json_cache(cache_key)
    if cached:
        answer = cached["answer"]
        contexts = cached.get("contexts", [])
        logger.info(f"查询缓存命中: question={req.question[:20]}")
        # 缓存命中仍落审计（traceability），但跳过 LLM 调用/评估
        from src.knowledge.audit import QueryAuditLog
        QueryAuditLog.log(
            user_id=user.user_id,
            role=user.role,
            question=req.question,
            intent="multi_channel:cache_hit",
            channels=req.channels,
            answer_preview=answer,
            duration_ms=0,
        )
        return ResponseSchema(data=SearchResponse(
            question=req.question,
            answer=answer,
            channels=req.channels,
        ))

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
            db_session=alm_db if "nl2sql" in req.channels else None,
            channels=req.channels,
            role=user.role,
            use_hyde=req.use_hyde,
        )
    answer = result["answer"] if isinstance(result, dict) else result
    contexts = result.get("contexts", []) if isinstance(result, dict) else []

    # ── 写入查询缓存（文档/图谱相对静态，缓存安全；无上下文时不缓存空壳答案）──
    if contexts:
        await set_json_cache(cache_key, {"answer": answer, "contexts": contexts})

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


@router.post("/search-stream")
async def search_knowledge_stream(
    req: SearchRequest,
    request: Request,
    rate_limit: None = Depends(check_rate_limit),
    user: UserContext = Depends(get_current_user),
    # ★ nl2sql 通道查 ALM 业务库，必须用只读 session
    alm_db: AsyncSession = Depends(get_db_readonly),
):
    """多通道知识检索 — SSE 流式：检索进度事件 + 答案 token 流式输出。

    事件格式（data: {...}\n\n）：
      {"type":"progress","channel":"doc_rag","status":"ok|empty|failed|skipped","count":N}
      {"type":"delta","content":"答案片段"}
      {"type":"done","answer":"完整答案","contexts":[...]}
      {"type":"error","content":"错误信息"}
    """
    llm = get_llm()
    embedding_model = get_embedding_model()
    milvus_client = get_milvus_client()
    neo4j_driver = get_neo4j_driver()

    # ── Token 追踪 ──
    token_tracker = TokenTracker()
    llm = llm.with_config({"callbacks": [token_tracker]})

    from src.knowledge.audit import QueryAuditLog, Timer
    from src.knowledge.fusion import multi_channel_search

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        answer_chunks: list[str] = []
        stream_contexts: list = []

        def sink(msg: dict):
            queue.put_nowait(msg)

        async def run_to_queue():
            try:
                await multi_channel_search(
                    question=req.question,
                    llm=llm,
                    embedding_model=embedding_model,
                    milvus_client=milvus_client,
                    neo4j_driver=neo4j_driver,
                    db_session=alm_db if "nl2sql" in req.channels else None,
                    channels=req.channels,
                    role=user.role,
                    use_hyde=req.use_hyde,
                    event_sink=sink,
                )
            except Exception as e:
                logger.error(f"流式知识检索失败: {e}")
                await queue.put({"type": "error", "content": str(e)})
            finally:
                await queue.put(None)  # sentinel

        task = asyncio.ensure_future(run_to_queue())

        with Timer() as timer:
            try:
                while True:
                    msg = await queue.get()
                    if msg is None:
                        break
                    if msg.get("type") == "delta":
                        answer_chunks.append(msg.get("content", ""))
                    elif msg.get("type") == "done":
                        stream_contexts = msg.get("contexts") or []
                    yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            finally:
                await task  # ensure pipeline completes

        answer = "".join(answer_chunks)

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
        evaluator.evaluate_knowledge(question=req.question, answer=answer, contexts=stream_contexts, token_usage=token_tracker.usage)

        # ── Token 用量 ──
        usage = token_tracker.usage
        logger.info(
            f"[Token] 知识检索流式 input={usage['input_tokens']} output={usage['output_tokens']} "
            f"total={usage['total_tokens']} calls={usage['calls']} "
            f"cost=${usage['cost_usd']:.6f}"
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
