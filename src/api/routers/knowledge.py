# ============================================================
# 知识检索 API
#
# POST /api/v1/knowledge/search        多通道检索
# POST /api/v1/knowledge/search-stream 多通道检索（SSE 流式）
# POST /api/v1/knowledge/feedback      用户反馈
# GET  /api/v1/knowledge/docs          文档列表
# ============================================================

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from src.api.deps import get_embedding_model, get_llm
from src.core.base_schema import PageResult, ResponseSchema
from src.core.cache import build_search_cache_key, get_json_cache, set_json_cache
from src.core.config import get_settings
from src.core.deps import PageParams, UserContext, get_current_user
from src.core.logger import logger
from src.core.quota import add_tokens, check_token_quota
from src.core.rate_limit import check_rate_limit
from src.infra.db import get_db
from src.infra.milvus_client import get_milvus_client
from src.infra.neo4j_client import get_neo4j_driver
from src.knowledge.model import KnowledgeDoc, KnowledgeFeedback
from src.rag.evaluation.async_tracker import get_async_evaluator
from src.rag.evaluation.guardrails import check_output
from src.rag.evaluation.token_tracker import TokenTracker

router = APIRouter(prefix="/api/v1/knowledge", tags=["知识检索"])

settings = get_settings()

# SSE 事件队列上限：断连后生产端继续入队时防内存膨胀；队满丢弃进度类事件
_SSE_QUEUE_MAX = 500
# 消费队列的等待上限（秒）：到点回到断连检查，同时给网关兜底心跳之外的活性
_SSE_QUEUE_POLL_TIMEOUT = 20


# ── Request / Response models ────────────────────────────────────────────

class SearchRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    channels: list[str] = Field(
        default=["doc_rag"],
        description="检索通道: doc_rag / graph_rag / nl2sql（graph_rag 需图谱已建，默认不启用）"
    )
    doc_type: str = Field(default="", description="文档类型过滤: repair_manual/spec_doc/tsb/issue_case")
    model_code: str = Field(default="", description="车型代码过滤")
    use_hyde: bool = Field(default=False, description="是否启用 HyDE（假设性文档嵌入，默认关）")


class SearchResponse(BaseModel):
    question: str
    answer: str
    channels: list[str]
    image_urls: list[str] = Field(default=[], description="检索命中的图片 URL（可点击查看原图）")
    status: str = Field(
        default="ok",
        description="ok=全部通道正常 / partial=有通道降级 / empty=无相关内容 / low_confidence=置信度不足已拒答",
    )
    failed_channels: list[str] = Field(default=[], description="本次失败的通道，便于前端提示降级")


class FeedbackRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    answer_preview: str = Field(default="", description="答案摘要")
    rating: int = Field(default=0, description="评分: 1 赞 / -1 踩 / 0 中性")
    comment: str = Field(default="", description="评语")
    intent: str = Field(default="", description="意图分类")
    channels: str = Field(default="", description="使用的检索通道")
    trace_id: str = Field(default="", description="关联请求")


# ── Endpoints ────────────────────────────────────────────────────────────

def _enforce_model_code(req: SearchRequest) -> None:
    """RAG_REQUIRE_MODEL_CODE=true 时拒绝无车型的检索请求。

    车型过滤为空时检索会退化成"全车型混合"，跨车型串味是安全事故而非体验问题；
    静默放开比直接拒绝更危险，所以生产可把开关打开。
    """
    if settings.RAG_REQUIRE_MODEL_CODE and not req.model_code.strip():
        raise HTTPException(
            status_code=400,
            detail="本服务要求指定 model_code（车型隔离），请补充后重试",
        )


@router.post("/search", response_model=ResponseSchema[SearchResponse])
async def search_knowledge(
    req: SearchRequest,
    rate_limit: None = Depends(check_rate_limit),
    quota: None = Depends(check_token_quota),
    user: UserContext = Depends(get_current_user),
):
    """多通道知识检索：文档 + 图谱 + NL2SQL(rd-chatBI) 并行查询（命中查询缓存则直接返回）"""
    _enforce_model_code(req)
    cache_key = build_search_cache_key(
        question=req.question,
        channels=req.channels,
        doc_type=req.doc_type,
        model_code=req.model_code,
        use_hyde=req.use_hyde,
        role=user.role,
        owner_domain_id=user.owner_domain_id,
        business_line=user.business_line,
    )
    cached = await get_json_cache(cache_key)
    if cached:
        answer = cached["answer"]
        contexts = cached.get("contexts", [])
        image_urls = cached.get("image_urls", [])
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
            image_urls=image_urls,
        ))

    llm = get_llm()
    embedding_model = get_embedding_model()
    milvus_client = get_milvus_client()
    neo4j_driver = get_neo4j_driver()

    # ── Token 追踪 ──
    token_tracker = TokenTracker()
    llm = llm.with_config({"callbacks": [token_tracker]})

    from src.knowledge.audit import QueryAuditLog, Timer
    from src.knowledge.fusion import RetrievalUnavailableError, multi_channel_search

    with Timer() as timer:
        try:
            result = await multi_channel_search(
                question=req.question,
                llm=llm,
                embedding_model=embedding_model,
                milvus_client=milvus_client,
                neo4j_driver=neo4j_driver,
                channels=req.channels,
                role=user.role,
                use_hyde=req.use_hyde,
                owner_domain_id=user.owner_domain_id,
                business_line=user.business_line,
                doc_type=req.doc_type,
                model_code=req.model_code,
            )
        except RetrievalUnavailableError as e:
            # ★ 全部通道故障 ≠ 知识库没有内容。前者按 503 应答，
            #   既让用户看到"服务不可用"而不是"查无此文"，也让 5xx 告警生效。
            logger.error(f"检索服务不可用: {e}")
            QueryAuditLog.log(
                user_id=user.user_id,
                role=user.role,
                question=req.question,
                intent="multi_channel:unavailable",
                channels=req.channels,
                answer_preview="",
                duration_ms=timer.elapsed_ms,
            )
            raise HTTPException(
                status_code=503,
                detail="检索服务暂时不可用（知识库依赖异常），请稍后重试",
            )
    answer = result["answer"] if isinstance(result, dict) else result
    contexts = result.get("contexts", []) if isinstance(result, dict) else []
    image_urls = result.get("image_urls", []) if isinstance(result, dict) else []
    status = result.get("status", "ok") if isinstance(result, dict) else "ok"
    failed_channels = result.get("failed_channels", []) if isinstance(result, dict) else []

    # ── 写入查询缓存（文档/图谱相对静态，缓存安全；无上下文时不缓存空壳答案）──
    if contexts:
        await set_json_cache(cache_key, {"answer": answer, "contexts": contexts, "image_urls": image_urls})

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
    evaluator.evaluate_knowledge(question=req.question, answer=answer, contexts=contexts,
                                 token_usage=token_tracker.usage, latency_ms=timer.elapsed_ms)

    # ── Token 用量 ──
    usage = token_tracker.usage
    logger.info(
        f"[Token] 知识检索 input={usage['input_tokens']} output={usage['output_tokens']} "
        f"total={usage['total_tokens']} calls={usage['calls']} "
        f"cost=${usage['cost_usd']:.6f}"
    )
    # 成本维度记账（配额拦截在下次请求生效，见 core/quota.py）
    await add_tokens(
        user.user_id,
        int(usage["total_tokens"]),
        float(usage.get("cost_usd") or 0.0),
        model=str(usage.get("model") or ""),
    )

    return ResponseSchema(data=SearchResponse(
        question=req.question,
        answer=answer,
        channels=req.channels,
        image_urls=image_urls,
        status=status,
        failed_channels=failed_channels,
    ))


@router.post("/search-stream")
async def search_knowledge_stream(
    req: SearchRequest,
    request: Request,
    rate_limit: None = Depends(check_rate_limit),
    quota: None = Depends(check_token_quota),
    user: UserContext = Depends(get_current_user),
):
    """多通道知识检索 — SSE 流式：检索进度事件 + 答案 token 流式输出。

    事件格式（data: {...}\n\n）：
      {"type":"progress","channel":"doc_rag","status":"ok|empty|failed|skipped|denied","count":N}
        （denied=该通道因权限被拒（如图谱通道对非 admin），发生在检索发起之前）
      {"type":"delta","content":"答案片段"}
      {"type":"done","answer":"完整答案","contexts":[...],
       "status":"ok|empty|partial|low_confidence"}
      {"type":"warning","content":"幻觉检测提示"}
      {"type":"degraded","failed_channels":[...]}  全部通道故障（随后以 error 事件告知用户）
      {"type":"error","content":"错误信息"}
    """
    llm = get_llm()
    embedding_model = get_embedding_model()
    milvus_client = get_milvus_client()
    neo4j_driver = get_neo4j_driver()

    _enforce_model_code(req)

    # ── Token 追踪 ──
    token_tracker = TokenTracker()
    llm = llm.with_config({"callbacks": [token_tracker]})

    from src.knowledge.audit import QueryAuditLog, Timer
    from src.knowledge.fusion import RetrievalUnavailableError, multi_channel_search

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue(maxsize=_SSE_QUEUE_MAX)
        answer_chunks: list[str] = []
        stream_contexts: list = []

        def sink(msg: dict):
            """背压分级：进度事件满了可丢；答案 token / done / error 必达。

            必达事件遇队满时挤掉最旧的进度事件腾位——旧实现无差别丢弃，
            队满时 error 事件会被静默吞掉，用户拿到的是无提示的截断答案。
            queue._queue 是 asyncio.Queue 的内部 deque：公开 API 没有
            「按类型移除」的能力；scan/del 之间无 await，单事件循环内安全。
            """
            try:
                queue.put_nowait(msg)
                return
            except asyncio.QueueFull:
                pass
            if msg.get("type") == "progress":
                return  # 进度事件可丢
            q = queue._queue
            for i, old in enumerate(q):
                if isinstance(old, dict) and old.get("type") == "progress":
                    del q[i]
                    break
            else:
                # 极端情况：整队都是必达事件——丢最旧的保最新的
                if queue.qsize() >= _SSE_QUEUE_MAX:
                    queue.get_nowait()
            queue.put_nowait(msg)
            logger.warning("SSE 事件队列已满，为必达事件挤掉了一条旧事件")

        async def run_to_queue():
            try:
                await multi_channel_search(
                    question=req.question,
                    llm=llm,
                    embedding_model=embedding_model,
                    milvus_client=milvus_client,
                    neo4j_driver=neo4j_driver,
                    channels=req.channels,
                    role=user.role,
                    use_hyde=req.use_hyde,
                    owner_domain_id=user.owner_domain_id,
                    business_line=user.business_line,
                    doc_type=req.doc_type,
                    model_code=req.model_code,
                    event_sink=sink,
                )
            except asyncio.CancelledError:
                raise
            except RetrievalUnavailableError:
                # 依赖故障：给用户明确的"服务不可用"，而不是让前端看到"未找到"
                logger.error("流式检索：全部通道不可用")
                sink({"type": "error",
                      "content": "检索服务暂时不可用（知识库依赖异常），请稍后重试"})
            except Exception as e:
                # ★ 内部细节只进日志，前端只收通用错误信息（防栈/连接串泄露）
                logger.exception(f"流式知识检索失败: {e}")
                sink({"type": "error", "content": "检索服务内部错误，请稍后重试"})
            finally:
                try:
                    queue.put_nowait(None)  # sentinel
                except asyncio.QueueFull:
                    await queue.put(None)

        task = asyncio.ensure_future(run_to_queue())
        cancelled_by_client = False

        with Timer() as timer:
            try:
                while True:
                    # 定期回到断连检查：客户端断开立即取消管线，LLM token 不白烧
                    if await request.is_disconnected():
                        logger.info("SSE 客户端已断开，取消检索管线")
                        cancelled_by_client = True
                        task.cancel()
                        break
                    try:
                        msg = await asyncio.wait_for(
                            queue.get(), timeout=_SSE_QUEUE_POLL_TIMEOUT,
                        )
                    except TimeoutError:
                        continue
                    if msg is None:
                        break
                    if msg.get("type") == "delta":
                        answer_chunks.append(msg.get("content", ""))
                    elif msg.get("type") == "done":
                        stream_contexts = msg.get("contexts") or []
                    yield {"data": json.dumps(msg, ensure_ascii=False)}
            finally:
                # 管线结果善后：取消的要吞掉 CancelledError，正常的等它收尾
                if not task.done():
                    task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    if cancelled_by_client:
                        logger.info("检索管线已随客户端断开而取消")
                    else:
                        logger.exception("检索管线收尾异常")

        if cancelled_by_client:
            return

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
        evaluator.evaluate_knowledge(question=req.question, answer=answer, contexts=stream_contexts,
                                     token_usage=token_tracker.usage, latency_ms=timer.elapsed_ms)

        # ── Token 用量 ──
        usage = token_tracker.usage
        logger.info(
            f"[Token] 知识检索流式 input={usage['input_tokens']} output={usage['output_tokens']} "
            f"total={usage['total_tokens']} calls={usage['calls']} "
            f"cost=${usage['cost_usd']:.6f}"
        )
        # 成本维度记账（配额拦截在下次请求生效，见 core/quota.py）
        await add_tokens(
            user.user_id,
            int(usage["total_tokens"]),
            float(usage.get("cost_usd") or 0.0),
            model=str(usage.get("model") or ""),
        )

    # EventSourceResponse：自带 ping 心跳（防网关掐空闲连接）+ 规范的 SSE 编码
    return EventSourceResponse(
        event_stream(),
        ping=15,
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
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """文档元数据列表。

    ★ 列表同样按角色过滤（与检索同一套 ACL）：非 admin 只能看到 acl_roles
    包含自己角色的文档——列表若对无权文档只藏正文不藏标题，本身就是泄露。
    acl_roles 为 NULL 的存量文档（未回填）对非 admin 不可见（fail-closed）。
    """
    from sqlalchemy import func, select

    from src.knowledge.acl import ADMIN_ROLE, ASSIGNABLE_ROLES

    if user.role != ADMIN_ROLE and user.role not in ASSIGNABLE_ROLES:
        # 词表外角色（token 被篡改/上游新增角色未同步）：不查库直接给空，
        # 通配符式取值（如 "%"）进 LIKE 模式等于权限开关。
        logger.warning(f"文档列表被拒: role={user.role!r} 不在 ACL 词表内（fail-closed）")
        return ResponseSchema(data=PageResult(items=[], total=0, page=page.page, page_size=page.page_size))

    base = select(KnowledgeDoc).where(KnowledgeDoc.status != "deleted")
    if page.keyword:
        from sqlalchemy import or_
        kw = f"%{page.keyword}%"
        base = base.where(or_(
            KnowledgeDoc.doc_name.like(kw),
            KnowledgeDoc.doc_type.like(kw),
            KnowledgeDoc.category.like(kw),
        ))
    if user.role != ADMIN_ROLE:
        # 首尾加逗号再做 LIKE，保证角色名是完整匹配项而非子串
        base = base.where(
            func.concat(",", KnowledgeDoc.acl_roles, ",").like(f"%,{user.role},%")
        )

    total = (await db.execute(
        select(func.count()).select_from(base.subquery())
    )).scalar_one()
    result = await db.execute(
        base.offset(page.offset).limit(page.page_size).order_by(KnowledgeDoc.id.asc())
    )
    items = list(result.scalars().all())

    docs = [
        {
            "id": str(d.id),
            "doc_id": d.doc_id,
            "doc_name": d.doc_name,
            "doc_type": d.doc_type,
            "category": d.category,
            "business_line": d.business_line,
            "model_code": d.model_code,
            "acl_roles": [r for r in (d.acl_roles or "").split(",") if r],
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
