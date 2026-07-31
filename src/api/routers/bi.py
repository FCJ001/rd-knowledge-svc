# ============================================================
# ChatBI API — NL2SQL 查询 + 图表推荐
#
# POST   /api/v1/bi/query               自然语言查数据 → 结果 + 图表配置 (旧引擎)
# POST   /api/v1/bi/query-stream        新 RAG 流水线 SSE 流式查询
# GET    /api/v1/bi/history/{session_id}    查看对话历史
# DELETE /api/v1/bi/history/{session_id}    清除对话历史
# ============================================================

import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_llm, get_embedding_model
from src.core.base_schema import ResponseSchema
from src.core.deps import UserContext, get_current_user
from src.core.logger import logger
from src.infra.db import AsyncSessionLocal
from src.infra.db_readonly import get_db_readonly
from src.infra.es_client import get_es_client
from src.infra.milvus_client import get_milvus_client
from src.nl2sql.chart_advisor import recommend_chart
from src.nl2sql.echarts_builder import to_echarts_option
from src.nl2sql.engine import ConversationContext, run_query
from src.nl2sql.repositories import (
    PgMetaRepository,
    MilvusColumnRepository,
    MilvusMetricRepository,
    ESValueRepository,
)

router = APIRouter(prefix="/api/v1/bi", tags=["ChatBI"])


# ── Request / Response models ────────────────────────────────────────────

class BIQueryRequest(BaseModel):
    question: str = Field(..., description="自然语言数据查询")
    session_id: str = Field(default="default", description="会话ID，同会话多轮下钻")
    with_chart: bool = Field(default=True, description="是否返回图表配置")


class BIQueryResponse(BaseModel):
    question: str
    sql: str = ""
    data: list[dict] = []
    columns: list[str] = []
    row_count: int = 0
    summary: str = ""
    chart: dict | None = None
    success: bool = True
    error: str = ""


from src.infra.redis_cache import get_redis_client


# ── 对话上下文存储（内存，同进程内）───────────────────────────────────────

_ctx_store: dict[str, ConversationContext] = {}


def _get_or_create_ctx(session_id: str) -> ConversationContext:
    if session_id not in _ctx_store:
        _ctx_store[session_id] = ConversationContext()
    return _ctx_store[session_id]


# ════════════════════════════════════════════════════════════════
# 旧引擎 — 一次性 JSON 响应（保留兼容）
# ════════════════════════════════════════════════════════════════

@router.post("/query", response_model=ResponseSchema[BIQueryResponse])
async def bi_query(
    req: BIQueryRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_readonly),
):
    """自然语言数据查询，返回 SQL + 数据 + 图表配置（旧引擎）"""
    llm = get_llm()
    ctx = _get_or_create_ctx(req.session_id)

    result = await run_query(
        question=req.question,
        llm=llm,
        db=db,
        role=user.role,
        owner_domain_id=user.owner_domain_id,
        business_line=user.business_line,
        context=ctx,
    )

    resp = BIQueryResponse(
        question=req.question,
        sql=result.sql,
        data=result.data if result.success else [],
        columns=result.columns,
        row_count=result.row_count,
        summary=result.summary,
        success=result.success,
        error=result.error,
    )

    if result.success and result.data and req.with_chart:
        try:
            chart_config = await recommend_chart(
                question=req.question,
                data=result.data,
                columns=result.columns,
                llm=llm,
            )
            resp.chart = to_echarts_option(result.data, chart_config)
        except Exception as e:
            logger.warning(f"图表生成失败: {e}")

    return ResponseSchema(data=resp)


# ════════════════════════════════════════════════════════════════
# 新 RAG 流水线 — SSE 流式响应
# ════════════════════════════════════════════════════════════════

def _make_json_safe(obj):
    """递归将 dataclass 对象转换为 JSON 可序列化的 dict"""
    from dataclasses import fields, is_dataclass

    if is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for f in fields(obj):
            result[f.name] = _make_json_safe(getattr(obj, f.name))
        return result
    elif isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(item) for item in obj]
    else:
        return obj


async def _build_pipeline_context(alm_db: AsyncSession) -> dict:
    """构建 DataAgentContext，供流水线节点使用"""
    milvus = get_milvus_client()
    es = await get_es_client()

    # PG 元数据需要 rd_knowledge 库的独立 session
    knowledge_db = AsyncSessionLocal()

    ctx = {
        "llm": get_llm(),
        "embedding_model": get_embedding_model(),
        "milvus_client": milvus,
        "milvus_column_repo": MilvusColumnRepository(milvus),
        "milvus_metric_repo": MilvusMetricRepository(milvus),
        "es_client": es,
        "es_value_repo": ESValueRepository(es),
        "pg_meta_repo": PgMetaRepository(knowledge_db),
        "dw_db_session": alm_db,
    }
    return ctx, knowledge_db


@router.post("/query-stream")
async def bi_query_stream(
    req: BIQueryRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
    alm_db: AsyncSession = Depends(get_db_readonly),
):
    """自然语言数据查询 — SSE 流式返回各节点执行状态 + 最终结果"""
    import asyncio as _asyncio
    ctx, knowledge_db = await _build_pipeline_context(alm_db)

    from src.nl2sql.pipeline import run_pipeline

    async def event_stream():
        queue: _asyncio.Queue = _asyncio.Queue()

        # 注入 writer 回调，节点通过它推送进度/结果事件
        def writer(msg: dict):
            queue.put_nowait({"__progress__": msg})
        ctx["writer"] = writer

        async def run_to_queue():
            try:
                async for event in run_pipeline(req.question, ctx):
                    await queue.put(event)
            except Exception as e:
                logger.error(f"Pipeline 执行失败: {e}")
                await queue.put({"__error__": str(e)})
            finally:
                await queue.put(None)  # sentinel

        task = _asyncio.ensure_future(run_to_queue())

        try:
            node_count = 0
            while True:
                event = await queue.get()
                if event is None:
                    break
                if "__error__" in event:
                    yield f"data: {json.dumps({'node': 'error', 'step': node_count, 'data': {'error': event['__error__']}}, ensure_ascii=False)}\n\n"
                    break
                if "__progress__" in event:
                    yield f"data: {json.dumps(event['__progress__'], ensure_ascii=False)}\n\n"
                    continue

                node_count += 1
                node_name = list(event.keys())[0] if event else "unknown"
                node_data = event.get(node_name, {})
                safe_data = _make_json_safe(node_data)

                payload = {
                    "node": node_name,
                    "step": node_count,
                    "data": safe_data if safe_data else {},
                }

                if node_name == "execute_sql" and isinstance(node_data, dict) and node_data.get("result_data"):
                    try:
                        chart_config = await recommend_chart(
                            question=req.question,
                            data=node_data["result_data"],
                            columns=node_data.get("result_columns", []),
                            llm=ctx["llm"],
                        )
                        payload["chart"] = to_echarts_option(
                            node_data["result_data"], chart_config
                        )
                    except Exception as e:
                        logger.warning(f"图表生成失败: {e}")

                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        finally:
            await task  # ensure pipeline completes
            await knowledge_db.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ════════════════════════════════════════════════════════════════
# 会话历史
# ════════════════════════════════════════════════════════════════

@router.get("/history/{session_id}", response_model=ResponseSchema[dict])
async def get_history(
    session_id: str,
    user: UserContext = Depends(get_current_user),
):
    """获取会话的 NL2SQL 对话历史"""
    redis = await get_redis_client()
    from src.knowledge.conversation import load_conversation_context

    history = await load_conversation_context(redis, user.user_id, session_id)

    ctx = _ctx_store.get(session_id)
    sql_history = []
    if ctx:
        for r in ctx.history:
            sql_history.append({
                "question": r.question,
                "sql": r.sql,
                "row_count": r.row_count,
                "summary": r.summary[:200] if r.summary else "",
                "success": r.success,
                "error": r.error,
            })

    return ResponseSchema(data={
        "session_id": session_id,
        "conversation_history": history,
        "sql_history": sql_history,
    })


@router.delete("/history/{session_id}", response_model=ResponseSchema[dict])
async def clear_history(
    session_id: str,
    user: UserContext = Depends(get_current_user),
):
    """清除会话历史"""
    redis = await get_redis_client()
    key = f"alm_ctx:{user.user_id}:{session_id}"
    await redis.delete(key)

    if session_id in _ctx_store:
        del _ctx_store[session_id]

    return ResponseSchema(data={"session_id": session_id, "status": "cleared"})
