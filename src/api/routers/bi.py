# ============================================================
# ChatBI API — NL2SQL 查询 + 图表推荐
#
# POST   /api/v1/bi/query              自然语言查数据 → 结果 + 图表配置
# GET    /api/v1/bi/history/{session_id}    查看对话历史
# DELETE /api/v1/bi/history/{session_id}    清除对话历史
# ============================================================

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_llm, get_embedding_model
from src.core.base_schema import ResponseSchema
from src.core.deps import UserContext, get_current_user
from src.core.logger import logger
from src.infra.db_readonly import get_db_readonly
from src.infra.redis_cache import get_redis_client
from src.nl2sql.chart_advisor import recommend_chart
from src.nl2sql.echarts_builder import to_echarts_option
from src.nl2sql.engine import ConversationContext, run_query

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


# ── 对话上下文存储（内存，同进程内）───────────────────────────────────────

_ctx_store: dict[str, ConversationContext] = {}


def _get_or_create_ctx(session_id: str) -> ConversationContext:
    if session_id not in _ctx_store:
        _ctx_store[session_id] = ConversationContext()
    return _ctx_store[session_id]


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/query", response_model=ResponseSchema[BIQueryResponse])
async def bi_query(
    req: BIQueryRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_readonly),
):
    """自然语言数据查询，返回 SQL + 数据 + 图表配置"""
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

    # 图表推荐
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


@router.get("/history/{session_id}", response_model=ResponseSchema[dict])
async def get_history(
    session_id: str,
    user: UserContext = Depends(get_current_user),
):
    """获取会话的 NL2SQL 对话历史"""
    redis = await get_redis_client()
    from src.knowledge.conversation import load_conversation_context

    history = await load_conversation_context(redis, user.user_id, session_id)

    # 同时合并内存中的 QueryResult 历史
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
