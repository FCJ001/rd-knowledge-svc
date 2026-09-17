# ============================================================
# 用户反馈 API
#
# POST /api/v1/feedback                  提交反馈（👍/👎）
# GET  /api/v1/feedback/stats            反馈统计
# GET  /api/v1/feedback/trace/{id}       按 trace 查反馈
# ============================================================

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.base_schema import ResponseSchema
from src.core.deps import UserContext, get_current_user
from src.infra.db import get_db
from src.knowledge.feedbacks import (
    collect_feedback,
    get_feedback_stats,
    get_trace_feedback_stats,
)

router = APIRouter(prefix="/api/v1/feedback", tags=["用户反馈"])


class FeedbackRequest(BaseModel):
    question: str = Field(default="", description="用户问题")
    answer_preview: str = Field(default="", description="答案摘要")
    rating: int = Field(..., ge=-1, le=1, description="1=赞, -1=踩, 0=中性")
    comment: str = Field(default="", description="补充说明")
    intent: str = Field(default="", description="意图分类")
    channels: str = Field(default="", description="检索通道")
    trace_id: str = Field(default="", description="关联的请求trace_id")


@router.post("", response_model=ResponseSchema[dict])
async def submit_feedback(
    req: FeedbackRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """提交用户反馈。user_id 取自身份上下文（header 模式取 X-User-Id，jwt 模式取 claims），
    不再信任请求体自报的身份。"""
    fb = await collect_feedback(
        db=db,
        user_id=user.user_id or "anonymous",
        question=req.question,
        answer_preview=req.answer_preview,
        rating=req.rating,
        comment=req.comment,
        intent=req.intent,
        channels=req.channels,
        trace_id=req.trace_id,
    )
    await db.commit()
    return ResponseSchema(data={
        "id": fb.id,
        "trace_id": req.trace_id,
        "rating": req.rating,
    })


@router.get("/stats", response_model=ResponseSchema[dict])
async def feedback_stats(
    days: int = 30,
    db: AsyncSession = Depends(get_db),
):
    """近 N 天反馈统计"""
    stats = await get_feedback_stats(db, days=days)
    return ResponseSchema(data=stats)


@router.get("/trace/{trace_id}", response_model=ResponseSchema[dict])
async def trace_feedback(
    trace_id: str,
    db: AsyncSession = Depends(get_db),
):
    """查询指定 trace 的用户反馈"""
    result = await get_trace_feedback_stats(db, trace_id)
    return ResponseSchema(data=result)
