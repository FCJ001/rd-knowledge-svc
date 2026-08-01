# ============================================================
# 用户反馈收集：用户反馈 → 聚合统计 → 趋势分析
# ============================================================

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.knowledge.model import KnowledgeFeedback


async def collect_feedback(
    db: AsyncSession,
    user_id: str,
    question: str,
    answer_preview: str,
    rating: int,
    comment: str = "",
    intent: str = "",
    channels: str = "",
    trace_id: str = "",
) -> KnowledgeFeedback:
    """记录单条用户反馈"""
    fb = KnowledgeFeedback(
        user_id=user_id,
        question=question,
        answer_preview=answer_preview,
        rating=rating,
        comment=comment or None,
        intent=intent or None,
        channels=channels or None,
        trace_id=trace_id or None,
    )
    db.add(fb)
    await db.flush()
    return fb


async def get_feedback_stats(
    db: AsyncSession,
    days: int = 30,
) -> dict:
    """近 N 天反馈统计"""
    since = datetime.utcnow() - __import__("datetime").timedelta(days=days)

    total_result = await db.execute(
        select(func.count(KnowledgeFeedback.id)).where(
            KnowledgeFeedback.created_at >= since,
        )
    )
    total = total_result.scalar_one()

    praise_result = await db.execute(
        select(func.count(KnowledgeFeedback.id)).where(
            KnowledgeFeedback.created_at >= since,
            KnowledgeFeedback.rating == 1,
        )
    )
    praise = praise_result.scalar_one()

    negative_result = await db.execute(
        select(func.count(KnowledgeFeedback.id)).where(
            KnowledgeFeedback.created_at >= since,
            KnowledgeFeedback.rating == -1,
        )
    )
    negative = negative_result.scalar_one()

    return {
        "days": days,
        "total": total,
        "praise": praise,
        "negative": negative,
        "praise_rate": round(praise / total, 3) if total > 0 else 0,
        "negative_rate": round(negative / total, 3) if total > 0 else 0,
    }


async def get_channel_stats(db: AsyncSession, days: int = 30) -> list[dict]:
    """按检索通道统计好评率"""
    since = datetime.utcnow() - __import__("datetime").timedelta(days=days)

    result = await db.execute(
        select(
            KnowledgeFeedback.channels,
            func.count(KnowledgeFeedback.id).label("total"),
            func.sum(
                func.case((KnowledgeFeedback.rating == 1, 1), else_=0)
            ).label("praise_count"),
        )
        .where(KnowledgeFeedback.created_at >= since)
        .group_by(KnowledgeFeedback.channels)
    )
    rows = result.all()
    return [
        {
            "channel": row.channels or "unknown",
            "total": row.total,
            "praise_count": row.praise_count,
            "praise_rate": round(row.praise_count / row.total, 3) if row.total > 0 else 0,
        }
        for row in rows
    ]


async def get_feedback_by_trace_id(
    db: AsyncSession,
    trace_id: str,
) -> list[KnowledgeFeedback]:
    """按 trace_id 查询用户反馈，用于关联 TruLens trace"""
    result = await db.execute(
        select(KnowledgeFeedback).where(KnowledgeFeedback.trace_id == trace_id)
    )
    return list(result.scalars().all())


async def link_feedback_to_trace(
    db: AsyncSession,
    trace_id: str,
    rating: int = 0,
    comment: str = "",
) -> KnowledgeFeedback | None:
    """查找已有反馈并追加关联信息，或返回 None 表示无匹配 trace"""
    feedbacks = await get_feedback_by_trace_id(db, trace_id)
    if not feedbacks:
        return None
    fb = feedbacks[0]
    if rating != 0:
        fb.rating = rating
    if comment:
        fb.comment = (fb.comment or "") + f" | {comment}"
    await db.flush()
    return fb


async def get_trace_feedback_stats(
    db: AsyncSession,
    trace_id: str,
) -> dict:
    """获取指定 trace 的反馈统计（供 Dashboard 使用）"""
    feedbacks = await get_feedback_by_trace_id(db, trace_id)
    if not feedbacks:
        return {"trace_id": trace_id, "feedback_count": 0}

    ratings = [fb.rating for fb in feedbacks]
    return {
        "trace_id": trace_id,
        "feedback_count": len(feedbacks),
        "praise_count": sum(1 for r in ratings if r == 1),
        "negative_count": sum(1 for r in ratings if r == -1),
        "latest_comment": feedbacks[-1].comment or "",
        "created_at": feedbacks[0].created_at.isoformat() if feedbacks[0].created_at else "",
    }
