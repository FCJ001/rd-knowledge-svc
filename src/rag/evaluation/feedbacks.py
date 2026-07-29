# ============================================================
# 评估反馈收集：用户反馈 → 聚合统计 → 趋势分析
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
