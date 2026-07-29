# ============================================================
# 知识更新通知
# ============================================================

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.knowledge.model import KnowledgeNotification
from src.core.logger import logger


async def notify_doc_update(
    doc_name: str,
    doc_type: str,
    category: str,
    action: str,  # upload / update / delete
    db: AsyncSession,
) -> KnowledgeNotification:
    notif = KnowledgeNotification(
        doc_name=doc_name,
        doc_type=doc_type,
        category=category,
        action=action,
        is_read=False,
    )
    db.add(notif)
    await db.flush()
    logger.info(f"知识库通知: {action} {doc_name}")
    return notif


async def get_unread_notifications(db: AsyncSession) -> list[KnowledgeNotification]:
    result = await db.execute(
        select(KnowledgeNotification)
        .where(KnowledgeNotification.is_read == False)
        .order_by(desc(KnowledgeNotification.created_at))
    )
    return list(result.scalars().all())


async def mark_notifications_read(ids: list[int], db: AsyncSession) -> None:
    for nid in ids:
        notif = await db.get(KnowledgeNotification, nid)
        if notif:
            notif.is_read = True
    await db.flush()
    logger.info(f"标记 {len(ids)} 条通知为已读")
