# ============================================================
# 入库任务队列（Redis Stream）
#
# producer（API 进程）：enqueue_ingest_job → XADD
# consumer（worker 进程）：XREADGROUP 消费（见 worker.py）
#
# ★ 入队失败不静默：返回 False，由调用方决定如何返回（503 并提示重试）。
# ============================================================

from __future__ import annotations

import json

from loguru import logger

from src.core.config import get_settings

settings = get_settings()


def _client():
    import redis.asyncio as aioredis

    from src.infra.redis_cache import redis_pool

    return aioredis.Redis(connection_pool=redis_pool)


async def ensure_group(client) -> None:
    """创建 Stream + 消费者组（幂等：已存在则忽略 BUSYGROUP）。"""
    try:
        await client.xgroup_create(
            settings.INGEST_STREAM,
            settings.INGEST_CONSUMER_GROUP,
            id="0",
            mkstream=True,
        )
    except Exception as e:
        if "BUSYGROUP" in str(e):
            return
        raise


async def enqueue_ingest_job(payload: dict) -> bool:
    """投递一条入库任务。成功返回 True，Redis 异常返回 False。"""
    try:
        client = _client()
        await ensure_group(client)
        await client.xadd(
            settings.INGEST_STREAM,
            {"payload": json.dumps(payload, ensure_ascii=False)},
            maxlen=settings.INGEST_STREAM_MAX_LEN,
        )
        return True
    except Exception as e:
        logger.error(f"入库任务入队失败: {e}")
        return False
