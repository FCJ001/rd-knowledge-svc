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
    """投递一条入库任务。成功返回 True，Redis 异常或队列已满返回 False。

    ★ 深度护栏：逼近 maxlen 就拒收（调用方按入队失败补偿 + 503 处理），
      绝不让 maxlen 把还没投递的最老任务静默裁掉 —— maxlen 裁剪不区分
      消息是否已投递，被裁掉的任务在 DB 里永远停在 queued，且 PEL 回收
      只能救「已投递」的消息，救不了「从未投递就被裁掉」的。
      XLEN 与 XADD 之间存在并发窗口，突发下可能少量越过上限 —— 护栏
      只求「不丢任务」，越限量级 = 并发 producer 数，可接受。
    """
    try:
        client = _client()
        await ensure_group(client)
        depth = await client.xlen(settings.INGEST_STREAM)
        if depth >= settings.INGEST_STREAM_MAX_LEN:
            logger.error(
                f"入库队列已满 depth={depth}/{settings.INGEST_STREAM_MAX_LEN}，拒绝入队"
                f" job_id={payload.get('job_id', '-')}"
            )
            return False
        await client.xadd(
            settings.INGEST_STREAM,
            {"payload": json.dumps(payload, ensure_ascii=False)},
            maxlen=settings.INGEST_STREAM_MAX_LEN,
        )
        return True
    except Exception as e:
        logger.error(f"入库任务入队失败: {e}")
        return False
