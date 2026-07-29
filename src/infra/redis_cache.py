# ============================================================
# Redis 连接 — 多轮上下文存储
#
# 项目二不做 Agent，只需业务池（decode_responses=True），
# 比 tiangong-agent 少一个 checkpointer 连接池。
# key 前缀 alm_ctx:，与 P1 的 triage:* 隔离。
# ============================================================

import redis.asyncio as redis

from src.core.config import get_settings

settings = get_settings()

redis_pool = redis.ConnectionPool(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    password=settings.REDIS_PASSWORD or None,
    decode_responses=True,
    encoding="utf-8",
)

_redis_client = redis.Redis(connection_pool=redis_pool)


async def get_redis_client() -> redis.Redis:
    """FastAPI Depends 注入用"""
    return _redis_client
