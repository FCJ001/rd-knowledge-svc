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
    # ★ 建连/读写超时 + 连接数上限：Redis 故障时快速失败，不无界堆积连接
    socket_connect_timeout=settings.REDIS_SOCKET_TIMEOUT,
    socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
    max_connections=settings.REDIS_MAX_CONNECTIONS,
    health_check_interval=30,
)

_redis_client = redis.Redis(connection_pool=redis_pool)


async def get_redis_client() -> redis.Redis:
    """FastAPI Depends 注入用"""
    return _redis_client


def get_redis_sync_client() -> redis.Redis:
    """非 Depends 场景（韧性层熔断状态等）直接拿共享连接池的客户端。

    ★ 名字里的 sync 指"非 Depends 注入的同步取用方式"——
      返回的是 async 客户端，调用必须 await，不能在事件循环里直接调。"""
    return _redis_client


# 更准确的别名，新代码用这个
get_shared_redis_client = get_redis_sync_client
