# ============================================================
# API 限流
#
# 后端可切换：
#   - redis（默认）：ZSET 滑动窗口，Lua 脚本原子 check+add，跨 worker 精确
#   - memory：进程内 deque（单 worker 调试 / 无 Redis 环境）
#
# ★ fail-open：Redis 不可用时放行并告警，绝不让限流拖垮主链路。
# ============================================================

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from uuid import uuid4

from fastapi import Depends, HTTPException, Request
from loguru import logger

from src.core.config import get_settings
from src.core.deps import UserContext, get_current_user
from src.core.metrics import RATE_LIMIT_REJECTED

settings = get_settings()


class SlidingWindowRateLimiter:
    """按 key 的滑动窗口限流（进程内 deque 实现）。

    保留为：算法参照 + 单进程/无 Redis 后端的回退实现。
    """

    # 增量清理参数：每 N 次判定触发一次，单次最多扫 M 个 key——
    # 分摊清理开销，避免请求路径出现周期性尖刺
    _OPS_PER_SWEEP = 512
    _SWEEP_KEY_CAP = 256

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()
        self._ops = 0

    async def allow(self, key: str) -> bool:
        """记录一次请求；窗口内未超限返回 True，否则返回 False。"""
        await self._maybe_cleanup()
        now = time.monotonic()
        cutoff = now - self.window_seconds
        async with self._lock:
            window = self._hits[key]
            while window and window[0] <= cutoff:
                window.popleft()
            if len(window) < self.max_requests:
                window.append(now)
                return True
            return False

    async def _maybe_cleanup(self) -> None:
        """移除窗口已空的 key，防止 user_id 基数下内存缓慢泄漏。

        增量而非全量：限流判定在请求关键路径上，全表扫描会周期性
        卡住并发请求。顺序扫一段 key（list() 拷贝避免迭代中变更），
        下次触发从哨兵位置继续，最终覆盖全表。
        """
        self._ops += 1
        if self._ops % self._OPS_PER_SWEEP != 0:
            return
        now = time.monotonic()
        cutoff = now - self.window_seconds
        keys = list(self._hits.keys())
        start = (self._ops // self._OPS_PER_SWEEP * self._SWEEP_KEY_CAP) % max(1, len(keys)) if keys else 0
        removed = 0
        async with self._lock:
            for i in range(min(self._SWEEP_KEY_CAP, len(keys))):
                key = keys[(start + i) % len(keys)]
                window = self._hits.get(key)
                if window is None:
                    continue
                while window and window[0] <= cutoff:
                    window.popleft()
                if not window:
                    self._hits.pop(key, None)
                    removed += 1
        if removed:
            logger.debug(f"限流内存后端增量清理：移除 {removed} 个空 key")


class RedisSlidingWindowRateLimiter:
    """Redis ZSET 滑动窗口限流（跨 worker 精确计数）。

    用 Lua 脚本原子完成"清过期 + 计数 + 记录"三步，避免并发竞态。
    key 格式：alm_rl:{key}，ZSET 成员 = 时间戳(ms)，score = 同一时间戳。
    """

    # KEYS[1]=限流key  ARGV[1]=now(ms)  ARGV[2]=window(ms)  ARGV[3]=max_requests
    #  ARGV[4]=本次请求唯一 id —— ZSET member 必须唯一：member=now 时同一毫秒内
    #  的并发请求会互相覆盖，计数偏少导致限流被击穿
    _LUA = """
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local max = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - window)
local count = redis.call('ZCARD', KEYS[1])
if count < max then
  redis.call('ZADD', KEYS[1], now, ARGV[4])
  redis.call('PEXPIRE', KEYS[1], window)
  return 1
end
return 0
"""

    def __init__(self, redis_client, max_requests: int, window_seconds: float):
        self._redis = redis_client
        self.max_requests = max_requests
        self._window_ms = int(window_seconds * 1000)
        self._script = redis_client.register_script(self._LUA)

    async def allow(self, key: str) -> bool:
        now_ms = int(time.time() * 1000)
        try:
            result = await self._script(
                keys=[f"alm_rl:{key}"],
                args=[
                    str(now_ms), str(self._window_ms), str(self.max_requests),
                    f"{now_ms}-{uuid4().hex}",
                ],
            )
            return bool(result)
        except Exception as e:
            logger.warning(f"限流 Redis 异常，fail-open 放行: {e}")
            return True


_limiter: object | None = None


def get_rate_limiter():
    """模块级单例，按 settings.RATE_LIMIT_BACKEND 选择后端"""
    global _limiter
    if _limiter is None:
        if settings.RATE_LIMIT_BACKEND == "redis":
            import redis.asyncio as aioredis

            from src.infra.redis_cache import redis_pool

            client = aioredis.Redis(connection_pool=redis_pool)
            _limiter = RedisSlidingWindowRateLimiter(
                client,
                max_requests=settings.RATE_LIMIT_MAX_REQUESTS,
                window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
            )
        else:
            _limiter = SlidingWindowRateLimiter(
                max_requests=settings.RATE_LIMIT_MAX_REQUESTS,
                window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
            )
    return _limiter


async def check_rate_limit(
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> None:
    """FastAPI 依赖：按 user_id 维度限流，超限返回 429。

    用于重负载端点（ingest/upload、knowledge/search、search-stream）。
    """
    if not settings.RATE_LIMIT_ENABLED:
        return
    key = user.user_id or (request.client.host if request.client else "anon")
    if not await get_rate_limiter().allow(key):
        RATE_LIMIT_REJECTED.labels(endpoint=request.url.path).inc()
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后再试",
        )
