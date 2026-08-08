# ============================================================
# API 限流：滑动窗口计数（内存实现，仿 dataset_rag rate_limit_utils）
# ★ 单进程有效：uvicorn 多 worker 时各 worker 独立计数，不精确。
#   如需跨进程精确计数，可换 Redis 后端（REDIS_URL 已配置）。
# ============================================================

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request

from src.core.config import get_settings
from src.core.deps import UserContext, get_current_user

settings = get_settings()


class SlidingWindowRateLimiter:
    """按 key 的滑动窗口限流：窗口内请求数 < max_requests 放行，否则拒绝。"""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        """记录一次请求；窗口内未超限返回 True，否则返回 False。"""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        async with self._lock:
            window = self._hits[key]
            # 弹出窗口外的时间戳
            while window and window[0] <= cutoff:
                window.popleft()
            if len(window) < self.max_requests:
                window.append(now)
                return True
            return False


_limiter: SlidingWindowRateLimiter | None = None


def get_rate_limiter() -> SlidingWindowRateLimiter:
    """模块级单例，按 settings 初始化"""
    global _limiter
    if _limiter is None:
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
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后再试",
        )
