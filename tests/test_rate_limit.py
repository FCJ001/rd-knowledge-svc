# ============================================================
# SlidingWindowRateLimiter 滑动窗口限流单元测试
# 覆盖：窗口内放行 / 超限拒绝 / 窗口滑动后恢复 / 不同 key 隔离
# ============================================================

import asyncio

from src.core.rate_limit import SlidingWindowRateLimiter


async def test_within_window_allowed():
    limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        assert await limiter.allow("u1") is True


async def test_exceed_limit_rejected():
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    assert await limiter.allow("u1") is True
    assert await limiter.allow("u1") is True
    assert await limiter.allow("u1") is False


async def test_window_slides_and_recovers():
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=0.05)
    assert await limiter.allow("u1") is True
    assert await limiter.allow("u1") is False
    await asyncio.sleep(0.06)  # 窗口滑过
    assert await limiter.allow("u1") is True


async def test_keys_isolated():
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
    assert await limiter.allow("u1") is True
    assert await limiter.allow("u2") is True  # 不同 key 互不影响
    assert await limiter.allow("u1") is False
