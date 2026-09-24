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


async def test_incremental_cleanup_removes_empty_keys(monkeypatch):
    """过期空 key 被增量清理移除（曾经的 _cleanup 是死代码，key 只增不减）。"""
    limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=0.01)
    for k in ("a", "b", "c"):
        await limiter.allow(k)
    await asyncio.sleep(0.02)  # 窗口滑过，a/b/c 全部过期
    monkeypatch.setattr(type(limiter), "_OPS_PER_SWEEP", 2)  # 加快触发
    await limiter.allow("fresh")  # 第 4 次判定 → 触发增量清理
    assert all(k not in limiter._hits for k in ("a", "b", "c"))
    assert "fresh" in limiter._hits
