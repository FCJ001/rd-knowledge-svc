# ============================================================
# 韧性原语单元测试：with_retry / CircuitBreaker
# 覆盖：重试成功 / 重试耗尽 / 熔断打开 / open 快速拒绝 /
#       半开探针成功恢复 / 探针失败回到 open
# ============================================================

import asyncio

import pytest

from src.core.resilience import CircuitBreaker, CircuitOpenError, with_retry


# ── with_retry ───────────────────────────────────────────────

async def test_retry_succeeds_on_second_attempt():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("boom")
        return "ok"

    result = await with_retry(flaky, attempts=3, base_delay=0)
    assert result == "ok"
    assert calls["n"] == 2


async def test_retry_exhausted_raises():
    async def always_fail():
        raise ValueError("x")

    with pytest.raises(ValueError):
        await with_retry(always_fail, attempts=3, base_delay=0)


async def test_retry_only_on_specified_exceptions():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("transient")
        return "ok"

    # 只重试 ValueError → 第三次成功
    result = await with_retry(flaky, attempts=3, base_delay=0, retry_on=(ValueError,))
    assert result == "ok"


async def test_retry_does_not_catch_unlisted_exception():
    async def boom():
        raise KeyError("nope")

    with pytest.raises(KeyError):
        await with_retry(boom, attempts=3, base_delay=0, retry_on=(ValueError,))


# ── CircuitBreaker ───────────────────────────────────────────

async def test_breaker_opens_after_threshold():
    breaker = CircuitBreaker(target="test", failure_threshold=3, reset_timeout=60)

    async def fail():
        raise RuntimeError("down")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call(fail)
    assert breaker.state == "open"


async def test_breaker_rejects_while_open():
    breaker = CircuitBreaker(target="test", failure_threshold=1, reset_timeout=60)

    async def fail():
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        await breaker.call(fail)
    assert breaker.state == "open"

    # open 期间立即拒绝，不触发目标调用
    with pytest.raises(CircuitOpenError):
        await breaker.call(fail)


async def test_breaker_half_open_probe_success_recovers():
    breaker = CircuitBreaker(target="test", failure_threshold=1, reset_timeout=0.05)

    async def fail():
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        await breaker.call(fail)
    assert breaker.state == "open"

    await asyncio.sleep(0.06)  # 过了复位窗口

    async def ok():
        return "recovered"

    result = await breaker.call(ok)
    assert result == "recovered"
    assert breaker.state == "closed"


async def test_breaker_probe_failure_reopens():
    breaker = CircuitBreaker(target="test", failure_threshold=1, reset_timeout=0.05)

    async def fail():
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        await breaker.call(fail)
    assert breaker.state == "open"

    await asyncio.sleep(0.06)

    # 探针失败 → 回到 open
    with pytest.raises(RuntimeError):
        await breaker.call(fail)
    assert breaker.state == "open"


async def test_breaker_success_resets_failures():
    breaker = CircuitBreaker(target="test", failure_threshold=2, reset_timeout=60)

    async def flaky():
        raise RuntimeError("down")

    # 1 次失败（未到阈值）
    with pytest.raises(RuntimeError):
        await breaker.call(flaky)
    assert breaker.state == "closed"

    async def ok():
        return "fine"

    assert await breaker.call(ok) == "fine"
    assert breaker.state == "closed"

    # 失败计数被成功重置 → 再来 2 次才打开
    with pytest.raises(RuntimeError):
        await breaker.call(flaky)
    assert breaker.state == "closed"
    with pytest.raises(RuntimeError):
        await breaker.call(flaky)
    assert breaker.state == "open"
