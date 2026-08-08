# ============================================================
# 韧性原语：超时 / 重试退避 / 熔断
#
# 用法：
#   from src.core.resilience import with_retry, CircuitBreaker
#
#   # 超时
#   result = await asyncio.wait_for(coro, timeout=10)   # 也可直接内联
#
#   # 重试（指数退避 + 抖动）
#   result = await with_retry(fn, attempts=3, task="channel:doc_rag")
#
#   # 熔断：目标持续失败 → open → 快速失败 → 半开探针 → 恢复
#   breaker = get_channel_breaker("graph_rag")
#   result = await breaker.call(fn)
#
# 语义：全部 fail-open —— 韧性层只管"更快地失败"，
# 最终是否降级由调用方（fusion.py 的 return_exceptions 语义）决定。
# ============================================================

from __future__ import annotations

import asyncio
import random
import time

from src.core.metrics import ASYNC_TASK_RETRIES, CIRCUIT_BREAKER_CHANGES


class CircuitOpenError(Exception):
    """熔断器处于 open 状态，目标被快速拒绝。"""

    def __init__(self, target: str):
        self.target = target
        super().__init__(f"熔断器已打开，目标 {target} 暂不可用")


async def with_retry(
    fn,
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    task: str = "generic",
) -> object:
    """指数退避 + 抖动重试。fn 为 async callable。

    - 失败后延迟 = min(max_delay, base_delay * 2**attempt) * uniform(0.5, 1.5)
    - attempts 次全失败后抛出最后一次异常
    """
    delay = base_delay
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except retry_on as e:
            last_exc = e
            if attempt == attempts - 1:
                break
            ASYNC_TASK_RETRIES.labels(task=task).inc()
            await asyncio.sleep(min(max_delay, delay) * random.uniform(0.5, 1.5))
            delay *= 2
    assert last_exc is not None
    raise last_exc


class CircuitBreaker:
    """按目标的熔断器（进程内单例），三态：closed / open / half_open。

    - closed：正常放行；累计 failure_threshold 次失败 → open
    - open：立即抛 CircuitOpenError（快速失败）；reset_timeout 过后进入 half_open
    - half_open：放一个探针；成功 → closed（清空计数）；失败 → 回到 open
    """

    def __init__(
        self,
        target: str,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
    ):
        self.target = target
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._state = "closed"
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()
        self._probe_in_flight = False

    # ── 状态查询 ───────────────────────────────────────────────

    def is_open(self) -> bool:
        if self._state == "open" and self._opened_at is not None:
            # open 但已过 reset 窗口 → 允许一个探针（返回 False 视为可放行）
            if time.monotonic() - self._opened_at >= self.reset_timeout:
                return False
        return self._state == "open"

    @property
    def state(self) -> str:
        if self._state == "open" and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self.reset_timeout:
                return "half_open"
        return self._state

    # ── 调用入口 ───────────────────────────────────────────────

    async def call(self, fn, *args, **kwargs):
        """执行目标调用，带熔断保护。fn 为 async callable。"""
        now = time.monotonic()
        async with self._lock:
            if self._state == "open":
                if now - self._opened_at >= self.reset_timeout:
                    self._set_state("half_open")
                else:
                    raise CircuitOpenError(self.target)

        try:
            result = await fn(*args, **kwargs)
        except Exception:
            async with self._lock:
                self._failures += 1
                if self._state == "half_open":
                    # 探针失败 → 立即回到 open
                    self._open()
                elif self._failures >= self.failure_threshold:
                    self._open()
            raise

        async with self._lock:
            if self._state == "half_open":
                # 探针成功 → 熔断恢复
                self._set_state("closed")
                self._failures = 0
            elif self._failures:
                # closed 下成功一次少记一次失败（缓慢恢复）
                self._failures = max(0, self._failures - 1)
        return result

    # ── 内部状态迁移 ───────────────────────────────────────────

    def _open(self) -> None:
        self._state = "open"
        self._opened_at = time.monotonic()
        self._record("open")

    def _set_state(self, state: str) -> None:
        if self._state != state:
            self._state = state
            self._record(state)

    def _record(self, state: str) -> None:
        CIRCUIT_BREAKER_CHANGES.labels(target=self.target, state=state).inc()


# 通道级熔断器单例（进程内共享，跨请求持久）
_channel_breakers: dict[str, CircuitBreaker] = {}


def get_channel_breaker(channel: str) -> CircuitBreaker:
    """获取某检索通道的熔断器。首次调用按 settings 初始化。"""
    from src.core.config import get_settings

    breaker = _channel_breakers.get(channel)
    if breaker is None:
        s = get_settings()
        breaker = CircuitBreaker(
            target=f"channel:{channel}",
            failure_threshold=s.CIRCUIT_FAILURE_THRESHOLD,
            reset_timeout=s.CIRCUIT_RESET_TIMEOUT,
        )
        _channel_breakers[channel] = breaker
    return breaker
