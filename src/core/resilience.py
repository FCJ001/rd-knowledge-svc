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
#
# 熔断状态存储：
#   - use_redis=True（生产 get_channel_breaker 默认）：state/failures/opened_at
#     存 Redis Hash（cb:{target}），多副本共享同一份熔断状态 —— 任一副本观察到
#     目标持续失败，全集群一起快速失败；半开探针用 SETNX 锁（cb:{target}:probe）
#     协调，复位窗口内只放一个副本探测，避免 N 个副本同时打探针压垮目标。
#   - use_redis=False 或 Redis 读写异常：降级进程内状态（单副本语义），
#     韧性层自身永不成为新的故障点。
#   - Redis 侧时钟统一用 wall clock（跨进程可比），进程内用单调时钟。
# ============================================================

from __future__ import annotations

import asyncio
import random
import time

from loguru import logger

from src.core.metrics import ASYNC_TASK_RETRIES, CIRCUIT_BREAKER_CHANGES

# Redis 单次操作超时：熔断检查在请求关键路径上，不能被慢 Redis 拖住
_REDIS_OP_TIMEOUT = 0.5


class CircuitOpenError(Exception):
    """熔断器处于 open 状态，目标被快速拒绝。"""

    def __init__(self, target: str):
        self.target = target
        super().__init__(f"熔断器已打开，目标 {target} 暂不可用")

# ── Lua：原子状态迁移（KEYS[1]=状态 Hash，KEYS[2]=探针锁）───────────────

# 返回 1=放行 0=拒绝 2=放行且本调用是半开探针
_LUA_ACQUIRE = """
local st = redis.call('HGET', KEYS[1], 'state')
if not st then st = 'closed' end
local now = tonumber(ARGV[1])
local reset = tonumber(ARGV[2])
if st == 'open' then
  local opened = tonumber(redis.call('HGET', KEYS[1], 'opened_at') or '0')
  if now - opened < reset then
    return 0
  end
  -- 复位窗口已过：SETNX 抢探针锁，只放一个副本探测
  local ttl = math.max(1, math.ceil(reset))
  if redis.call('SET', KEYS[2], '1', 'EX', ttl, 'NX') then
    redis.call('HSET', KEYS[1], 'state', 'half_open')
    return 2
  end
  return 0
end
return 1
"""

# 失败上报：half_open 失败 → 立即回 open；closed 下累计到阈值 → open。
# 返回迁移后的状态
_LUA_REPORT_FAILURE = """
local st = redis.call('HGET', KEYS[1], 'state')
if not st then st = 'closed' end
local failures = redis.call('HINCRBY', KEYS[1], 'failures', 1)
local now = tonumber(ARGV[1])
local threshold = tonumber(ARGV[2])
if st == 'half_open' or failures >= threshold then
  redis.call('HSET', KEYS[1], 'state', 'open', 'opened_at', now)
  redis.call('DEL', KEYS[2])
  return 'open'
end
return 'closed'
"""

# 成功上报：half_open 成功 → closed 清零；closed 下缓慢恢复（失败计数 -1）
_LUA_REPORT_SUCCESS = """
local st = redis.call('HGET', KEYS[1], 'state')
if not st then st = 'closed' end
if st == 'half_open' then
  redis.call('HSET', KEYS[1], 'state', 'closed', 'failures', 0)
  redis.call('DEL', KEYS[2])
  return 'closed'
end
local failures = tonumber(redis.call('HGET', KEYS[1], 'failures') or '0')
if failures > 0 then
  redis.call('HSET', KEYS[1], 'failures', failures - 1)
end
return st
"""


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
    """按目标的熔断器，三态：closed / open / half_open。

    - closed：正常放行；累计 failure_threshold 次失败 → open
    - open：立即抛 CircuitOpenError（快速失败）；reset_timeout 过后进入 half_open
    - half_open：放一个探针（Redis 模式下跨副本只放一个）；成功 → closed，
      失败 → 回到 open

    use_redis=False（默认，单测/单副本）：状态仅存进程内。
    use_redis=True：状态外置 Redis，多副本共享；Redis 异常自动降级进程内。
    同步属性 state / is_open 读的是本地镜像（最近的已知状态），仅供观测，
    拒绝判定以 call() 里的原子检查为准。
    """

    def __init__(
        self,
        target: str,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        use_redis: bool = False,
    ):
        self.target = target
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.use_redis = use_redis
        # 进程内镜像 / 降级状态
        self._state = "closed"
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()

    # ── Redis key ──────────────────────────────────────────────

    @property
    def _hash_key(self) -> str:
        return f"cb:{self.target}"

    @property
    def _probe_key(self) -> str:
        return f"cb:{self.target}:probe"

    def _redis_client(self):
        from src.infra.redis_cache import get_redis_sync_client

        return get_redis_sync_client()

    async def _eval(self, script: str, *args):
        """带超时的 Lua 执行；返回 None 表示 Redis 不可用（调用方走降级）。"""
        try:
            client = self._redis_client()
            return await asyncio.wait_for(
                client.eval(script, 2, self._hash_key, self._probe_key, *args),
                timeout=_REDIS_OP_TIMEOUT,
            )
        except Exception as e:
            logger.warning(f"[CIRCUIT] Redis 不可用，熔断器 {self.target} 降级进程内: {e}")
            return None

    # ── 状态查询（本地镜像，仅供观测）──────────────────────────
    # 镜像统一用 wall clock：Redis 侧 opened_at 是跨进程的 wall clock，
    # 降级路径也沿用同一时钟，避免混用导致 open/half_open 判断错乱

    def is_open(self) -> bool:
        if self._state == "open" and self._opened_at is not None:
            # open 但已过 reset 窗口 → 允许一个探针（返回 False 视为可放行）
            if time.time() - self._opened_at >= self.reset_timeout:
                return False
        return self._state == "open"

    @property
    def state(self) -> str:
        if self._state == "open" and self._opened_at is not None:
            if time.time() - self._opened_at >= self.reset_timeout:
                return "half_open"
        return self._state

    # ── 调用入口 ───────────────────────────────────────────────

    async def call(self, fn, *args, **kwargs):
        """执行目标调用，带熔断保护。fn 为 async callable。"""
        async with self._lock:
            allowed = await self._acquire()
        if not allowed:
            raise CircuitOpenError(self.target)

        try:
            result = await fn(*args, **kwargs)
        except Exception:
            async with self._lock:
                await self._report_failure()
            raise

        async with self._lock:
            await self._report_success()
        return result

    # ── 状态迁移（Redis 优先，异常降级进程内）──────────────────

    async def _acquire(self) -> bool:
        """准入检查。True=放行（含半开探针），False=熔断期拒绝。"""
        if self.use_redis:
            ret = await self._eval(
                _LUA_ACQUIRE, repr(time.time()), repr(self.reset_timeout),
            )
            if ret is not None:
                if ret == 2:
                    self._set_local("half_open")
                    self._record("half_open")
                elif ret == 0:
                    # 拒绝时镜像保持 open（若已过窗口，镜像标 half_open 更贴近）
                    self._set_local("half_open" if self.state == "half_open" else "open")
                else:
                    self._set_local("closed")
                return ret in (1, 2)

        # 降级：进程内判定（与原单副本语义一致）
        now = time.time()
        if self._state == "open":
            if now - (self._opened_at or now) >= self.reset_timeout:
                self._set_local("half_open")
                self._record("half_open")
            else:
                return False
        return True

    async def _report_failure(self) -> None:
        if self.use_redis:
            ret = await self._eval(
                _LUA_REPORT_FAILURE, repr(time.time()), repr(self.failure_threshold),
            )
            if ret is not None:
                if ret == "open":
                    self._open_local()
                return

        # 降级：进程内计数
        self._failures += 1
        if self._state == "half_open":
            self._open_local()
        elif self._failures >= self.failure_threshold:
            self._open_local()

    async def _report_success(self) -> None:
        if self.use_redis:
            ret = await self._eval(_LUA_REPORT_SUCCESS)
            if ret is not None:
                if ret == "closed" and self._state == "half_open":
                    self._set_local("closed")
                    self._failures = 0
                    self._record("closed")
                elif self._failures:
                    self._failures = max(0, self._failures - 1)
                return

        if self._state == "half_open":
            self._set_local("closed")
            self._failures = 0
            self._record("closed")
        elif self._failures:
            self._failures = max(0, self._failures - 1)

    # ── 本地镜像维护 ───────────────────────────────────────────

    def _set_local(self, state: str) -> None:
        self._state = state

    def _open_local(self) -> None:
        self._state = "open"
        self._opened_at = time.time()
        self._record("open")

    def _record(self, state: str) -> None:
        CIRCUIT_BREAKER_CHANGES.labels(target=self.target, state=state).inc()


# 通道级熔断器单例（进程内共享，跨请求持久；状态本身按 use_redis 外置）
_channel_breakers: dict[str, CircuitBreaker] = {}


def get_channel_breaker(channel: str) -> CircuitBreaker:
    """获取某检索通道的熔断器。首次调用按 settings 初始化。

    CIRCUIT_REDIS_ENABLED=True（默认）时状态外置 Redis，多副本部署下
    所有进程共享同一份熔断状态与半开探针锁。"""
    from src.core.config import get_settings

    breaker = _channel_breakers.get(channel)
    if breaker is None:
        s = get_settings()
        breaker = CircuitBreaker(
            target=f"channel:{channel}",
            failure_threshold=s.CIRCUIT_FAILURE_THRESHOLD,
            reset_timeout=s.CIRCUIT_RESET_TIMEOUT,
            use_redis=s.CIRCUIT_REDIS_ENABLED,
        )
        _channel_breakers[channel] = breaker
    return breaker
