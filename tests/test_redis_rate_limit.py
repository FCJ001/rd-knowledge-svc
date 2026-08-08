# ============================================================
# RedisSlidingWindowRateLimiter 单元测试
# 用 FakeRedis 校验"脚本结果解释 + fail-open"逻辑，
# 窗口语义本身由 SlidingWindowRateLimiter（内存版）测试覆盖。
# ============================================================

import pytest

from src.core.cache import build_search_cache_key
from src.core.rate_limit import RedisSlidingWindowRateLimiter


class FakeRedis:
    """最小 fake：register_script 返回可 await 的脚本对象，结果可配置/抛错"""

    def __init__(self, results, raise_on_call=False):
        self._results = list(results)
        self._raise = raise_on_call
        self.calls = []

    def register_script(self, _script):
        return self

    async def __call__(self, keys, args):
        self.calls.append((keys, args))
        if self._raise:
            raise ConnectionError("redis down")
        return self._results.pop(0)


async def test_allow_when_script_returns_1():
    fake = FakeRedis([1])
    limiter = RedisSlidingWindowRateLimiter(fake, max_requests=10, window_seconds=60)
    assert await limiter.allow("u1") is True
    # key 前缀 + 参数传递正确
    keys, args = fake.calls[0]
    assert keys == ["alm_rl:u1"]
    assert args[2] == "10"


async def test_reject_when_script_returns_0():
    fake = FakeRedis([0])
    limiter = RedisSlidingWindowRateLimiter(fake, max_requests=2, window_seconds=60)
    assert await limiter.allow("u1") is False


async def test_fail_open_on_redis_error():
    fake = FakeRedis([], raise_on_call=True)
    limiter = RedisSlidingWindowRateLimiter(fake, max_requests=10, window_seconds=60)
    # Redis 异常 → 放行（fail-open），不抛
    assert await limiter.allow("u1") is True


def test_build_search_cache_key():
    k1 = build_search_cache_key("电池过热", ["doc_rag", "graph_rag"], "", "EV160", False, "engineer")
    k2 = build_search_cache_key("电池过热", ["doc_rag", "graph_rag"], "", "EV160", False, "engineer")
    k3 = build_search_cache_key("电池过热", ["doc_rag", "graph_rag"], "", "EV160", False, "admin")
    assert k1 == k2
    assert k1 != k3  # role 不同 → key 不同（NL2SQL 行过滤不同）
    assert k1.startswith("alm_cache:search:")


def test_build_search_cache_key_channel_order_insensitive():
    k1 = build_search_cache_key("q", ["doc_rag", "graph_rag"], "", "", False, "engineer")
    k2 = build_search_cache_key("q", ["graph_rag", "doc_rag"], "", "", False, "engineer")
    assert k1 == k2  # 通道顺序不影响 key
