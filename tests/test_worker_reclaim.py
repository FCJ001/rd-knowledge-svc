# ============================================================
# worker 崩溃回收单元测试
#
# 覆盖 _claim_stale 的认领判据：
#   空闲超阈值 且 原持有人心跳已失效 才认领；原持有人还活着绝不抢。
# 用 FakeRedis 校验调用序列，不打真 Redis（CI 约定：纯函数单测不进外部服务）。
# ============================================================

import pytest
from src.rag.ingestion import worker as w


class FakeRedis:
    """最小可用子集：xpending_range / exists / xclaim。"""

    def __init__(self, pending=None, alive=()):
        self.pending = pending or []      # xpending_range 返回值
        self.alive = set(alive)           # 心跳仍有效的 consumer
        self.xclaim_calls = []

    async def xpending_range(self, stream, group, *, min, max, count, idle):
        # 只返回 idle 达标的条目（与真 Redis 的 idle 过滤语义一致）
        return [e for e in self.pending if e["time_since_delivered"] >= idle]

    async def exists(self, key):
        return 1 if key in self.alive else 0

    async def xclaim(self, stream, group, consumer, *, min_idle_time, message_ids):
        self.xclaim_calls.append((consumer, tuple(message_ids)))
        # 返回 (id, fields) 元组列表，与 redis-py 一致
        return [(mid, {"payload": "{}"}) for mid in message_ids]


async def test_claim_skips_alive_owner():
    """原持有人心跳仍在（可能正在处理大文档）→ 绝不认领。"""
    fake = FakeRedis(
        pending=[{
            "message_id": "1-1",
            "consumer": "host-a",
            "time_since_delivered": w.PEL_MIN_IDLE_MS + 1,
        }],
        alive={w._alive_key("host-a")},
    )
    claimed = await w._claim_stale(fake, "host-b")
    assert claimed == []
    assert fake.xclaim_calls == []


async def test_claim_takes_dead_owner_message():
    """空闲超阈值 且 心跳失效 → 认领并返回消息。"""
    fake = FakeRedis(
        pending=[
            {"message_id": "1-1", "consumer": "host-a",
             "time_since_delivered": w.PEL_MIN_IDLE_MS + 1},
            {"message_id": "1-2", "consumer": "host-a",
             "time_since_delivered": w.PEL_MIN_IDLE_MS - 1},  # 空闲不够，不该出现
        ],
    )
    claimed = await w._claim_stale(fake, "host-b")
    assert [mid for mid, _ in claimed] == ["1-1"]
    # XCLAIM 带 min_idle 二次校验
    assert fake.xclaim_calls == [("host-b", ("1-1",))]


async def test_claim_survives_redis_error():
    """回收路径的异常不许外抛：下个巡检周期重试即可。"""
    class DownRedis(FakeRedis):
        async def xpending_range(self, *a, **k):
            raise ConnectionError("redis down")

    assert await w._claim_stale(DownRedis(), "host-b") == []


@pytest.mark.parametrize("count", [0])
async def test_claim_empty_pending(count):
    assert await w._claim_stale(FakeRedis(pending=[]), "host-b") == []
