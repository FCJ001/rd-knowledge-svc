# ============================================================
# 入库任务队列（Redis Stream）单元测试
# 用 FakeRedis 校验：入队 XADD / 消费者组幂等创建 / 入队失败返回 False /
#                   队列深度护栏（逼近 maxlen 拒收，防静默裁剪丢任务）
# ============================================================

import json

from src.rag.ingestion import queue


class FakeRedis:
    def __init__(self, depth: int = 0):
        self.xadd_calls = []
        self.group_errors = []
        self.depth = depth

    async def xgroup_create(self, *args, **kwargs):
        if self.group_errors:
            e = self.group_errors.pop(0)
            raise e
        return "OK"

    async def xlen(self, stream):
        return self.depth

    async def xadd(self, stream, fields, **kwargs):
        self.xadd_calls.append((stream, fields, kwargs))
        self.depth += 1
        return "123-0"


async def test_enqueue_success(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(queue, "_client", lambda: fake)

    ok = await queue.enqueue_ingest_job({"job_id": "1", "doc_name": "a.pdf"})
    assert ok is True
    stream, fields, kwargs = fake.xadd_calls[0]
    assert stream == queue.settings.INGEST_STREAM
    assert json.loads(fields["payload"])["doc_name"] == "a.pdf"
    assert kwargs["maxlen"] == queue.settings.INGEST_STREAM_MAX_LEN


async def test_ensure_group_idempotent_on_busy(monkeypatch):
    fake = FakeRedis()
    fake.group_errors = [Exception("BUSYGROUP Consumer Group name already exists")]
    monkeypatch.setattr(queue, "_client", lambda: fake)
    # 不抛异常即通过
    assert await queue.enqueue_ingest_job({"job_id": "2"}) is True


async def test_enqueue_failure_returns_false(monkeypatch):
    class DownRedis(FakeRedis):
        async def xadd(self, *a, **k):
            raise ConnectionError("redis down")

    monkeypatch.setattr(queue, "_client", lambda: DownRedis())
    assert await queue.enqueue_ingest_job({"job_id": "3"}) is False


async def test_enqueue_rejected_when_queue_full(monkeypatch):
    """深度 ≥ maxlen 时拒收：maxlen 裁剪不区分是否已投递，被裁的任务会
    在 DB 里永远停在 queued 且无任何报错 —— 宁可 503 让上游重试。"""
    fake = FakeRedis(depth=queue.settings.INGEST_STREAM_MAX_LEN)
    monkeypatch.setattr(queue, "_client", lambda: fake)

    ok = await queue.enqueue_ingest_job({"job_id": "4", "doc_name": "b.pdf"})
    assert ok is False
    assert fake.xadd_calls == []  # 拒收时绝不 XADD
