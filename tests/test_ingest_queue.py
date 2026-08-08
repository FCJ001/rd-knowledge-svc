# ============================================================
# 入库任务队列（Redis Stream）单元测试
# 用 FakeRedis 校验：入队 XADD / 消费者组幂等创建 / 入队失败返回 False
# ============================================================

import json

from src.rag.ingestion import queue


class FakeRedis:
    def __init__(self):
        self.xadd_calls = []
        self.group_errors = []

    async def xgroup_create(self, *args, **kwargs):
        if self.group_errors:
            e = self.group_errors.pop(0)
            raise e
        return "OK"

    async def xadd(self, stream, fields, **kwargs):
        self.xadd_calls.append((stream, fields, kwargs))
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
