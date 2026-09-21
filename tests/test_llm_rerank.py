# ============================================================
# LLM 清单式重排（RERANK_PROVIDER=deepseek）单元测试
# 不出网：monkeypatch get_llm / settings
# ============================================================

import asyncio

from src.knowledge import reranker as rr
from src.knowledge.reranker import _parse_llm_rerank, rerank_docs


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, content: str = "", error: Exception | None = None):
        self._content = content
        self._error = error

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, prompt):
        if self._error:
            raise self._error
        return _FakeMessage(self._content)


def _docs(n: int) -> list[dict]:
    return [{"text": f"doc{i}", "doc_id": str(i)} for i in range(n)]


def test_parse_llm_rerank_valid_and_clamped():
    content = (
        '{"results": [{"index": 2, "score": 1.7}, '
        '{"index": 0, "score": -0.5}, {"index": 1, "score": 0.8}]}'
    )
    parsed = _parse_llm_rerank(content, 3)
    assert parsed == [(2, 1.0), (0, 0.0), (1, 0.8)]


def test_parse_llm_rerank_drops_invalid_entries():
    content = (
        '{"results": [{"index": 9, "score": 0.9}, {"index": 1, "score": 0.8}, '
        '{"index": 1, "score": 0.1}, {"index": "x", "score": 0.5}, {"noindex": true}]}'
    )
    parsed = _parse_llm_rerank(content, 2)
    assert parsed == [(1, 0.8)]


def test_rerank_docs_provider_off_passthrough(monkeypatch):
    monkeypatch.setattr(rr.settings, "RERANK_PROVIDER", "off")
    docs = _docs(4)
    out = asyncio.run(rerank_docs("q", docs, top_k=2))
    assert [d["doc_id"] for d in out] == ["0", "1"]
    # 不修改原列表
    assert all("rerank_score" not in d for d in docs)


def test_rerank_docs_llm_orders_and_scores(monkeypatch):
    monkeypatch.setattr(rr.settings, "RERANK_PROVIDER", "deepseek")
    monkeypatch.setattr(rr.settings, "RAG_DYNAMIC_TOPK", False)
    content = (
        '{"results": [{"index": 2, "score": 0.95}, '
        '{"index": 0, "score": 0.60}, {"index": 1, "score": 0.30}]}'
    )
    monkeypatch.setattr(rr, "get_llm", lambda: _FakeLLM(content))
    out = asyncio.run(rerank_docs("q", _docs(3), top_k=2))
    assert [d["doc_id"] for d in out] == ["2", "0"]
    assert out[0]["rerank_score"] == 0.95


def test_rerank_docs_llm_missing_docs_appended(monkeypatch):
    monkeypatch.setattr(rr.settings, "RERANK_PROVIDER", "deepseek")
    monkeypatch.setattr(rr.settings, "RAG_DYNAMIC_TOPK", False)
    content = '{"results": [{"index": 2, "score": 0.9}]}'
    monkeypatch.setattr(rr, "get_llm", lambda: _FakeLLM(content))
    out = asyncio.run(rerank_docs("q", _docs(4), top_k=3))
    # LLM 只回了 index 2，漏掉的 0/1/3 按原相对顺序排在末尾
    assert [d["doc_id"] for d in out] == ["2", "0", "1"]
    assert out[-1]["rerank_score"] == 0.0


def test_rerank_docs_llm_failure_falls_back(monkeypatch):
    monkeypatch.setattr(rr.settings, "RERANK_PROVIDER", "deepseek")
    monkeypatch.setattr(rr, "get_llm", lambda: _FakeLLM(error=RuntimeError("boom")))
    out = asyncio.run(rerank_docs("q", _docs(4), top_k=2))
    assert [d["doc_id"] for d in out] == ["0", "1"]


def test_rerank_docs_llm_unparseable_falls_back(monkeypatch):
    monkeypatch.setattr(rr.settings, "RERANK_PROVIDER", "deepseek")
    monkeypatch.setattr(rr, "get_llm", lambda: _FakeLLM("不是 JSON"))
    out = asyncio.run(rerank_docs("q", _docs(4), top_k=2))
    assert [d["doc_id"] for d in out] == ["0", "1"]


def test_rerank_docs_short_circuits(monkeypatch):
    monkeypatch.setattr(rr.settings, "RERANK_PROVIDER", "deepseek")
    # 空列表 / 少于 top_k 直接透传，不出网
    assert asyncio.run(rerank_docs("q", [], top_k=3)) == []
    docs = _docs(2)
    assert asyncio.run(rerank_docs("q", docs, top_k=3)) == docs
