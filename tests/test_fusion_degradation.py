# ============================================================
# 检索失败语义：必须区分「依赖故障」与「检索成功但无内容」
#
# 若不区分，Milvus/embedding 挂掉会被伪装成"知识库里没有这个信息"——
# 用户据此做出错误判断，监控侧也因为没有 5xx 而完全静默。
# ============================================================

import pytest
from src.knowledge import fusion as fusion_mod
from src.knowledge.fusion import RetrievalUnavailableError, multi_channel_search


class _FakeLLM:
    """最小 LLM 桩：生成阶段直接返回固定答案。

    幻觉检测用它做裁判会解析失败，而 check_hallucination 是 fail-open，
    因此不需要额外的桩。
    """

    def __init__(self, content: str = "[答案]"):
        self._content = content

    def with_config(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        class _Msg:
            content = self._content
        return _Msg()


async def _noop_rewrite(question, llm, role, event_sink):
    return {"queries": [question], "intent": "knowledge_qa"}


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(fusion_mod, "_rewrite_question", _noop_rewrite)
    monkeypatch.setattr(fusion_mod, "check_hallucination", _grounded)


async def _grounded(question, evidence, answer, llm, threshold=0.7):
    return {"is_grounded": True, "unsupported_claims": [], "confidence": 1.0}


async def test_all_channels_failed_raises_unavailable(patched, monkeypatch):
    """全部通道失败 → 抛 RetrievalUnavailableError（调用方据此应答 503）。"""

    async def _boom(*args, **kwargs):
        raise RuntimeError("milvus 连接失败")

    monkeypatch.setattr(fusion_mod, "search_docs_raw", _boom)

    with pytest.raises(RetrievalUnavailableError) as exc:
        await multi_channel_search(
            question="刹车异响", llm=_FakeLLM(), embedding_model=None,
            milvus_client=None, neo4j_driver=None, channels=["doc_rag"],
        )
    assert exc.value.channels == ["doc_rag"]


async def test_all_failed_emits_degraded_event(patched, monkeypatch):
    """故障必须推 degraded 事件，前端才能提示"服务不可用"而非"未找到"。"""
    events: list[dict] = []

    async def _boom(*args, **kwargs):
        raise RuntimeError("embedding 上游超时")

    monkeypatch.setattr(fusion_mod, "search_docs_raw", _boom)

    with pytest.raises(RetrievalUnavailableError):
        await multi_channel_search(
            question="q", llm=_FakeLLM(), embedding_model=None,
            milvus_client=None, neo4j_driver=None, channels=["doc_rag"],
            event_sink=events.append,
        )
    assert any(e.get("type") == "degraded" for e in events)


async def test_empty_result_is_not_treated_as_failure(patched, monkeypatch):
    """检索成功但没有内容 → status=empty（知识库覆盖问题），不得抛异常。"""

    async def _empty(*args, **kwargs):
        return []

    monkeypatch.setattr(fusion_mod, "search_docs_raw", _empty)

    result = await multi_channel_search(
        question="q", llm=_FakeLLM(), embedding_model=None,
        milvus_client=None, neo4j_driver=None, channels=["doc_rag"],
    )
    assert result["status"] == "empty"
    assert result["failed_channels"] == []
    assert "未找到" in result["answer"]


async def test_partial_failure_is_flagged_but_answers(patched, monkeypatch):
    """部分通道失败 → 正常作答 + status=partial + 列出失败通道。"""

    async def _hits(*args, **kwargs):
        return [{"text": "刹车片磨损", "doc_name": "手册.pdf", "page_number": 12,
                 "chunk_index": 0, "score": 0.9}]

    async def _boom(*args, **kwargs):
        raise RuntimeError("neo4j 不可用")

    monkeypatch.setattr(fusion_mod, "search_docs_raw", _hits)
    monkeypatch.setattr(fusion_mod, "search_graph_raw", _boom)

    result = await multi_channel_search(
        question="q", llm=_FakeLLM(), embedding_model=None,
        milvus_client=None, neo4j_driver=None,
        channels=["doc_rag", "graph_rag"],
    )
    assert result["status"] == "partial"
    assert result["failed_channels"] == ["graph_rag"]
    assert result["answer"]


async def test_doc_filters_reach_retrieval_layer(patched, monkeypatch):
    """model_code / doc_type 必须一路传到检索层——过滤条件丢失=跨车型串味。"""
    seen: list[dict] = []

    async def _capture(q, embedding_model, milvus_client, **kwargs):
        seen.append(kwargs)
        return []

    monkeypatch.setattr(fusion_mod, "search_docs_raw", _capture)

    await multi_channel_search(
        question="q", llm=_FakeLLM(), embedding_model=None,
        milvus_client=None, neo4j_driver=None, channels=["doc_rag"],
        doc_type="repair_manual", model_code="EV160",
    )
    assert seen, "检索层没有被调用"
    assert seen[0]["doc_type"] == "repair_manual"
    assert seen[0]["model_code"] == "EV160"


async def test_empty_filter_becomes_none(patched, monkeypatch):
    """空过滤条件传 None 而不是空字符串（空串会被拼成 doc_type == "" 而查空）。"""
    seen: list[dict] = []

    async def _capture(q, embedding_model, milvus_client, **kwargs):
        seen.append(kwargs)
        return []

    monkeypatch.setattr(fusion_mod, "search_docs_raw", _capture)

    await multi_channel_search(
        question="q", llm=_FakeLLM(), embedding_model=None,
        milvus_client=None, neo4j_driver=None, channels=["doc_rag"],
    )
    assert seen[0]["doc_type"] is None
    assert seen[0]["model_code"] is None


def test_default_channels_exclude_dead_graph():
    """默认通道不得包含 graph_rag。

    当前 Neo4j 为空图（0 节点 0 关系），graph_rag 在关键路径上要花 2 次
    LLM 调用（抽实体 + 生成 Cypher）才能返回 0 条，而 fusion 会等所有通道
    settle 才开始生成——它直接推迟答案。
    """
    import inspect

    src = inspect.getsource(fusion_mod.multi_channel_search)
    assert 'channels = ["doc_rag"]' in src
