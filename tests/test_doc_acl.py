# ============================================================
# 文档级 ACL：权限谓词必须进检索请求（检索前过滤），不做"检索后再丢弃"
#
# 为什么盯住"表达式进没进检索请求"：
#   1) 先检索 top_k 再按权限丢弃，top_k 会被无权内容吃掉——表现为"库里明明有
#      却查不到"，且结果侧无法与"确实没有"区分；
#   2) 结果侧过滤点不止一处（上下文/图片/引用/缓存），漏一处就是泄露。
#   所以正确形态只有一个：ACL 谓词出现在 Milvus 查询的布尔表达式里，
#   无权内容从不进入候选集。下面每个测试都在盯这个不变量。
# ============================================================

import inspect

import pytest
from src.core.config import Settings, get_settings
from src.knowledge import fusion as fusion_mod
from src.knowledge.acl import (
    AclConfigError,
    doc_acl_expr,
    effective_acl_roles,
    graph_read_allowed,
    parse_acl_roles,
)
from src.knowledge.doc_rag import search_docs_raw, search_docs_with_stages
from src.knowledge.fusion import multi_channel_search
from src.rag.retrieval.hybrid_search import hybrid_search

# ── 角色解析 ─────────────────────────────────────────────────────────────

def test_parse_acl_roles_dedupes_and_strips_admin():
    assert parse_acl_roles("engineer, business,engineer") == ["business", "engineer"]
    assert parse_acl_roles("admin,engineer") == ["engineer"]  # admin 恒可见，不进列表


def test_parse_acl_roles_rejects_unknown_role():
    """拼错的角色名必须报错而不是丢弃：静默丢弃 = 这份文档谁（除 admin）都看不见，
    只能靠用户投诉发现。"""
    with pytest.raises(AclConfigError, match="enginer"):
        parse_acl_roles("enginer")


def test_effective_acl_roles_none_falls_back_to_config(monkeypatch):
    monkeypatch.setattr(get_settings(), "DOC_ACL_DEFAULT_ROLES", "engineer")
    assert effective_acl_roles(None) == ["engineer"]


def test_effective_acl_roles_empty_means_admin_only(monkeypatch):
    """显式空串 = 上传者声明"仅 admin"（与"没传"语义不同，不得被配置覆盖）。"""
    monkeypatch.setattr(get_settings(), "DOC_ACL_DEFAULT_ROLES", "engineer")
    assert effective_acl_roles("") == []


# ── 检索谓词 ─────────────────────────────────────────────────────────────

def test_doc_acl_expr_scopes_to_role():
    assert doc_acl_expr("engineer") == 'array_contains_any(acl_roles, ["engineer"])'


def test_doc_acl_expr_admin_bypasses():
    assert doc_acl_expr("admin") is None


def test_doc_acl_expr_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(get_settings(), "DOC_ACL_ENABLED", False)
    assert doc_acl_expr("engineer") is None


def test_doc_acl_expr_unknown_role_fail_closed():
    """词表外角色（token 被篡改/上游新增未同步）：谓词仍在、匹配不到任何文档。
    不得静默放行（那是提权），也不得抛错打挂整条链路。"""
    expr = doc_acl_expr("wizard")
    assert expr is not None
    assert '"wizard"' in expr


def test_hybrid_search_requires_acl_expr_argument():
    """acl_expr 不得有默认值：None 的语义是"已确认无需 ACL"（admin/开关关闭），
    有默认值就会有人"忘了传"——忘了传等于无权限检索。"""
    param = inspect.signature(hybrid_search).parameters["acl_expr"]
    assert param.default is inspect.Parameter.empty


async def test_hybrid_search_ands_acl_into_every_request():
    """ACL 谓词必须并进每个 AnnSearchRequest（dense/BM25/extra 各路同一表达式），
    且与业务过滤逐项加括号——少一层括号，谓词可能被 and 的另一项吃掉。"""

    class _FakeMilvus:
        reqs = None

        def hybrid_search(self, collection_name, reqs, **kwargs):
            _FakeMilvus.reqs = reqs
            return [[]]

    fake = _FakeMilvus()
    await hybrid_search(
        fake, "alm_docs", dense_embedding=[0.1], query_text="q",
        acl_expr='array_contains_any(acl_roles, ["engineer"])',
        top_k=5, filters={"doc_type": "spec_doc"},
    )
    assert fake.reqs, "检索请求未发出"
    for req in fake.reqs:
        assert req.expr == (
            '(doc_type == "spec_doc") and (array_contains_any(acl_roles, ["engineer"]))'
        )


async def test_hybrid_search_admin_expr_none_keeps_filter_only():
    class _FakeMilvus:
        reqs = None

        def hybrid_search(self, collection_name, reqs, **kwargs):
            _FakeMilvus.reqs = reqs
            return [[]]

    fake = _FakeMilvus()
    await hybrid_search(
        fake, "alm_docs", dense_embedding=[0.1], query_text="q",
        acl_expr=None, top_k=5,
    )
    assert all(req.expr == "" for req in fake.reqs)


async def test_dense_fallback_keeps_acl(monkeypatch):
    """hybrid 挂掉降级为 dense-only 时同样必须带 ACL——降级路径漏权限 = 一次
    上游抖动就变成全量检索（提权）。"""

    async def _boom(*args, **kwargs):
        raise RuntimeError("RRF 融合失败")

    monkeypatch.setattr("src.rag.retrieval.hybrid_search.hybrid_search", _boom)

    captured = {}

    class _FakeClient:
        def search(self, **kwargs):
            captured.update(kwargs)
            return [[]]

    class _FakeEmbed:
        async def aembed_query(self, q):
            return [0.1]

    hits = await search_docs_raw(
        "刹车异响", _FakeEmbed(), _FakeClient(), role="engineer",
        doc_type="spec_doc",
    )
    assert hits == []
    assert captured["filter"] == (
        '(doc_type == "spec_doc") and (array_contains_any(acl_roles, ["engineer"]))'
    )


async def test_dense_fallback_admin_no_acl_clause(monkeypatch):
    async def _boom(*args, **kwargs):
        raise RuntimeError("RRF 融合失败")

    monkeypatch.setattr("src.rag.retrieval.hybrid_search.hybrid_search", _boom)

    captured = {}

    class _FakeClient:
        def search(self, **kwargs):
            captured.update(kwargs)
            return [[]]

    class _FakeEmbed:
        async def aembed_query(self, q):
            return [0.1]

    await search_docs_raw("q", _FakeEmbed(), _FakeClient(), role="admin")
    assert captured["filter"] is None


def test_search_docs_role_has_no_default():
    """role 无默认值：安全参数不允许隐含取值（忘了传必须在调用点报错，
    而不是按某个角色口径静默检索）。"""
    for fn in (search_docs_raw, search_docs_with_stages):
        assert inspect.signature(fn).parameters["role"].default is inspect.Parameter.empty


# ── 图谱通道门禁 ─────────────────────────────────────────────────────────

def test_graph_read_allowed_denies_non_admin(monkeypatch):
    monkeypatch.setattr(get_settings(), "GRAPH_ACL_ALLOW_NON_ADMIN", False)
    assert graph_read_allowed("engineer") is False
    assert graph_read_allowed("admin") is True


def test_graph_read_allowed_explicit_risk_optin(monkeypatch):
    monkeypatch.setattr(get_settings(), "GRAPH_ACL_ALLOW_NON_ADMIN", True)
    assert graph_read_allowed("engineer") is True


# ── fusion 通道编排 ──────────────────────────────────────────────────────

async def _noop_rewrite(question, llm, role, event_sink):
    return {"queries": [question], "intent": "knowledge_qa"}


async def _grounded(question, evidence, answer, llm, threshold=0.7):
    return {"is_grounded": True, "unsupported_claims": [], "confidence": 1.0}


class _FakeLLM:
    def with_config(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        class _Msg:
            content = "[答案]"
        return _Msg()

    async def astream(self, messages):
        class _Chunk:
            content = "[答案]"
        yield _Chunk()


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(fusion_mod, "_rewrite_question", _noop_rewrite)
    monkeypatch.setattr(fusion_mod, "check_hallucination", _grounded)


async def test_graph_channel_denied_before_retrieval_for_non_admin(patched, monkeypatch):
    """非 admin 请求 graph_rag：通道在发起检索**之前**被拒（不是查完再丢弃），
    文档通道不受影响。拒在检索前是唯一安全位置——LLM 生成的 Cypher 一旦执行，
    结果侧过滤已不可信。"""
    events: list[dict] = []

    async def _hits(*args, **kwargs):
        return [{"text": "内容", "doc_name": "手册.pdf", "page_number": 1,
                 "chunk_index": 0, "score": 0.9}]

    async def _must_not_run(*args, **kwargs):
        raise AssertionError("非 admin 不应对图谱发起检索")

    monkeypatch.setattr(fusion_mod, "search_docs_raw", _hits)
    monkeypatch.setattr(fusion_mod, "search_graph_raw", _must_not_run)
    monkeypatch.setattr(get_settings(), "GRAPH_ACL_ALLOW_NON_ADMIN", False)

    result = await multi_channel_search(
        question="q", llm=_FakeLLM(), embedding_model=None,
        milvus_client=None, neo4j_driver=None,
        channels=["doc_rag", "graph_rag"], role="engineer",
        event_sink=events.append,
    )
    assert result["status"] == "ok"
    assert not result["failed_channels"]
    assert any(e.get("channel") == "graph_rag" and e.get("status") == "denied"
               for e in events)


async def test_graph_channel_denied_only_channel_yields_empty(patched, monkeypatch):
    """仅请求 graph_rag 且被拒 → status=empty（对用户表现为"未找到"，
    不暴露"存在你看不到的内容"）。"""
    monkeypatch.setattr(get_settings(), "GRAPH_ACL_ALLOW_NON_ADMIN", False)

    result = await multi_channel_search(
        question="q", llm=_FakeLLM(), embedding_model=None,
        milvus_client=None, neo4j_driver=None,
        channels=["graph_rag"], role="business",
    )
    assert result["status"] == "empty"


async def test_graph_channel_runs_for_admin(patched, monkeypatch):
    async def _hits(*args, **kwargs):
        return []

    async def _graph(*args, **kwargs):
        return [{"name": "电驱系统域"}]

    monkeypatch.setattr(fusion_mod, "search_docs_raw", _hits)
    monkeypatch.setattr(fusion_mod, "search_graph_raw", _graph)

    result = await multi_channel_search(
        question="q", llm=_FakeLLM(), embedding_model=None,
        milvus_client=None, neo4j_driver=None,
        channels=["graph_rag"], role="admin",
    )
    assert result["status"] == "ok"


async def test_fusion_forwards_role_to_doc_channel(patched, monkeypatch):
    """role 必须透传到文档检索层：在这里丢掉 = 整条文档通道退化成无权限检索。"""
    seen: list[dict] = []

    async def _capture(q, embedding_model, milvus_client, **kwargs):
        seen.append(kwargs)
        return []

    monkeypatch.setattr(fusion_mod, "search_docs_raw", _capture)

    await multi_channel_search(
        question="q", llm=_FakeLLM(), embedding_model=None,
        milvus_client=None, neo4j_driver=None,
        channels=["doc_rag"], role="aftersales",
    )
    assert seen and all(kw["role"] == "aftersales" for kw in seen)


# ── prod 防呆 ────────────────────────────────────────────────────────────

def test_prod_guard_rejects_acl_disabled():
    """prod 下关闭 DOC_ACL_ENABLED 必须拒绝启动：等于所有角色都能检索全部文档。"""
    with pytest.raises(ValueError) as exc:
        Settings(APP_ENV="prod", DOC_ACL_ENABLED=False)
    assert "DOC_ACL_ENABLED" in str(exc.value)


def test_prod_guard_rejects_graph_acl_optin():
    """prod 下 GRAPH_ACL_ALLOW_NON_ADMIN=true 必须拒绝启动：图谱节点级 ACL
    未实现，对非 admin 开放等于无权限隔离。"""
    with pytest.raises(ValueError) as exc:
        Settings(APP_ENV="prod", GRAPH_ACL_ALLOW_NON_ADMIN=True)
    assert "GRAPH_ACL_ALLOW_NON_ADMIN" in str(exc.value)
