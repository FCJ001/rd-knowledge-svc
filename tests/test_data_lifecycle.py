# ============================================================
# 数据生命周期：版本化写入 / 内容指纹决策 / 源侧对账 / 过期过滤
#
# 盯住的不变量：
#   1) chunk 主键跨版本不冲突，删旧版本的过滤器必须覆盖 null（补字段前的存量）
#   2) "内容未变更跳过"绝不吞掉元数据变更——chunk 冗余存着 ACL/车型过滤字段
#   3) 对账的权威源按文档来源分流：目录（脚本来源）vs MinIO（API 上传）
#   4) 过期谓词三种"不过滤"取值（null / 0 / 未来）都必须放行，缺一即存量不可见
# ============================================================

import sys
from pathlib import Path

import pytest

# 对账核心是 scripts/maintenance.py 里的纯函数，直接按路径导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from src.knowledge.doc_ingestion import parse_expire_date
from src.knowledge.doc_rag import expiry_expr
from src.rag.ingestion.pipeline import (
    PIPELINE_VERSION,
    chunk_pk,
    older_versions_filter,
)

from maintenance import DocView, compute_reconcile_plan

# ── 版本化写入 ───────────────────────────────────────────────────────────

def test_chunk_pk_is_versioned_and_collision_free():
    """同 doc 不同版本主键必须不同——这是"先插新后删旧"的前提。"""
    assert chunk_pk("abc123", 1, 0) == "abc123_v1_0"
    assert chunk_pk("abc123", 2, 0) == "abc123_v2_0"
    assert chunk_pk("abc123", 1, 0) != chunk_pk("abc123", 2, 0)


def test_older_versions_filter_covers_null_backfill_rows():
    """is null 分支不可省：add_collection_field 补字段前的存量行 doc_version
    为 null，Milvus 里 null 的数值比较恒 false——漏掉会让旧块永远删不掉。"""
    f = older_versions_filter("abc123", 3)
    assert 'doc_id == "abc123"' in f
    assert "doc_version is null" in f
    assert "doc_version < 3" in f


def test_older_versions_filter_escapes_doc_id():
    """doc_id 走 md5 是 hex，但过滤器构造必须仍按不可信输入转义（纵深防御）。"""
    f = older_versions_filter('x" or 1==1', 2)
    assert '"x\\" or 1==1"' in f


def test_pipeline_version_is_positive_int():
    """report/reingest 用它比较新旧；0 是存量"未标记"语义，不能与当前版本混淆。"""
    assert isinstance(PIPELINE_VERSION, int) and PIPELINE_VERSION >= 1


# ── expire_date 解析与过期谓词 ────────────────────────────────────────────

def test_parse_expire_date_empty_means_forever():
    assert parse_expire_date("") == 0
    assert parse_expire_date(None) == 0
    assert parse_expire_date("  ") == 0


def test_parse_expire_date_takes_end_of_day():
    """取当日 23:59:59：同日设置的过期文档当天仍可检索，次日起淡出。"""
    from datetime import datetime, time
    ts = parse_expire_date("2030-01-02")
    assert ts == int(datetime.combine(
        datetime.strptime("2030-01-02", "%Y-%m-%d").date(), time(23, 59, 59)
    ).timestamp())


def test_parse_expire_date_rejects_bad_format():
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        parse_expire_date("2030/01/02")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        parse_expire_date("not-a-date")


def test_expiry_expr_covers_all_not_expired_cases():
    """null（补字段前存量）/ <=0（未设置）/ >now（未到期）三种都必须放行。"""
    expr = expiry_expr()
    assert expr is not None
    assert "expire_ts is null" in expr
    assert "expire_ts <= 0" in expr
    assert "expire_ts > " in expr


def test_expiry_expr_disabled_returns_none(monkeypatch):
    from src.core.config import get_settings
    monkeypatch.setattr(get_settings(), "DOC_EXPIRE_FILTER_ENABLED", False)
    assert expiry_expr() is None


# ── 上传侧"内容未变更跳过"的元数据守卫 ──────────────────────────────────

class _FakeDoc:
    """路由侧 _ingest_metadata_changed 只读这些字段，用假对象免建 ORM 行。"""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _base_doc():
    return _FakeDoc(
        doc_type="spec_doc", category="高压", business_line="aftersales",
        model_code="EV160", acl_roles="engineer,business",
        chunk_strategy="fixed", expire_date="",
    )


def _kwargs(**overrides):
    kw = dict(
        doc_type="spec_doc", category="高压", business_line="aftersales",
        model_code="EV160", acl_roles="engineer,business",
        chunk_strategy="fixed", expire_date="",
    )
    kw.update(overrides)
    return kw


def test_metadata_unchanged_when_all_equal():
    from src.api.routers.ingest import _ingest_metadata_changed
    assert _ingest_metadata_changed(_base_doc(), **_kwargs()) is False


def test_metadata_change_forces_reingest():
    """ACL/车型/类型/失效日期任一变化都必须走全量重入库——chunk 里冗余存着
    这些过滤字段，跳过管线只改 PG 会让"改了可见角色却不生效"。"""
    from src.api.routers.ingest import _ingest_metadata_changed
    doc = _base_doc()
    assert _ingest_metadata_changed(doc, **_kwargs(acl_roles="engineer")) is True
    assert _ingest_metadata_changed(doc, **_kwargs(model_code="EV200")) is True
    assert _ingest_metadata_changed(doc, **_kwargs(doc_type="repair_manual")) is True
    assert _ingest_metadata_changed(doc, **_kwargs(expire_date="2030-01-01")) is True


def test_metadata_none_vs_empty_string_treated_equal():
    """PG 的 NULL 与上传的空串语义相同（未设置），不算变更。"""
    from src.api.routers.ingest import _ingest_metadata_changed
    doc = _base_doc()
    doc.category = None
    doc.model_code = None
    assert _ingest_metadata_changed(doc, **_kwargs(category="", model_code="")) is False


# ── 源侧对账（纯函数）────────────────────────────────────────────────────

def test_reconcile_splits_authority_by_provenance():
    """权威源分流：minio_key 为空看目录、非空看 MinIO——混用口径会把
    API 上传的文档全部误判为"源已消失"。"""
    pg = [
        DocView("d1", "脚本来源在目录.pdf", "indexed", minio_key=None),
        DocView("d2", "脚本来源已消失.pdf", "indexed", minio_key=None),
        DocView("d3", "API上传在MinIO.pdf", "indexed", minio_key="spec_doc/common/API上传在MinIO.pdf"),
        DocView("d4", "API上传对象丢了.pdf", "indexed", minio_key="spec_doc/common/API上传对象丢了.pdf"),
        DocView("d5", "已软删残留.pdf", "deleted", minio_key=None),
    ]
    source = {"脚本来源在目录.pdf", "新文档待入库.pdf"}
    milvus = {
        "脚本来源在目录.pdf": 10,
        "脚本来源已消失.pdf": 5,      # PG 在库但源没了 → 孤儿文档（软删会连带清块）
        "API上传在MinIO.pdf": 8,
        "API上传对象丢了.pdf": 4,     # MinIO 对象没了但块还在 → 孤儿文档（块随软删清）
        "幽灵文档.pdf": 3,            # PG 完全没有 → 孤儿块
        "已软删残留.pdf": 2,          # PG status=deleted → 孤儿块
    }
    minio_keys = {"spec_doc/common/API上传在MinIO.pdf"}  # d4 的对象不存在

    plan = compute_reconcile_plan(source, pg, milvus, lambda k: k in minio_keys)

    assert [d.doc_name for d in plan.orphan_docs] == ["脚本来源已消失.pdf", "API上传对象丢了.pdf"]
    assert plan.missing_in_index == ["新文档待入库.pdf"]
    assert {name for name, _, _ in plan.milvus_orphans} == {"幽灵文档.pdf", "已软删残留.pdf"}
    assert plan.broken_docs == []


def test_reconcile_detects_half_failed_docs():
    """PG=indexed 但 Milvus 零块 = 半失败态（入库中断），必须显式报告。"""
    pg = [DocView("d1", "半失败.pdf", "indexed", minio_key=None)]
    plan = compute_reconcile_plan({"半失败.pdf"}, pg, {"半失败.pdf": 0}, lambda k: True)
    assert [d.doc_name for d in plan.broken_docs] == ["半失败.pdf"]
    assert plan.orphan_docs == []
    assert plan.missing_in_index == []


def test_reconcile_all_consistent():
    pg = [DocView("d1", "一致.pdf", "indexed", minio_key=None)]
    plan = compute_reconcile_plan({"一致.pdf"}, pg, {"一致.pdf": 7}, lambda k: True)
    assert plan.missing_in_index == []
    assert plan.orphan_docs == []
    assert plan.milvus_orphans == []
    assert plan.broken_docs == []
