# ============================================================
# 表格原图对照（bbox 裁剪）单元测试
# 覆盖：单页表格嵌入 / 跨页表格续页块归并多图 / 续页块无主块走附录 /
#       主块无法匹配走附录 / 无表格不修改 / 渲染失败跳过不阻塞
# 说明：monkeypatch 掉 MinIO 上传、pymupdf.open，只测对齐与嵌入逻辑
# ============================================================

import asyncio

import pytest
import pymupdf

from src.rag.ingestion import pipeline as pipe_mod


class FakePix:
    def tobytes(self, fmt: str) -> bytes:
        return b"PNG-DATA"


class FakePage:
    def __init__(self, rect, blocks):
        self.rect = rect
        self._blocks = blocks

    def get_text(self, kind: str):
        assert kind == "blocks"
        return self._blocks

    def get_pixmap(self, **kwargs):
        return FakePix()


class FakeDoc:
    def __init__(self, pages):
        self.pages = pages
        self.page_count = len(pages)

    def __getitem__(self, idx):
        return self.pages[idx]


A4 = pymupdf.Rect(0.0, 0.0, 595.3, 841.9)


def _block(page_idx, bbox, body=None):
    d = {"page_idx": page_idx, "bbox": bbox}
    if body is not None:
        d["table_body"] = body
    return d


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(pipe_mod, "upload_file", lambda *a, **k: None)
    monkeypatch.setattr(pipe_mod, "ensure_bucket_exists", lambda: None)
    # 绕过 __init__（会连 Milvus 建 collection），表格原图逻辑不使用实例状态
    return object.__new__(pipe_mod.IngestionPipeline)


@pytest.fixture
def fake_doc(monkeypatch):
    """拦截 pymupdf.open，返回文本层页面（含表格行文本块）"""
    rows = [(120.0, 240.0, 580.0, 300.0)]  # 表格文本块
    page = FakePage(A4, blocks=[(110.0, 240.0, 580.0, 300.0, "A", None, None)])
    doc = FakeDoc([page, page])
    monkeypatch.setattr(pymupdf, "open", lambda path: doc)
    return doc


def _run(p, md, blocks):
    return asyncio.run(
        p._embed_table_originals(md, blocks, "/tmp/fake.pdf", "doc123", "test.pdf")
    )


def _table_html(n: int) -> str:
    rows = "".join(f"<tr><td>{i}</td><td>内容{i}</td></tr>" for i in range(1, n + 1))
    return f"<table>{rows}</table>"


def test_single_page_table_embedded_after_close(pipeline, fake_doc):
    html = _table_html(3)
    md = f"前文\n{html}\n后文"
    out = _run(pipeline, md, [_block(0, [100, 100, 842, 400], body=html)])
    # 表格原图紧跟 </table> 之后、后文之前
    t_end = out.index("</table>")
    img = out.index("![表格原图](", t_end)
    assert out.index("后文") > img
    assert "images/doc123/table/p0_" in out


def test_cross_page_continuation_merged_to_single_table(pipeline, fake_doc):
    """跨页表格：主块（带合并 body）+ 续页块（body=None）→ 两张图嵌到同一 </table> 后"""
    html = _table_html(16)
    md = f"前文\n{html}\n后文"
    blocks = [
        _block(0, [147, 268, 842, 910], body=html),   # 首页主块（完整合并 HTML）
        _block(1, [147, 80, 842, 305], body=None),    # 续页块（仅 bbox）
    ]
    out = _run(pipeline, md, blocks)
    assert out.count("![表格原图](") == 2
    t_end = out.index("</table>")
    p1 = out.index("![表格原图](", t_end)
    p2 = out.index("![表格原图](", p1 + 1)
    assert t_end < p1 < p2 < out.index("后文")
    assert "p0_" in out and "p1_" in out  # 两页各裁剪一张


def test_continuation_without_anchor_goes_appendix(pipeline, fake_doc):
    """续页块（body=None）没有已匹配主块 → 文末附录，原图不丢"""
    md = "无表格\n正文"
    out = _run(pipeline, md, [_block(0, [147, 80, 842, 305], body=None)])
    assert "表格原图（识别对照）" in out
    assert "![表格原图](" in out


def test_unmatched_body_goes_appendix(pipeline, fake_doc):
    """主块 body 无法匹配任何 md <table> → 文末附录"""
    md = "<table><tr><td>X</td></tr></table>"
    blocks = [_block(0, [0, 0, 842, 400], body="<table><tr><td>完全不同的表</td></tr></table>")]
    out = _run(pipeline, md, blocks)
    assert "表格原图（识别对照）" in out
    assert out.count("![表格原图](") == 1


def test_no_table_blocks_returns_unchanged(pipeline, fake_doc):
    md = "纯文本 <table><tr><td>a</td></tr></table> 无表格块"
    assert _run(pipeline, md, []) == md


def test_render_failure_skips_block_not_blocking(pipeline, fake_doc, monkeypatch):
    """裁剪/上传失败只跳过单块，其余正常嵌入"""
    def _raise(*a, **k):
        raise RuntimeError("upload boom")
    monkeypatch.setattr(pipe_mod, "upload_file", _raise)
    html = _table_html(3)
    md = f"前文\n{html}\n后文"
    out = _run(pipeline, md, [_block(0, [100, 100, 842, 400], body=html)])
    assert "![表格原图](" not in out
    assert out == md


def test_multiple_tables_match_in_order(pipeline, fake_doc):
    """两个独立表格按文档顺序各嵌各的图，不错位"""
    html1, html2 = _table_html(2), _table_html(2)
    md = f"表一\n{html1}\n\n表二\n{html2}\n"
    blocks = [
        _block(0, [0, 0, 842, 400], body=html1),
        _block(1, [0, 0, 842, 400], body=html2),
    ]
    out = _run(pipeline, md, blocks)
    assert out.count("![表格原图](") == 2
    # 第一张图在第一个 </table> 后，第二张在第二个 </table> 后
    e1, e2 = out.index("</table>"), out.rindex("</table>")
    assert e1 < out.index("![表格原图](", e1) < e2
    assert e2 < out.index("![表格原图](", e2)
