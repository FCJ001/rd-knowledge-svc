# ============================================================
# 公式原图对照（bbox 裁剪）单元测试
# 覆盖：公式后插入原图 / 多公式按文档顺序对齐 / 匹配不上走附录 /
#       无公式块不修改 / 渲染失败跳过不阻塞 / 行内公式不参与
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
    def __init__(self, rect):
        self.rect = rect

    def get_text(self, kind: str):
        assert kind == "blocks"
        return []

    def get_pixmap(self, **kwargs):
        return FakePix()


class FakeDoc:
    def __init__(self, pages):
        self.pages = pages
        self.page_count = len(pages)

    def __getitem__(self, idx):
        return self.pages[idx]


A4 = pymupdf.Rect(0.0, 0.0, 595.3, 841.9)


def _equation(page_idx, bbox, text):
    return {"page_idx": page_idx, "bbox": bbox, "text": text}


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(pipe_mod, "upload_file", lambda *a, **k: None)
    monkeypatch.setattr(pipe_mod, "ensure_bucket_exists", lambda: None)
    # 绕过 __init__（会连 Milvus 建 collection），公式原图逻辑不使用实例状态
    # 定位器 _locate_formula_rect 依赖 page.search_for（FakePage 没有），
    # 且属于渲染像素细节，此处固定返回 bbox 附近一个合理裁剪框
    def _fake_locate(page, bbox, md_before):
        return pymupdf.Rect(
            max(bbox.x0 - 6.0, 0.0), max(bbox.y0 - 14.0, 0.0),
            min(bbox.x1 + 20.0, page.rect.width),
            min(bbox.y1 + 18.0, page.rect.height),
        )
    monkeypatch.setattr(pipe_mod, "_locate_formula_rect", _fake_locate)
    return object.__new__(pipe_mod.IngestionPipeline)


@pytest.fixture
def fake_doc(monkeypatch):
    """拦截 pymupdf.open，返回 3 页的页面"""
    doc = FakeDoc([FakePage(A4) for _ in range(3)])
    monkeypatch.setattr(pymupdf, "open", lambda path: doc)
    return doc


def _run(p, md, equations):
    return asyncio.run(
        p._embed_formula_originals(md, equations, "/tmp/fake.pdf", "doc123", "test.pdf")
    )


def test_formula_embedded_after_display_formula(pipeline, fake_doc):
    md = "前文\n$$\nR = \\frac{U}{I}\n$$\n后文"
    eq = _equation(0, [410, 828, 546, 870], "$$\nR = \\frac{U}{I}\n$$")
    out = _run(pipeline, md, [eq])
    # 原图紧跟公式之后、后文之前；命名进 formula 目录
    f_end = out.index("$$") + 1
    img = out.index("![公式原图](", f_end)
    assert out.index("后文") > img
    assert "images/doc123/formula/p0_" in out


def test_multiple_formulas_match_in_document_order(pipeline, fake_doc):
    md = "一\n$$\nR = U/I\n$$\n二\n$$\nP = U*I\n$$\n三"
    eqs = [
        _equation(0, [410, 828, 546, 870], "$$\nR = U/I\n$$"),
        _equation(1, [410, 828, 546, 870], "$$\nP = U*I\n$$"),
    ]
    out = _run(pipeline, md, eqs)
    assert out.count("![公式原图](") == 2
    r1 = out.index("![公式原图](")
    r2 = out.index("![公式原图](", r1 + 1)
    f1 = out.index("$$\nR = U/I\n$$")
    f2 = out.index("$$\nP = U*I\n$$")
    # 第一张图在公式1后、公式2前；第二张在公式2后
    assert f1 < r1 < f2 < r2
    # 跨页：p0_、p1_ 各裁剪一张
    assert "p0_" in out and "p1_" in out


def test_unmatched_formula_goes_appendix(pipeline, fake_doc):
    """equation 的 LaTeX 在 md 中找不到对应公式 → 文末附录，原图不丢"""
    md = "正文没有这个公式"
    eq = _equation(0, [410, 828, 546, 870], "$$\nX = 1\n$$")
    out = _run(pipeline, md, [eq])
    assert "公式原图（识别对照）" in out
    assert "![公式原图](" in out


def test_no_equation_blocks_returns_unchanged(pipeline, fake_doc):
    md = "纯文本，无公式"
    assert _run(pipeline, md, []) == md


def test_render_failure_skips_formula_not_blocking(pipeline, fake_doc, monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("upload boom")
    monkeypatch.setattr(pipe_mod, "upload_file", _raise)
    md = "$$\nR = U/I\n$$"
    eq = _equation(0, [410, 828, 546, 870], "$$\nR = U/I\n$$")
    out = _run(pipeline, md, [eq])
    assert "![公式原图](" not in out
    assert out == md


def test_inline_formula_not_a_target(pipeline, fake_doc):
    """行内公式 $..$ 不是 equation 块的目标；原图只插到行间公式后"""
    md = "电压 $U_{max}$ 与公式 $$\nR = U/I\n$$\n"
    eq = _equation(0, [410, 828, 546, 870], "$$\nR = U/I\n$$")
    out = _run(pipeline, md, [eq])
    assert "![公式原图](" in out
    assert out.index("![公式原图](") > out.index("$$\nR = U/I\n$$")
