# ============================================================
# 表格 VL 增强通道单元测试
# 覆盖：VL 开启时摘要+校读插入 / 开关关闭不调用 / VL 失败或空结果保持原样 /
#       VL 异常 fail-open / 续页块（body=None）不触发 VL
# 说明：monkeypatch summarize_table 与 MinIO/pymupdf，不打真实网络
# ============================================================

import asyncio

import pymupdf
import pytest
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

VL_MD = "| 螺栓 | 扭矩 |\n|---|---|\n| M12 | 130N·m |"


def _fake_summarizer(summary: str, md: str, calls: list | None = None):
    async def _summarize(img_bytes, mime_type="image/png"):
        if calls is not None:
            calls.append(img_bytes)
        return summary, md

    return _summarize


def _block(page_idx, bbox, body=None):
    d = {"page_idx": page_idx, "bbox": bbox}
    if body is not None:
        d["table_body"] = body
    return d


def _table_html(n: int) -> str:
    rows = "".join(f"<tr><td>{i}</td><td>内容{i}</td></tr>" for i in range(1, n + 1))
    return f"<table>{rows}</table>"


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(pipe_mod, "upload_file", lambda *a, **k: None)
    monkeypatch.setattr(pipe_mod, "ensure_bucket_exists", lambda: None)
    monkeypatch.setattr(pipe_mod.settings, "TABLE_VL_ENABLED", True)
    return object.__new__(pipe_mod.IngestionPipeline)


@pytest.fixture
def fake_doc(monkeypatch):
    page = FakePage(A4, blocks=[(110.0, 240.0, 580.0, 300.0, "A", None, None)])
    doc = FakeDoc([page, page])
    monkeypatch.setattr(pymupdf, "open", lambda path: doc)
    return doc


def _run(p, md, blocks):
    return asyncio.run(
        p._embed_table_originals(md, blocks, "/tmp/fake.pdf", "doc123", "test.pdf")
    )


def test_vl_summary_and_markdown_inserted(pipeline, fake_doc, monkeypatch):
    """VL 成功：摘要在原图引用之后、后文之前，校读 Markdown 逐行带引用前缀"""
    calls: list = []
    monkeypatch.setattr(pipe_mod, "summarize_table", _fake_summarizer("列出扭矩值", VL_MD, calls))
    html = _table_html(3)
    md = f"前文\n{html}\n后文"
    out = _run(pipeline, md, [_block(0, [100, 100, 842, 400], body=html)])

    t_end = out.index("</table>")
    img = out.index("![表格原图](", t_end)
    note = out.index("> 表格摘要：列出扭矩值", img)
    assert out.index("> VL 校读：", note) > note
    assert "| M12 | 130N·m |" in out
    assert out.index("后文") > out.index(VL_MD.splitlines()[-1])
    # VL 拿到的正是裁剪出的同一份字节
    assert calls and calls[0] == b"PNG-DATA"


def test_vl_disabled_no_call(pipeline, fake_doc, monkeypatch):
    monkeypatch.setattr(pipe_mod.settings, "TABLE_VL_ENABLED", False)
    calls: list = []
    monkeypatch.setattr(pipe_mod, "summarize_table", _fake_summarizer("x", "y", calls))
    html = _table_html(3)
    md = f"前文\n{html}\n后文"
    out = _run(pipeline, md, [_block(0, [100, 100, 842, 400], body=html)])
    assert calls == []
    assert "表格摘要" not in out and "![表格原图](" in out


def test_vl_empty_result_keeps_original(pipeline, fake_doc, monkeypatch):
    """VL 返回空（fail-open）→ 只有原图，无摘要/校读块"""
    monkeypatch.setattr(pipe_mod, "summarize_table", _fake_summarizer("", ""))
    html = _table_html(3)
    md = f"前文\n{html}\n后文"
    out = _run(pipeline, md, [_block(0, [100, 100, 842, 400], body=html)])
    assert "表格摘要" not in out and "VL 校读" not in out
    assert "![表格原图](" in out


def test_vl_exception_failopen(pipeline, fake_doc, monkeypatch):
    async def _boom(img_bytes, mime_type="image/png"):
        raise RuntimeError("vl boom")

    monkeypatch.setattr(pipe_mod, "summarize_table", _boom)
    html = _table_html(3)
    md = f"前文\n{html}\n后文"
    out = _run(pipeline, md, [_block(0, [100, 100, 842, 400], body=html)])
    assert "![表格原图](" in out and "表格摘要" not in out


def test_continuation_block_skips_vl(pipeline, fake_doc, monkeypatch):
    """跨页续页块（body=None）只归并原图，不触发 VL（VL 只对主块做一次）"""
    calls: list = []
    monkeypatch.setattr(pipe_mod, "summarize_table", _fake_summarizer("摘要", VL_MD, calls))
    html = _table_html(16)
    md = f"前文\n{html}\n后文"
    blocks = [
        _block(0, [147, 268, 842, 910], body=html),
        _block(1, [147, 80, 842, 305], body=None),
    ]
    out = _run(pipeline, md, blocks)
    assert len(calls) == 1  # 只有一次 VL 调用
    assert out.count("![表格原图](") == 2  # 两页图都嵌了
    assert out.count("> 表格摘要：") == 1


# ── 双通道数值互查 ──


def test_numbers_consistent_no_warning(pipeline, fake_doc, monkeypatch):
    """两份转写数值一致 → 无 ⚠️ 标记"""
    html = "<table><tr><td>M12</td><td>130</td></tr></table>"
    vl = "| 螺栓 | 扭矩 |\n|---|---|\n| M12 | 130 |"
    monkeypatch.setattr(pipe_mod, "summarize_table", _fake_summarizer("扭矩表", vl))
    md = f"前文\n{html}\n后文"
    out = _run(pipeline, md, [_block(0, [100, 100, 842, 400], body=html)])
    assert "双通道转写数值不一致" not in out


def test_numbers_mismatch_warning(pipeline, fake_doc, monkeypatch):
    """两份转写数值冲突 → 插入 ⚠️ 标记并列出双方差异值"""
    html = "<table><tr><td>M12</td><td>130</td></tr></table>"
    vl = "| 螺栓 | 扭矩 |\n|---|---|\n| M12 | 135 |"
    monkeypatch.setattr(pipe_mod, "summarize_table", _fake_summarizer("扭矩表", vl))
    md = f"前文\n{html}\n后文"
    out = _run(pipeline, md, [_block(0, [100, 100, 842, 400], body=html)])
    assert "双通道转写数值不一致" in out
    assert "130" in out and "135" in out


def test_number_format_normalized(pipeline, fake_doc, monkeypatch):
    """千分位逗号与尾零归一化：1,300==1300、1.50==1.5 不误报"""
    html = "<table><tr><td>1,300</td><td>1.50</td></tr></table>"
    vl = "| a | b |\n|---|---|\n| 1300 | 1.5 |"
    monkeypatch.setattr(pipe_mod, "summarize_table", _fake_summarizer("数值表", vl))
    md = f"前文\n{html}\n后文"
    out = _run(pipeline, md, [_block(0, [100, 100, 842, 400], body=html)])
    assert "双通道转写数值不一致" not in out


def test_single_side_extras_no_warning(pipeline, fake_doc, monkeypatch):
    """仅单侧多出数值（如 VL 对条款号省略、认不准写 null）不算实质分歧"""
    html = "<table><tr><td>3.2.1</td><td>500</td></tr></table>"
    vl = "| 项目 | 说明 |\n|---|---|\n| null | null |"
    monkeypatch.setattr(pipe_mod, "summarize_table", _fake_summarizer("规格表", vl))
    md = f"前文\n{html}\n后文"
    out = _run(pipeline, md, [_block(0, [100, 100, 842, 400], body=html)])
    assert "双通道转写数值不一致" not in out
