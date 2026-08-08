# ============================================================
# 公式原图对照（双通道）单元测试
# 覆盖：公式 token 序列识别 / 行间公式嵌入原图（行内跳过）/
#       数量不匹配走文末附录 / 无原图不修改 / 空 markdown
# 说明：monkeypatch 掉 MinIO 上传与 VL 调用，只测对齐与嵌入逻辑
# ============================================================

import asyncio

import pytest

from src.rag.ingestion import pipeline as pipe_mod


def _formula_images(n: int) -> list[tuple[str, bytes]]:
    return [(f"images/f{i}.png", f"img{i}".encode()) for i in range(n)]


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(pipe_mod, "upload_file", lambda *a, **k: None)
    monkeypatch.setattr(pipe_mod.settings, "IMAGE_SUMMARIZE_ENABLED", False)
    # 绕过 __init__（会连 Milvus 建 collection），_embed_formula_images 不使用实例状态
    return object.__new__(pipe_mod.IngestionPipeline)


def _run(p, md, images):
    return asyncio.run(p._embed_formula_images(md, images, "doc123", "test.pdf"))


def _formula_types(md):
    import re
    pat = re.compile(r"\$\$[\s\S]+?\$\$|\$[^$\n]+?\$")
    return [tok.startswith("$$") for tok in pat.findall(md)]


def test_display_only_embedded_in_order(pipeline):
    md = "前文\n$$\nR = \\frac{U}{I}\n$$\n后文\n$$\nP = UI\n$$\n末尾"
    out = _run(pipeline, md, _formula_images(2))
    assert _formula_types(out) == [True, True]
    # 每个行间公式后都紧跟一张原图引用，且顺序一致
    ref0 = "![公式原图](http://localhost:9000/knowledge-docs/images/doc123/formula/f0.png)"
    ref1 = "![公式原图](http://localhost:9000/knowledge-docs/images/doc123/formula/f1.png)"
    assert ref0 in out and ref1 in out
    assert out.index(ref0) < out.index(ref1)
    # 原图紧跟公式之后（位于后续正文之前）
    assert "$$\nR = \\frac{U}{I}\n$$\n\n" + ref0 in out
    assert "$$\nP = UI\n$$\n\n" + ref1 in out


def test_inline_formula_gets_no_image_but_keeps_alignment(pipeline):
    md = "电压 $U_{max}$ 与电阻 $$\nR\n$$\n及电流 $I$ 与功率 $$\nP\n$$\n"
    out = _run(pipeline, md, _formula_images(4))
    types = _formula_types(out)
    assert types == [False, True, False, True]
    # 行内公式 $U_{max}$、$I$ 后不出现图片；行间公式后紧跟对应原图
    ref0 = "![公式原图](http://localhost:9000/knowledge-docs/images/doc123/formula/f0.png)"
    ref1 = "![公式原图](http://localhost:9000/knowledge-docs/images/doc123/formula/f1.png)"
    ref2 = "![公式原图](http://localhost:9000/knowledge-docs/images/doc123/formula/f2.png)"
    ref3 = "![公式原图](http://localhost:9000/knowledge-docs/images/doc123/formula/f3.png)"
    # f0、f2 对应行内公式，不嵌入；f1、f3 对应行间公式 R/P，嵌入
    assert ref0 not in out and ref2 not in out
    assert ref1 in out and ref3 in out


def test_count_mismatch_falls_back_to_appendix(pipeline):
    md = "只有两个公式 $$\nA\n$$\n和行内 $x$，但有三张原图"
    out = _run(pipeline, md, _formula_images(3))
    assert "公式原图（识别对照）" in out
    # 三个原图都进附录，不逐公式嵌入
    assert out.count("![公式原图](") == 3
    assert "$$\nA\n$$\n\n\n!" not in out


def test_no_formula_images_returns_unchanged(pipeline):
    md = "纯文本，无公式无图片"
    out = _run(pipeline, md, [])
    assert out == md


def test_formula_inline_and_display_count(pipeline):
    md = "a $x$ b $$y$$ c $z$ d $$w$$"
    types = _formula_types(md)
    assert types == [False, True, False, True]
    assert len(types) == 4
