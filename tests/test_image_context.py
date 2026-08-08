# ============================================================
# 图片 VL 摘要辅助上下文提取测试
# 覆盖：取图前段落 / 空行截断 / 邻居图引用剔除 / 引用缺失返回空
# ============================================================

from src.rag.ingestion.pipeline import IngestionPipeline


def test_context_takes_preceding_paragraph():
    md = "本节介绍 EV160 电池包结构，高压接插件位于电池包后部。\n![](images/fig1.png)"
    ctx = IngestionPipeline._extract_image_context(md, "images/fig1.png")
    assert "EV160" in ctx
    assert "电池包" in ctx
    assert "![]" not in ctx  # 图片引用本身不混入


def test_context_stops_at_blank_line():
    md = "上一段落与图片无关。\n\n2.3 电池过热排查，图1 展示 BMS 位置。\n![](images/fig2.png)"
    ctx = IngestionPipeline._extract_image_context(md, "images/fig2.png")
    assert "上一段落" not in ctx  # 空行之前的内容被截掉
    assert "BMS" in ctx


def test_context_excludes_neighbor_image_ref():
    md = "图A 说明。![](images/fig0.png)\n这是图1的正文，讲高压回路。\n![](images/fig1.png)"
    ctx = IngestionPipeline._extract_image_context(md, "images/fig1.png")
    assert "![]" not in ctx  # 邻居图的引用字符串不混入
    assert "高压回路" in ctx


def test_context_empty_when_ref_missing():
    ctx = IngestionPipeline._extract_image_context("没有这张图", "images/fig9.png")
    assert ctx == ""
