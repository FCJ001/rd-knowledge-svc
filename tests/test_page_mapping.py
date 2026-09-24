# ============================================================
# 页码链路：解析锚点 → chunk 映射 → 引用渲染 → 页级评测标签
#
# 背景（真实缺陷）：实测 1097 个 chunk 的 page_number 全是 0，
# 原因是 MinerU 的 v2 协议路径从不填充 pages，而 pipeline 又用
# `pages[chunk_idx]` 取页码——把「块下标」当「切片下标」用。
# 两处都已修，这里锁定行为。
# ============================================================

from dataclasses import dataclass

from src.knowledge.doc_rag import format_doc_context
from src.knowledge.mineru_client import (
    _block_anchor_text,
    _page_anchors_from_content_list,
)
from src.rag.evaluation.retrieval_metrics import build_page_ref, parse_page_ref
from src.rag.ingestion.pipeline import _assign_chunk_pages


@dataclass
class _Chunk:
    text: str


# ── 解析期：content_list → 页码锚点 ──────────────────────────────

def test_anchors_are_one_based():
    """page_idx 是 0-based，锚点页码要 +1——否则首页的 0 会与"未知"撞值。"""
    anchors = _page_anchors_from_content_list([
        {"type": "text", "page_idx": 0, "text": "第一章 概述 内容足够长可以定位"},
        {"type": "text", "page_idx": 2, "text": "第三章 电池系统 内容也足够长"},
    ])
    assert [p for p, _ in anchors] == [1, 3]


def test_anchors_skip_too_short_blocks():
    """过短片段无法可靠定位，必须丢弃而不是当成锚点。"""
    anchors = _page_anchors_from_content_list([
        {"type": "text", "page_idx": 0, "text": "短"},
        {"type": "text", "page_idx": 1, "text": "足够长的正文片段用于定位测试"},
    ])
    assert len(anchors) == 1
    assert anchors[0][0] == 2


def test_anchors_accept_json_string_and_empty():
    assert _page_anchors_from_content_list("") == []
    assert _page_anchors_from_content_list("not json") == []
    assert _page_anchors_from_content_list([{"page_idx": 0, "text": "x" * 20}])[0][0] == 1


def test_anchor_text_reads_multiple_block_shapes():
    """不同 type 的块，文本键不同——都要能取到。"""
    assert _block_anchor_text({"type": "text", "text": "正文内容"}) == "正文内容"
    assert _block_anchor_text({"type": "table", "table_body": "<table/>"}) == "<table/>"
    assert _block_anchor_text({"type": "image", "img_caption": ["图1", "说明"]}) == "图1 说明"
    assert _block_anchor_text({"type": "image"}) == ""


# ── 入库期：chunk → 页码 ────────────────────────────────────────

def test_chunk_pages_map_by_text_anchor():
    anchors = [
        (1, "第一章 概述 这是第一章的正文内容"),
        (5, "第五章 制动系统 这是第五章的正文内容"),
    ]
    chunks = [
        _Chunk("第一章 概述 这是第一章的正文内容"),
        _Chunk("这是第五章的正文内容 制动系统的补充说明"),
    ]
    assert _assign_chunk_pages(chunks, anchors) == [1, 5]


def test_chunk_page_carries_forward_when_unmatched():
    """匹配不到就沿用上一个已知页码（文档序下是合理推断），不凭空猜。"""
    anchors = [(3, "第三章 内容片段足够长用于定位")]
    chunks = [
        _Chunk("第三章 内容片段足够长用于定位"),
        _Chunk("这一段完全匹配不上任何锚点文本内容"),
    ]
    assert _assign_chunk_pages(chunks, anchors) == [3, 3]


def test_no_anchors_yields_all_unknown():
    """一个锚点都没有 → 全部 0（未知），绝不猜页码。

    错误的引用比没有引用更糟：显示"第7页"而实际在第3页，
    用户会按错误页码去翻手册。
    """
    chunks = [_Chunk("任意内容"), _Chunk("另一段内容")]
    assert _assign_chunk_pages(chunks, []) == [0, 0]


def test_unmatched_chunks_stay_unknown_when_nothing_matched():
    """锚点存在但全都没匹配上 → 仍为 0，不能沿用"初始 last_known"。"""
    anchors = [(4, "第四章 内容片段足够长用于定位")]
    chunks = [_Chunk("毫不相关的内容无法匹配任何锚点")]
    assert _assign_chunk_pages(chunks, anchors) == [0]


def test_empty_chunks_returns_empty():
    assert _assign_chunk_pages([], [(1, "任意长文本片段用于定位")]) == []


# ── 页级引用与渲染 ─────────────────────────────────────────────

def test_page_ref_treats_zero_as_unknown():
    """0 是「未知」哨兵，不是第 0 页——否则会造出一堆 xxx.pdf:p0 的假标签。"""
    assert build_page_ref("手册.pdf", 0) == ""
    assert build_page_ref("手册.pdf", None) == ""
    assert build_page_ref("手册.pdf", "") == ""
    assert build_page_ref("手册.pdf", 12) == "手册.pdf:p12"
    assert build_page_ref("", 12) == ""


def test_parse_page_ref_round_trip():
    assert parse_page_ref("手册.pdf:p12") == ("手册.pdf", "12")
    assert parse_page_ref("a:b.pdf:p3") == ("a:b.pdf", "3")   # 文档名含冒号
    assert parse_page_ref("手册.pdf") is None
    assert parse_page_ref(":p12") is None
    assert parse_page_ref("") is None


def test_citation_hides_page_when_unknown():
    """页码未知时不显示页码，而不是显示"第0页"。"""
    hits = [{"text": "内容", "doc_name": "手册.pdf", "page_number": 0}]
    ctx = format_doc_context(hits)
    assert "第0页" not in ctx
    assert "手册.pdf" in ctx


def test_citation_shows_real_page():
    hits = [{"text": "内容", "doc_name": "手册.pdf", "page_number": 42}]
    assert "第42页" in format_doc_context(hits)


def test_page_mapping_does_not_run_away_to_last_anchor():
    """回归：全局取最大重叠会让一次错误匹配把游标带到文末。

    实测后果：289 页手册上 727 块里有 604 块塌到最后一页。
    修法是「窗口内首个达标即命中」——真匹配就在游标附近。
    """
    # 构造：前 5 块的锚点在前方，第 6 块与文末锚点有高重叠（诱饵）
    anchors = [(p, f"第{p}章 正文内容片段用于定位测试 {p}") for p in range(1, 6)]
    anchors.append((99, "第99章 正文内容片段用于定位测试 99"))
    chunks = [_Chunk(f"第{p}章 正文内容片段用于定位测试 {p}") for p in range(1, 6)]
    chunks.append(_Chunk("第99章 正文内容片段用于定位测试 99"))

    pages = _assign_chunk_pages(chunks, anchors)
    assert pages[:5] == [1, 2, 3, 4, 5]
    assert pages[5] == 99


def test_page_mapping_is_monotonic():
    """页码必须单调不减（chunk 与锚点都是文档序）。"""
    anchors = [(p, f"章节{p} 的正文内容片段足够长用于定位") for p in (1, 3, 7, 12)]
    chunks = [
        _Chunk("章节1 的正文内容片段足够长用于定位"),
        _Chunk("完全无法匹配的文本内容"),
        _Chunk("章节7 的正文内容片段足够长用于定位"),
        _Chunk("章节12 的正文内容片段足够长用于定位"),
    ]
    pages = _assign_chunk_pages(chunks, anchors)
    assert pages == sorted(pages)
    assert pages == [1, 1, 7, 12]


def test_window_limits_forward_jump_on_noise():
    """窗口限制：与远处锚点的高重叠不应把游标拽过去。"""
    anchors = [(1, "开头内容片段足够长用于定位测试")]
    anchors += [(50 + i, f"无关内容{i} 的片段足够长用于定位测试") for i in range(200)]
    chunks = [_Chunk("开头内容片段足够长用于定位测试"), _Chunk("另一个无法匹配的短文本")]
    pages = _assign_chunk_pages(chunks, anchors, window=10)
    assert pages == [1, 1]


def test_soft_jump_guard_blocks_low_confidence_far_jump():
    """软跳变守卫：低重叠加远跳 = 误配，不采纳。

    真实远跳（文档中途缺文本）会近乎精确匹配，那种应放行；
    只有"像但其实不像"的巧合匹配才该拦。
    """
    anchors = [
        (1, "开头的内容片段足够长用于定位测试"),
        (200, "完全无关的另一段内容片段足够长用于定位"),   # 与下一块只有部分重叠
    ]
    chunks = [
        _Chunk("开头的内容片段足够长用于定位测试"),
        _Chunk("完全无关的第三段内容片段足够长用于定位"),
    ]
    pages = _assign_chunk_pages(chunks, anchors, window=10)
    assert pages == [1, 1], pages
