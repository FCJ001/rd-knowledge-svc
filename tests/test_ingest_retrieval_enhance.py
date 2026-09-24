# ============================================================
# 入库侧检索增强：表格/公式块保护、章节前缀、低置信拒答
#
# 这三项的共同目标：让"检索到的内容"更完整、更可定位、更不容易被硬编。
# ============================================================

from dataclasses import dataclass

from src.knowledge import fusion as fusion_mod
from src.knowledge.fusion import _should_refuse, _top_doc_score
from src.rag.config import ChunkingConfig
from src.rag.ingestion.chunkers import (
    FixedChunker,
    ProtectedBlockChunker,
    merge_short_chunks,
    split_protected_blocks,
)
from src.rag.ingestion.pipeline import (
    _assign_chunk_sections,
    _build_chunk_prefix,
    _heading_paths,
)


@dataclass
class _Chunk:
    text: str


# ── 表格 / 公式块保护 ───────────────────────────────────────────

TABLE_WITH_NOTES = """<table><tr><td>项目</td><td>限值</td></tr></table>
> 表格摘要：汇总了绝缘电阻要求。
> VL 校读：| 项目 | 限值 |
![表格原图](http://x/t.png)"""


def test_table_block_keeps_its_annotations():
    """注解行（摘要/VL 校读/原图）语义上属于这张表，必须跟着一起走。

    丢掉它们等于丢掉表格的解释——这正是"原图对照"功能的价值所在。
    """
    segs = split_protected_blocks(f"正文甲。\n\n{TABLE_WITH_NOTES}\n\n正文乙。")
    atomic = [s for s, a in segs if a]
    assert len(atomic) == 1
    assert "</table>" in atomic[0]
    assert "表格摘要" in atomic[0]
    assert "VL 校读" in atomic[0]
    assert "表格原图" in atomic[0]


def test_formula_block_is_protected():
    md = "公式如下：\n$$ R = \\frac{U}{1000} $$\n![公式原图](http://x/f.png)\n尾部。"
    atomic = [s for s, a in split_protected_blocks(md) if a]
    assert len(atomic) == 1
    assert "$$" in atomic[0] and "公式原图" in atomic[0]


def test_segments_preserve_document_order():
    segs = split_protected_blocks("甲。\n\n<table><tr/></table>\n\n乙。")
    assert [a for _, a in segs] == [False, True, False]
    assert segs[0][0].startswith("甲")
    assert segs[2][0].startswith("乙")


def test_protected_chunker_does_not_split_atomic_blocks():
    """大表格在基础切分器下会被腰斩；保护后必须整块保留。"""
    md = f"前置说明。\n\n{TABLE_WITH_NOTES}\n\n后置说明。"
    base = FixedChunker(ChunkingConfig(chunk_size=50, chunk_overlap=0))
    protected = ProtectedBlockChunker(base).chunk(md, {"doc_name": "x.pdf"})

    atomic = [c for c in protected if c.metadata.get("atomic")]
    assert len(atomic) == 1
    assert atomic[0].text.count("</table>") == 1
    assert "表格原图" in atomic[0].text

    # 对照：不加保护时同一张表会被切开
    naive = base.chunk(md, {"doc_name": "x.pdf"})
    assert not any(c.metadata.get("atomic") for c in naive)


def test_merge_short_chunks_skips_atomic():
    """短块合并不能把受保护块并进去——那等于又破坏了边界。"""
    md = f"很短。\n\n{TABLE_WITH_NOTES}\n\n也很短。"
    base = FixedChunker(ChunkingConfig(chunk_size=200, chunk_overlap=0))
    chunks = ProtectedBlockChunker(base).chunk(md, {"doc_name": "x.pdf"})
    merged = merge_short_chunks(chunks, min_chars=200, max_chars=800)
    atomic = [c for c in merged if c.metadata.get("atomic")]
    assert len(atomic) == 1
    assert "表格原图" in atomic[0].text and "</table>" in atomic[0].text


def test_atomic_chunk_has_parent_text_for_context_formatting():
    """受保护块没有"父块"，但生成阶段会读 parent_text，需置为自身。"""
    md = TABLE_WITH_NOTES
    chunks = ProtectedBlockChunker(FixedChunker(ChunkingConfig())).chunk(md, {})
    assert chunks[0].metadata.get("parent_text") == chunks[0].text


def test_protect_blocks_can_be_disabled():
    cfg = ChunkingConfig(protect_blocks=False)
    assert cfg.protect_blocks is False


# ── 章节归属与检索用前缀 ────────────────────────────────────────

MD = """# 车型维修手册

## 第五章 制动系统

### 第一节 制动片检查

制动片厚度小于 3mm 时应更换。

### 第二节 制动液更换

制动液每两年更换一次，型号为 DOT4。

## 第六章 高压系统

高压维修前必须断电。
"""


def test_heading_paths_exclude_h1_document_title():
    """H1 是文档标题，与前缀里的「文档：xxx」重复，不进章节路径。"""
    paths = [p for _, p in _heading_paths(MD)]
    assert all(not p.startswith("车型维修手册") for p in paths)
    assert "第五章 制动系统 > 第一节 制动片检查" in paths


def test_chunk_section_attribution():
    chunks = [
        _Chunk("制动片厚度小于 3mm 时应更换。"),
        _Chunk("制动液每两年更换一次，型号为 DOT4。"),
        _Chunk("高压维修前必须断电。"),
    ]
    sections = _assign_chunk_sections(chunks, MD)
    assert sections[0] == "第五章 制动系统 > 第一节 制动片检查"
    assert sections[1] == "第五章 制动系统 > 第二节 制动液更换"
    assert sections[2] == "第六章 高压系统"


def test_section_inherits_when_unlocatable():
    """定位不到时继承上一块的章节——比留空更接近真相，且不会张冠李戴。"""
    chunks = [_Chunk("制动片厚度小于 3mm 时应更换。"), _Chunk("完全无法定位的文本内容")]
    sections = _assign_chunk_sections(chunks, MD)
    assert sections[1] == sections[0]


def test_no_headings_yields_empty_sections():
    chunks = [_Chunk("纯正文，没有任何标题。")]
    assert _assign_chunk_sections(chunks, "纯正文，没有任何标题。") == [""]


def test_prefix_composition_and_empty_case():
    @dataclass
    class Meta:
        doc_name: str = "维修手册.pdf"
        doc_type: str = "repair_manual"
        model_code: str = "EV160"

    prefix = _build_chunk_prefix(Meta(), "第五章 制动系统")
    assert "维修手册" in prefix and "repair_manual" in prefix
    assert "EV160" in prefix and "第五章 制动系统" in prefix
    assert prefix.endswith("\n")

    # 元数据全空时不产生空壳前缀
    assert _build_chunk_prefix(Meta(doc_name="", doc_type="", model_code=""), "") == ""

    # 无章节时前缀仍可用
    p2 = _build_chunk_prefix(Meta(), "")
    assert "章节" not in p2 and "维修手册" in p2


# ── 低置信拒答 ──────────────────────────────────────────────────

def _patch_threshold(monkeypatch, value: float):
    monkeypatch.setattr(fusion_mod._settings, "RAG_REFUSE_THRESHOLD", value, raising=False)


def test_top_doc_score_prefers_rerank_score():
    assert _top_doc_score([{"rerank_score": 0.9, "score": 0.01}]) == 0.9
    assert _top_doc_score([{"score": 0.02}]) == 0.02
    assert _top_doc_score([]) == -1.0
    assert _top_doc_score(None) == -1.0


def test_refuse_disabled_by_default(monkeypatch):
    """阈值为 0 = 不启用，避免默认行为被改变。"""
    _patch_threshold(monkeypatch, 0.0)
    assert _should_refuse([{"rerank_score": 0.01}], False, False) is False


def test_refuse_when_low_confidence(monkeypatch):
    _patch_threshold(monkeypatch, 0.95)
    assert _should_refuse([{"rerank_score": 0.30}], False, False) is True
    assert _should_refuse([{"rerank_score": 0.99}], False, False) is False


def test_refuse_skipped_when_other_channels_have_evidence(monkeypatch):
    """图谱/问数通道给出内容时说明确有依据，不该被文档分数压掉。"""
    _patch_threshold(monkeypatch, 0.95)
    low = [{"rerank_score": 0.10}]
    assert _should_refuse(low, has_graph=True, has_sql=False) is False
    assert _should_refuse(low, has_graph=False, has_sql=True) is False
    assert _should_refuse(low, has_graph=False, has_sql=False) is True


def test_refuse_skipped_without_doc_hits(monkeypatch):
    """没有文档命中是「空结果」的范畴，不是「低置信」。"""
    _patch_threshold(monkeypatch, 0.95)
    assert _should_refuse([], False, False) is False
    assert _should_refuse(None, False, False) is False
