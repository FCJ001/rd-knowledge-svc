# ============================================================
# merge_short_chunks 短 chunk 合并单元测试
# 覆盖：长+短合并 / 短+短合并 / 超 max_chars 不合并 / 跨 parent 不合并 /
#       空输入 / chunk_index 重编号 / 输入不被修改
# ============================================================

from src.rag.ingestion.chunkers import Chunk, merge_short_chunks


def _chunk(text: str, parent_index=None) -> Chunk:
    md = {"chunk_index": 0}
    if parent_index is not None:
        md["parent_index"] = parent_index
        md["parent_text"] = f"parent-{parent_index}"
    return Chunk(text=text, metadata=md)


def test_empty_input():
    assert merge_short_chunks([]) == []


def test_long_then_short_merged():
    long_text = "长" * 300
    short_text = "短" * 50  # 300+2+50=352 <= 800
    chunks = [_chunk(long_text), _chunk(short_text)]
    result = merge_short_chunks(chunks)
    assert len(result) == 1
    assert result[0].text == long_text + "\n\n" + short_text
    assert result[0].metadata["chunk_index"] == 0


def test_short_then_short_merged():
    a = _chunk("甲" * 100)
    b = _chunk("乙" * 100)
    result = merge_short_chunks([a, b])
    assert len(result) == 1
    assert result[0].text == "甲" * 100 + "\n\n" + "乙" * 100


def test_combined_exceeding_max_not_merged():
    a = _chunk("长" * 790)
    b = _chunk("短" * 50)  # 790+2+50=842 > 800 → 不合并
    result = merge_short_chunks([a, b], max_chars=800)
    assert len(result) == 2


def test_all_long_not_merged():
    a = _chunk("长" * 300)
    b = _chunk("长" * 300)
    result = merge_short_chunks([a, b], min_chars=200)
    assert len(result) == 2


def test_parent_child_cross_parent_not_merged():
    a = _chunk("短" * 100, parent_index=0)
    b = _chunk("短" * 100, parent_index=1)
    result = merge_short_chunks([a, b])
    assert len(result) == 2


def test_parent_child_same_parent_merged():
    a = _chunk("短" * 100, parent_index=0)
    b = _chunk("短" * 100, parent_index=0)
    result = merge_short_chunks([a, b])
    assert len(result) == 1
    assert result[0].metadata["parent_index"] == 0
    assert result[0].metadata["parent_text"] == "parent-0"


def test_chunk_index_renumbered():
    chunks = [_chunk("短" * 100), _chunk("短" * 100), _chunk("长" * 300)]
    result = merge_short_chunks(chunks)
    # 前两块合并为一块，第三块独立 → 2 块，索引 0,1
    assert [c.metadata["chunk_index"] for c in result] == [0, 1]


def test_input_not_mutated():
    a = _chunk("短" * 100)
    b = _chunk("短" * 100)
    original = [a, b]
    merge_short_chunks(original)
    # 输入不被修改，且两块的 text 长度不变
    assert len(original) == 2
    assert len(original[0].text) == 100
