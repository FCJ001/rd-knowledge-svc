# ============================================================
# _cliff_topk 动态 TopK 单元测试
# 覆盖：空输入 / 断崖(绝对gap) / 断崖(相对gap) / 无断崖 / 边界
# ============================================================

from src.knowledge.reranker import _cliff_topk


def test_empty_scores():
    assert _cliff_topk([]) == 0


def test_no_cliff_takes_max():
    # 分数平滑下降，无断崖 → 取满 max_topk
    scores = [0.9, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
    assert _cliff_topk(scores) == 7


def test_absolute_gap_cliff():
    # index=2 处 0.80 -> 0.20，绝对差 0.6 >= 0.5 → 截断于 index=2，保留 3 条
    scores = [0.95, 0.90, 0.80, 0.20, 0.18, 0.15]
    assert _cliff_topk(scores) == 3


def test_relative_gap_cliff():
    # index=1 处 0.50 -> 0.30，gap=0.2，rel=0.2/0.5=0.4 >= 0.25 → 保留 2 条
    scores = [0.60, 0.50, 0.30, 0.28]
    assert _cliff_topk(scores) == 2


def test_min_topk_floor():
    # 断崖出现在 index=0，但下限 min_topk=1，至少保留 1 条
    scores = [0.90, 0.10]
    assert _cliff_topk(scores) == 1


def test_cliff_in_early_range_respects_min():
    # min_topk=2 时，即使断崖在 index=0，也要从 index=1 开始扫描
    scores = [0.90, 0.89, 0.10, 0.09]
    assert _cliff_topk(scores, min_topk=2) == 2


def test_max_topk_caps_length():
    scores = [0.99, 0.98, 0.97, 0.96, 0.95]
    assert _cliff_topk(scores, max_topk=3) == 3


def test_scores_shorter_than_min():
    # 分数个数 < min_topk 时退化为返回 min_topk 上限（截到 len）
    scores = [0.9, 0.8]
    assert _cliff_topk(scores, min_topk=5) == 2
