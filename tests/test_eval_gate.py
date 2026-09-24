# ============================================================
# 检索质量回归门禁的逻辑测试
#
# 门禁本身必须"骗不过去"：数据集太小、基线缺失、指标回退
# 都应当被明确判定，而不是悄悄放行。
# ============================================================


from scripts.eval_gate import (
    GATED_METRICS,
    MIN_LABELED_QUESTIONS,
    _check_dataset_health,
    compare_to_baseline,
)


def _dataset(n_labeled: int, n_unlabeled: int = 0) -> list[dict]:
    items = [
        {"question": f"q{i}", "relevant": ["a.pdf"], "labeled": True}
        for i in range(n_labeled)
    ]
    items += [
        {"question": f"u{i}", "relevant": [], "labeled": False}
        for i in range(n_unlabeled)
    ]
    return items


def test_gate_rejects_too_small_dataset():
    """样本不足时门禁必须失败——否则 10 题上的 ±2 题波动会被当成"通过"。"""
    ok, msg = _check_dataset_health(_dataset(MIN_LABELED_QUESTIONS - 1))
    assert not ok
    assert "标注题数" in msg


def test_gate_accepts_sufficient_dataset():
    ok, _ = _check_dataset_health(_dataset(MIN_LABELED_QUESTIONS))
    assert ok


def test_unlabeled_rows_do_not_count():
    """未标注题不计入有效样本。"""
    ok, _ = _check_dataset_health(_dataset(5, n_unlabeled=100))
    assert not ok


def test_no_regression_passes():
    baseline = {"metrics": {"hit_rate@5": 0.70, "mrr": 0.60, "hit_rate@20": 0.80}}
    current = {"hit_rate@5": 0.72, "mrr": 0.61, "hit_rate@20": 0.81}
    assert compare_to_baseline(current, baseline, max_regression=0.05) == []


def test_small_drop_is_within_tolerance():
    """小于阈值的波动不拦截（千级语料上分不出信号）。"""
    baseline = {"metrics": {"hit_rate@5": 0.70, "mrr": 0.60, "hit_rate@20": 0.80}}
    current = {"hit_rate@5": 0.68, "mrr": 0.59, "hit_rate@20": 0.79}
    assert compare_to_baseline(current, baseline, max_regression=0.05) == []


def test_significant_drop_is_flagged():
    baseline = {"metrics": {"hit_rate@5": 0.70, "mrr": 0.60, "hit_rate@20": 0.80}}
    current = {"hit_rate@5": 0.60, "mrr": 0.60, "hit_rate@20": 0.80}
    regressions = compare_to_baseline(current, baseline, max_regression=0.05)
    assert len(regressions) == 1
    assert "hit_rate@5" in regressions[0]


def test_multiple_metric_regressions_all_reported():
    baseline = {"metrics": {"hit_rate@5": 0.70, "mrr": 0.60, "hit_rate@20": 0.80}}
    current = {"hit_rate@5": 0.50, "mrr": 0.40, "hit_rate@20": 0.80}
    regressions = compare_to_baseline(current, baseline, max_regression=0.05)
    assert len(regressions) == 2


def test_improvement_is_never_a_regression():
    baseline = {"metrics": {"hit_rate@5": 0.50, "mrr": 0.40, "hit_rate@20": 0.60}}
    current = {"hit_rate@5": 0.90, "mrr": 0.90, "hit_rate@20": 0.95}
    assert compare_to_baseline(current, baseline, max_regression=0.05) == []


def test_missing_metric_is_skipped_not_flagged():
    """缺指标的基线不应产生假回退（向后兼容旧基线文件）。"""
    baseline = {"metrics": {"hit_rate@5": 0.70}}
    current = {"hit_rate@5": 0.70, "mrr": 0.10, "hit_rate@20": 0.10}
    assert compare_to_baseline(current, baseline, max_regression=0.05) == []


def test_gated_metrics_are_all_higher_is_better():
    """门禁指标当前全是"越大越好"；若将来加入反向指标需显式处理方向。"""
    assert all(v is True for v in GATED_METRICS.values())
