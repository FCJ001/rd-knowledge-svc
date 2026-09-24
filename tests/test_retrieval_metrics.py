# ============================================================
# 检索质量指标单元测试
# 覆盖：HitRate@K / Recall@K / MRR / 多 chunk 去重保序 / 数据集加载
# ============================================================

from src.rag.evaluation.retrieval_metrics import (
    evaluate_retrieval,
    evaluate_score_separation,
    extract_doc_ids,
    hit_rate_at_k,
    load_eval_dataset,
    mrr,
    recall_at_k,
)


def test_hit_rate_at_k():
    retrieved = ["A", "B", "C"]
    assert hit_rate_at_k(retrieved, ["B"], 5) == 1.0
    assert hit_rate_at_k(retrieved, ["D"], 5) == 0.0
    assert hit_rate_at_k(retrieved, ["C"], 3) == 1.0
    assert hit_rate_at_k(retrieved, ["C"], 2) == 0.0  # K 截断后不在 top-K


def test_recall_at_k():
    retrieved = ["A", "B", "C"]
    assert recall_at_k(retrieved, ["A", "C"], 3) == 1.0
    assert recall_at_k(retrieved, ["A", "C", "D"], 3) == 2 / 3
    assert recall_at_k(retrieved, ["A"], 1) == 1.0
    assert recall_at_k([], ["A"], 5) == 0.0


def test_mrr():
    assert mrr(["X", "Y", "Z"], ["Z"]) == 1 / 3
    assert mrr(["X", "Y"], ["Z"]) == 0.0
    assert mrr(["Z"], ["Z", "Q"]) == 1.0


def test_empty_relevant_is_zero():
    # 未标注不该出现在评测里，但指标侧兜底返回 0 而非崩溃
    assert hit_rate_at_k(["A"], [], 5) == 0.0
    assert recall_at_k(["A"], [], 5) == 0.0
    assert mrr(["A"], []) == 0.0


def test_dedupe_keeps_first_rank():
    # 同一文档命中多个 chunk：按首次出现排名（去重后 C 升到第 3 位）
    retrieved = ["A", "B", "A", "C"]
    assert mrr(retrieved, ["C"]) == 1 / 3
    assert recall_at_k(retrieved, ["A", "C"], 3) == 1.0


def test_extract_doc_ids():
    hits = [{"doc_name": "A.pdf"}, {"text": "无 doc_name 的 hit"}, {"doc_name": "B.pdf"}]
    assert extract_doc_ids(hits) == ["A.pdf", "B.pdf"]


def test_evaluate_retrieval_aggregates():
    results = [
        (["A"], ["A"]),
        (["X"], ["A"]),
    ]
    s = evaluate_retrieval(results, ks=(1,))
    assert s["count"] == 2
    assert s["hit_rate@1"] == 0.5
    assert s["recall@1"] == 0.5
    assert s["mrr"] == 0.5


def test_evaluate_retrieval_empty():
    assert evaluate_retrieval([], ks=(5,)) == {"count": 0}


def test_load_eval_dataset(tmp_path):
    p = tmp_path / "ds.csv"
    p.write_text(
        "question,relevant_doc_ids,note\n"
        "问题一？,A.pdf;B.pdf,已标注\n"
        "问题二？,,待标注\n",
        encoding="utf-8",
    )
    items = load_eval_dataset(str(p))
    assert len(items) == 2
    assert items[0]["relevant"] == ["A.pdf", "B.pdf"]
    assert items[0]["labeled"] is True
    assert items[1]["relevant"] == []
    assert items[1]["labeled"] is False


# ── 页级标签与拒答题（2026-09 新增：doc 级标签指标退化的修复）────────────

def test_page_level_labels_are_derived_and_measured(tmp_path):
    """页级标签自动派生 doc 级，并额外产出页级指标。

    背景：doc 级标签下 hit_rate@5/@10/@20 会完全相同（命中某文档后
    取多深都一样），指标不含排序信息。页级标签才有分辨力。
    """
    p = tmp_path / "ds.csv"
    p.write_text(
        "question,relevant_doc_ids,relevant_pages,note\n"
        "扭矩多少？,,手册.pdf:p120,页级标签\n",
        encoding="utf-8",
    )
    items = load_eval_dataset(str(p))
    assert items[0]["relevant_pages"] == ["手册.pdf:p120"]
    assert items[0]["relevant"] == ["手册.pdf"]      # 自动派生
    assert items[0]["granularity"] == "page"

    summary = evaluate_retrieval([
        {"retrieved": ["手册.pdf"], "relevant": items[0]["relevant"],
         "retrieved_pages": ["手册.pdf:p120", "手册.pdf:p121"],
         "relevant_pages": items[0]["relevant_pages"]},
    ], ks=(5, 10, 20))
    assert summary["page_hit_rate@5"] == 1.0
    assert summary["page_recall@5"] == 1.0
    assert summary["page_labeled"] == 1


def test_page_metrics_distinguish_ranks_where_doc_metrics_cannot(tmp_path):
    """页级指标能区分"命中同一文档的不同页"，doc 级不能——这正是加它的理由。"""
    records = [
        # 命中了相关页
        {"retrieved": ["手册.pdf"], "relevant": ["手册.pdf"],
         "retrieved_pages": ["手册.pdf:p120"], "relevant_pages": ["手册.pdf:p120"]},
        # 只命中同一文档的无关页：doc 级算对，页级算错
        {"retrieved": ["手册.pdf"], "relevant": ["手册.pdf"],
         "retrieved_pages": ["手册.pdf:p9"], "relevant_pages": ["手册.pdf:p55"]},
    ]
    summary = evaluate_retrieval(records, ks=(5,))
    assert summary["hit_rate@5"] == 1.0        # doc 级：全中，看不出问题
    assert summary["page_hit_rate@5"] == 0.5   # 页级：暴露了一半是蒙的


def test_expect_no_answer_marked_labeled_but_excluded_from_recall(tmp_path):
    """拒答题参与标注统计但不进召回指标（它没有 ground truth 文档）。"""
    p = tmp_path / "ds.csv"
    p.write_text(
        "question,relevant_doc_ids,relevant_pages,expect_no_answer,note\n"
        "库里有没有2030款手册？,,,1,拒答题\n",
        encoding="utf-8",
    )
    items = load_eval_dataset(str(p))
    assert items[0]["expect_no_answer"] is True
    assert items[0]["labeled"] is True
    assert items[0]["granularity"] == "none"


def test_old_two_tuple_format_still_works():
    """旧脚本传 2 元组，不能因为新增页级能力而破坏兼容。"""
    summary = evaluate_retrieval([(["a.pdf"], ["a.pdf"])], ks=(5,))
    assert summary["hit_rate@5"] == 1.0
    assert "page_hit_rate@5" not in summary   # 没有页级数据就不产出页级指标


def test_score_separation_calibrates_threshold():
    """拒答题的 top-1 分数应明显低于有答案的题，据此标定"该拒答"阈值。"""
    result = evaluate_score_separation(
        answerable_scores=[0.9, 0.85, 0.8, 0.75],
        unanswerable_scores=[0.3, 0.25, 0.2, 0.15],
    )
    assert result["calibrated"] is True
    assert 0.3 < result["threshold"] <= 0.75
    assert result["balanced_accuracy"] == 1.0     # 两组完全可分
    assert result["answerable"]["n"] == 4
    assert result["unanswerable"]["median"] < result["answerable"]["median"]


def test_score_separation_reports_uncalibrated_without_both_groups():
    """只有一组样本时不能给阈值——单边数据标不出分界线。"""
    assert evaluate_score_separation([0.9], [])["calibrated"] is False
    assert evaluate_score_separation([], [0.1])["calibrated"] is False
