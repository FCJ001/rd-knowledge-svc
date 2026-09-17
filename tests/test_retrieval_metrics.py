# ============================================================
# 检索质量指标单元测试
# 覆盖：HitRate@K / Recall@K / MRR / 多 chunk 去重保序 / 数据集加载
# ============================================================

from src.rag.evaluation.retrieval_metrics import (
    evaluate_retrieval,
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
