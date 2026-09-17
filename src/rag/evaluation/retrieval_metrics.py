# ============================================================
# 检索质量指标 — HitRate@K / Recall@K / MRR
#
# 与 RAG Triad（LLM-as-Judge）互补：纯本地集合运算，零 LLM 成本、
# 可复现，用于检索调参（chunking / embedding / reranker 权重）的回归验证。
#
# ground truth 口径：doc_name 级标注（不标到 chunk，标注成本低、
# 对"是否找对文档"的决策粒度足够）。标注集见 scripts/eval_dataset.csv，
# 一行一题，多个相关文档用分号分隔。
#
# 用法：
#   dataset = load_eval_dataset("scripts/eval_dataset.csv")
#   results = [(extract_doc_ids(hits), item["relevant"]) ...]
#   summary = evaluate_retrieval(results, ks=(5, 10, 20))
# ============================================================

from __future__ import annotations

import csv


def load_eval_dataset(path: str) -> list[dict]:
    """读取标注数据集 CSV（列：question, relevant_doc_ids, note）。

    relevant_doc_ids 为分号分隔的 doc_name 列表；为空的题目标记
    labeled=False，评测时跳过（不计入指标分母）。
    """
    items = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            question = (row.get("question") or "").strip()
            if not question:
                continue
            raw = (row.get("relevant_doc_ids") or "").strip()
            relevant = [d.strip() for d in raw.split(";") if d.strip()]
            items.append({
                "question": question,
                "relevant": relevant,
                "labeled": bool(relevant),
            })
    return items


def extract_doc_ids(hits: list[dict]) -> list[str]:
    """从检索 hit（search_docs_raw 返回结构）按排名顺序提取 doc_name"""
    return [h["doc_name"] for h in hits if h.get("doc_name")]


def _dedupe(ids: list[str]) -> list[str]:
    """去重保序：同一文档可能命中多个 chunk，doc 级指标按首次出现排名"""
    seen: set[str] = set()
    out = []
    for doc_id in ids:
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def hit_rate_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """top-K 内命中任一相关文档 → 1.0，否则 0.0（按题平均即命中率）"""
    if not relevant:
        return 0.0
    top_k = set(_dedupe(retrieved)[:k])
    return 1.0 if top_k & set(relevant) else 0.0


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """top-K 命中的相关文档数 / 全部相关文档数"""
    if not relevant:
        return 0.0
    top_k = set(_dedupe(retrieved)[:k])
    return len(top_k & set(relevant)) / len(set(relevant))


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    """第一个相关文档的倒数排名；无命中 → 0"""
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(_dedupe(retrieved), 1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    results: list[tuple[list[str], list[str]]],
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict:
    """聚合一批 (retrieved_doc_ids, relevant_doc_ids) 的指标

    返回形如 {"count": 25, "hit_rate@5": 0.72, "recall@10": 0.55,
    "mrr": 0.61, ...}，比率均为题均。
    """
    n = len(results)
    if n == 0:
        return {"count": 0}
    summary: dict = {"count": n}
    for k in ks:
        summary[f"hit_rate@{k}"] = sum(hit_rate_at_k(r, rel, k) for r, rel in results) / n
        summary[f"recall@{k}"] = sum(recall_at_k(r, rel, k) for r, rel in results) / n
    summary["mrr"] = sum(mrr(r, rel) for r, rel in results) / n
    return summary
