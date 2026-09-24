# ============================================================
# 检索质量指标 — HitRate@K / Recall@K / MRR（doc 级 + 页级）
#
# 与 RAG Triad（LLM-as-Judge）互补：纯本地集合运算，零 LLM 成本、
# 可复现，用于检索调参（chunking / embedding / reranker / 融合）的回归验证。
#
# ★ 两种粒度，页级优先：
#   - doc 级（relevant_doc_ids）：命中任一份文档即算对。标注便宜，
#     但分辨力差——同一份手册命中任意页都算对，实测 hit_rate@5/@10/@20
#     会完全相同（三个 k 一模一样 = 指标不含排序信息）。
#   - 页级（relevant_pages，见 scripts/build_eval_set.py 生成骨架）：
#     命中具体某页才算对，能反映"召回得准不准"。页级标签自动派生 doc 级，
#     所以标了页级就不必再填 doc 级。
#
# 标注集列（更多列可选，缺失按空处理）：
#   question, relevant_doc_ids, relevant_pages, answer_type, modality,
#   expect_no_answer, note
#
# 用法：
#   dataset = load_eval_dataset("scripts/eval_dataset.csv")
#   results = [{"retrieved": extract_doc_ids(hits),
#               "retrieved_pages": extract_hit_refs(hits),
#               "relevant": item["relevant"],
#               "relevant_pages": item["relevant_pages"]} for ...]
#   summary = evaluate_retrieval(results, ks=(5, 10, 20))
# ============================================================

from __future__ import annotations

import csv

# 页级引用格式：`文档名.pdf:p12`。用 `:p` 而不是裸 `:`，
# 因为 doc_name 里可能含冒号（如 Windows 风格路径）。
PAGE_SEP = ":p"


def build_page_ref(doc_name: str, page_number) -> str:
    """构造页级引用。页码缺失或为 0（未知）时返回空串。

    ★ 0 是"页码未知"的哨兵，不是第 0 页：入库时若解析器没给出页码锚点
    （LlamaIndex 兜底路径），page_number 一律为 0。把 0 当作真实页码会
    造出一堆 `xxx.pdf:p0` 的假标签，让页级指标失去意义。
    """
    if not doc_name or page_number in ("", None, 0, "0"):
        return ""
    return f"{doc_name}{PAGE_SEP}{page_number}"


def parse_page_ref(ref: str) -> tuple[str, str] | None:
    """拆解页级引用为 (doc_name, page)。格式不合法返回 None。"""
    ref = (ref or "").strip()
    if PAGE_SEP not in ref:
        return None
    doc_name, _, page = ref.rpartition(PAGE_SEP)
    if not doc_name or not page:
        return None
    return doc_name, page


def load_eval_dataset(path: str) -> list[dict]:
    """读取标注数据集 CSV。

    - relevant_pages 存在时以页级为准，并自动派生 doc 级 relevant；
    - relevant_doc_ids 与 relevant_pages 都为空 → labeled=False，评测时跳过；
    - expect_no_answer=1 的题目是「拒答题」：知识库里确实没有答案，
      用于标定"该拒答/该扩大检索"的分数阈值（见 evaluate_score_separation），
      不参与 hit_rate/recall。
    """
    items = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            question = (row.get("question") or "").strip()
            if not question:
                continue

            raw_docs = (row.get("relevant_doc_ids") or "").strip()
            relevant = [d.strip() for d in raw_docs.split(";") if d.strip()]

            raw_pages = (row.get("relevant_pages") or "").strip()
            relevant_pages = [p.strip() for p in raw_pages.split(";") if p.strip()]

            # 页级标签自动派生 doc 级：标了页就不必再填 doc（避免两处不一致）
            if relevant_pages:
                derived = [parse_page_ref(p) for p in relevant_pages]
                doc_from_pages = [d for d in (x[0] if x else None for x in derived) if d]
                if not relevant:
                    relevant = list(dict.fromkeys(doc_from_pages))

            expect_no_answer = (row.get("expect_no_answer") or "").strip() in ("1", "true", "True", "是")

            items.append({
                "question": question,
                "relevant": relevant,
                "relevant_pages": relevant_pages,
                "granularity": "page" if relevant_pages else ("doc" if relevant else "none"),
                # 拒答题不参与召回指标（它没有 ground truth 文档）
                "labeled": bool(relevant or relevant_pages or expect_no_answer),
                "expect_no_answer": expect_no_answer,
                "answer_type": (row.get("answer_type") or "").strip(),
                "modality": (row.get("modality") or "").strip(),
                "note": (row.get("note") or "").strip(),
            })
    return items


def extract_doc_ids(hits: list[dict]) -> list[str]:
    """从检索 hit（search_docs_raw 返回结构）按排名顺序提取 doc_name"""
    return [h["doc_name"] for h in hits if h.get("doc_name")]


def extract_hit_refs(hits: list[dict]) -> list[str]:
    """按排名顺序提取页级引用（doc_name:pN）。

    去重键是「文档+页」，因此同一页的多个相邻块（overlap 造成的重复证据）
    会折叠成一条——这正是页级指标相对 chunk 级指标更抗灌水的原因。
    """
    return [ref for ref in (
        build_page_ref(h.get("doc_name"), h.get("page_number")) for h in hits
    ) if ref]


def extract_top_score(hits: list[dict]) -> float | None:
    """取 top-1 分数（有 rerank_score 用 rerank_score，否则用融合分）。

    用于标定"该拒答/该扩大检索"的阈值：知识库里没有答案的问题，
    top-1 分数的分布应当明显低于有答案的问题。
    """
    if not hits:
        return None
    top = hits[0]
    score = top.get("rerank_score")
    if score is None:
        score = top.get("score")
    try:
        return float(score) if score is not None else None
    except (TypeError, ValueError):
        return None



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
    results: list,
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict:
    """聚合一批检索结果。

    results 每项可以是：
      - 2 元组 (retrieved_doc_ids, relevant_doc_ids)  —— 旧格式，保持兼容
      - dict {"retrieved", "relevant", "retrieved_pages", "relevant_pages"}

    返回 doc 级指标（hit_rate@k / recall@k / mrr），有页级标注时
    额外返回 page_hit_rate@k / page_recall@k / page_mrr。
    比率均为题均；页级指标只在该子集上平均（分母是标了页级的题数）。
    """
    records = [_normalize_record(r) for r in results]
    n = len(records)
    if n == 0:
        return {"count": 0}

    # count = 参与评测的全部题（含拒答题）；doc_labeled/page_labeled = 实际进
    # 各项指标分母的题数。两者分开报，避免"92 题"这种数字被误读成指标样本量。
    summary: dict = {"count": n}

    doc_pairs = [(r["retrieved"], r["relevant"]) for r in records if r["relevant"]]
    if doc_pairs:
        m = len(doc_pairs)
        summary["doc_labeled"] = m
        for k in ks:
            summary[f"hit_rate@{k}"] = sum(hit_rate_at_k(r, rel, k) for r, rel in doc_pairs) / m
            summary[f"recall@{k}"] = sum(recall_at_k(r, rel, k) for r, rel in doc_pairs) / m
        summary["mrr"] = sum(mrr(r, rel) for r, rel in doc_pairs) / m

    page_pairs = [(r["retrieved_pages"], r["relevant_pages"])
                  for r in records if r["relevant_pages"]]
    if page_pairs:
        m = len(page_pairs)
        for k in ks:
            summary[f"page_hit_rate@{k}"] = sum(hit_rate_at_k(r, rel, k) for r, rel in page_pairs) / m
            summary[f"page_recall@{k}"] = sum(recall_at_k(r, rel, k) for r, rel in page_pairs) / m
        summary["page_mrr"] = sum(mrr(r, rel) for r, rel in page_pairs) / m
        summary["page_labeled"] = m

    return summary


def _normalize_record(rec) -> dict:
    """统一结果格式：兼容 2 元组与 dict，缺省字段补空。"""
    if isinstance(rec, dict):
        return {
            "retrieved": list(rec.get("retrieved") or []),
            "relevant": list(rec.get("relevant") or []),
            "retrieved_pages": list(rec.get("retrieved_pages") or []),
            "relevant_pages": list(rec.get("relevant_pages") or []),
        }
    retrieved, relevant = rec[0], rec[1]
    return {
        "retrieved": list(retrieved or []),
        "relevant": list(relevant or []),
        "retrieved_pages": [],
        "relevant_pages": [],
    }


def evaluate_score_separation(
    answerable_scores: list[float],
    unanswerable_scores: list[float],
) -> dict:
    """标定「该拒答 / 该扩大检索」的分数阈值。

    对每个候选阈值 τ 计算：把 top-1 分数 < τ 判为"无答案"时的
    正确率（有答案的未被误拒）与召回（无答案的确实被识别）。
    返回最优 τ 与该阈值下的表现，以及两组的分数统计。

    ★ 用途：
      - 幻觉治理的"宁可不答"阈值：分数低说明库里没有，直接拒答；
      - Query 改写按需触发的门控：分数低才值得多花几路检索。
    两处都需要在**自己的**分数尺度上标定——不能照搬别家的 0.65。
    """
    if not answerable_scores or not unanswerable_scores:
        return {"calibrated": False, "reason": "两组样本都需要至少 1 条"}

    candidates = sorted(set(answerable_scores) | set(unanswerable_scores))
    best = {"threshold": None, "balanced_accuracy": -1.0,
            "answerable_kept": 0.0, "unanswerable_rejected": 0.0}

    for tau in candidates:
        kept = sum(1 for s in answerable_scores if s >= tau) / len(answerable_scores)
        rejected = sum(1 for s in unanswerable_scores if s < tau) / len(unanswerable_scores)
        balanced = (kept + rejected) / 2
        if balanced > best["balanced_accuracy"]:
            best = {"threshold": tau, "balanced_accuracy": balanced,
                    "answerable_kept": kept, "unanswerable_rejected": rejected}

    def _stat(xs: list[float]) -> dict:
        s = sorted(xs)
        mid = len(s) // 2
        median = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
        return {"n": len(s), "min": s[0], "median": median, "max": s[-1]}

    return {
        "calibrated": True,
        **best,
        "answerable": _stat(answerable_scores),
        "unanswerable": _stat(unanswerable_scores),
    }
