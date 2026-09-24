#!/usr/bin/env python3
# ============================================================
# 检索诊断报告（比 eval_gate 多算"拒答信号是否可用"）
#
#   python scripts/eval_report.py --dataset scripts/eval_dataset_draft.csv
#
# 产出三部分：
#   1. doc 级 / 页级检索指标（HitRate@K / Recall@K / MRR）
#   2. 拒答题分数分离标定：有答案题 vs 拒答题的 top-1 分数是否可分
#   3. 失败清单：哪些题没召回对，便于逐条归因
#
# ★ 第 2 项是本脚本存在的主要理由：如果重排分数对"库里有没有答案"
#   没有区分度，那么"分数低就拒答"这条幻觉治理假设就是错的，
#   必须换信号（BM25/dense 的绝对相似度、或显式的可答性判别）。
# ============================================================

import argparse
import asyncio
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from src.core.config import get_settings
from src.infra.milvus_client import get_milvus_client
from src.knowledge.doc_rag import search_docs_with_stages
from src.rag.evaluation.retrieval_metrics import (
    evaluate_retrieval,
    evaluate_score_separation,
    extract_doc_ids,
    extract_hit_refs,
    extract_top_score,
    load_eval_dataset,
)

# 含 30：RAG_TOP_K 提高到 30 时需要看 raw@30（召回天花板随深度上升）
KS = (5, 10, 20, 30)


async def main(dataset_path: str, out_json: str | None, overrides: dict) -> int:
    s = get_settings()
    # 参数覆盖：直接改 reranker 模块持有的 settings 单例，便于一次跑多组配置
    if overrides:
        from src.knowledge import reranker as _rr
        for k, v in overrides.items():
            if v is not None:
                setattr(_rr.settings, k, v)
                setattr(s, k, v)
    emb = DashScopeEmbeddings(model=s.EMBEDDING_MODEL, dashscope_api_key=s.DASHSCOPE_API_KEY)
    llm = ChatOpenAI(model=s.CHAT_MODEL, api_key=s.chat_api_key,
                     base_url=s.BASE_URL_CHAT, temperature=0)
    mc = get_milvus_client()

    items = [x for x in load_eval_dataset(dataset_path) if x["labeled"]]
    if not items:
        print("[失败] 数据集中没有已标注题目")
        return 1
    print(f"评测 {len(items)} 题（含拒答题 {sum(1 for x in items if x['expect_no_answer'])}）")
    print(f"rerank_provider={s.RERANK_PROVIDER}  top_k={s.RAG_TOP_K}  "
          f"rerank_top_k={s.RAG_RERANK_TOP_K}\n")

    records, raw_records = [], []
    answerable_scores, unanswerable_scores = [], []
    failures = []
    raw_counts, final_counts = [], []
    for i, item in enumerate(items, 1):
        raw_hits, hits = await search_docs_with_stages(
            item["question"], emb, mc, role="engineer", llm=llm, use_hyde=False)
        docs = extract_doc_ids(hits)
        pages = extract_hit_refs(hits)
        raw_pages = extract_hit_refs(raw_hits)
        top = extract_top_score(hits)
        raw_counts.append(len(raw_hits))
        final_counts.append(len(hits))

        if item["expect_no_answer"]:
            if top is not None:
                unanswerable_scores.append(top)
            tag = "拒答"
        else:
            records.append({
                "retrieved": docs, "relevant": item["relevant"],
                "retrieved_pages": pages, "relevant_pages": item["relevant_pages"],
            })
            raw_records.append({
                "retrieved": extract_doc_ids(raw_hits), "relevant": item["relevant"],
                "retrieved_pages": raw_pages, "relevant_pages": item["relevant_pages"],
            })
            if top is not None:
                answerable_scores.append(top)
            ok = bool(set(docs) & set(item["relevant"]))
            tag = "命中" if ok else "未中"
            if not ok:
                failures.append({
                    "question": item["question"],
                    "expected": item["relevant"],
                    "got": docs[:3],
                    "top_score": top,
                })
        print(f"  [{i:3d}/{len(items)}] {tag} top={top if top is None else round(top, 3)}  "
              f"{item['question'][:38]}")

    print("\n" + "=" * 64)
    print("① 检索指标")
    print("=" * 64)
    summary = evaluate_retrieval(records, ks=KS)
    for k in KS:
        print(f"  HitRate@{k}={summary.get(f'hit_rate@{k}', 0):.2%}  "
              f"Recall@{k}={summary.get(f'recall@{k}', 0):.2%}")
    print(f"  MRR={summary.get('mrr', 0):.3f}")
    print(f"  （指标分母 = {summary.get('doc_labeled', 0)} 道可回答题；"
          f"另有 {summary.get('count', 0) - summary.get('doc_labeled', 0)} 道拒答题不进召回指标）")

    raw_summary = evaluate_retrieval(raw_records, ks=KS) if raw_records else {}
    if summary.get("page_labeled") and raw_summary.get("page_labeled"):
        print(f"\n  ── 页级两阶段对照（{summary['page_labeled']} 道有页级标签）──")
        print(f"  {'':22s}" + "".join(f"{'@'+str(k):>9s}" for k in KS))
        for label, sm, note in (
            ("raw（融合召回后）", raw_summary, "召回本身的上限"),
            ("final（截断后）", summary, "真正喂给生成的证据"),
        ):
            print(f"  {label:20s}" + "".join(
                f"{sm.get(f'page_recall@{k}', 0):>8.1%} " for k in KS) + f" ← {note}")
        gap = raw_summary.get("page_recall@20", 0) - summary.get("page_recall@20", 0)
        print(f"\n  截断吃掉的页级召回（raw@20 − final@20）= {gap:+.1%}")
        if gap > 0.05:
            print("  ★ 差值明显 → 瓶颈在**断崖截断**：召回拿得到但被切掉了，"
                  "调 RERANK_MIN_TOPK / RERANK_GAP_RATIO 而非 RAG_TOP_K")
        elif raw_summary.get("page_recall@20", 0) < 0.95:
            print("  ★ 差值很小但 raw 本身就不高 → 瓶颈在**召回**："
                  "调 RAG_TOP_K / 混合检索，截断不是问题")
        else:
            print("  ★ 两阶段都好 → 当前配置无瓶颈")

    print("\n  ── 证据条数分布 ──")
    for label, cnt in (("raw 融合召回", raw_counts), ("final 截断后", final_counts)):
        c = Counter(cnt)
        ones = c.get(1, 0)
        print(f"  {label:14s} 中位={statistics.median(cnt):.0f}  "
              f"仅 1 条={ones}/{len(cnt)}（{ones/len(cnt):.0%}）  "
              f"分布={dict(sorted(c.items()))}")

    print("\n" + "=" * 64)
    print("② 拒答信号是否可用（决定性判据）")
    print("=" * 64)
    sep = evaluate_score_separation(answerable_scores, unanswerable_scores)
    if not sep.get("calibrated"):
        print(f"  无法标定：{sep.get('reason')}")
    else:
        a, u = sep["answerable"], sep["unanswerable"]
        print(f"  可回答题 top-1 分数：n={a['n']} 中位={a['median']:.3f} "
              f"范围=[{a['min']:.3f}, {a['max']:.3f}]")
        print(f"  拒答题   top-1 分数：n={u['n']} 中位={u['median']:.3f} "
              f"范围=[{u['min']:.3f}, {u['max']:.3f}]")
        print(f"  最优阈值 τ={sep['threshold']:.3f}  平衡准确率={sep['balanced_accuracy']:.2f}")
        print(f"    → 该阈值下可回答题保留率 {sep['answerable_kept']:.0%}、"
              f"拒答题识别率 {sep['unanswerable_rejected']:.0%}")
        if sep["balanced_accuracy"] < 0.7:
            print("\n  ⚠️ 平衡准确率 <0.7：该分数**无法**区分「库里有没有答案」。")
            print("     结论：不能拿它做'分数低就拒答'的阈值，必须换信号。")
            print("     候选：BM25 绝对相似度 / dense 余弦相似度 / 显式的可答性判别。")
        else:
            print("\n  ✓ 该分数具备区分度，可用于拒答与按需改写门控。")

    print("\n" + "=" * 64)
    print(f"③ 失败清单（{len(failures)} 道未召回）")
    print("=" * 64)
    for f in failures:
        print(f"  ✗ {f['question'][:46]}")
        print(f"      期望={f['expected']}  实际前3={f['got']}")

    if out_json:
        Path(out_json).write_text(json.dumps({
            "dataset": dataset_path,
            "metrics": summary,
            "separation": sep,
            "failures": failures,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写出: {out_json}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="检索诊断报告")
    ap.add_argument("--dataset", default="scripts/eval_dataset_draft.csv")
    ap.add_argument("--out", default=None, help="另存 JSON 报告")
    ap.add_argument("--top-k", type=int, default=None, help="覆盖 RAG_TOP_K")
    ap.add_argument("--rerank-top-k", type=int, default=None, help="覆盖 RAG_RERANK_TOP_K")
    ap.add_argument("--rerank-min-topk", type=int, default=None, help="覆盖断崖下限")
    ap.add_argument("--gap-abs", type=float, default=None, help="覆盖断崖绝对阈值")
    ap.add_argument("--gap-ratio", type=float, default=None, help="覆盖断崖相对阈值")
    ap.add_argument("--rerank-max-candidates", type=int, default=None,
                    help="覆盖重排候选上限（须 ≥ top_k，否则多出的候选 LLM 看不到）")
    ap.add_argument("--dynamic-topk", choices=["on", "off"], default=None,
                    help="覆盖 RAG_DYNAMIC_TOPK（off = 固定取 rerank_top_k 条）")
    args = ap.parse_args()
    overrides = {
        "RAG_TOP_K": args.top_k,
        "RAG_RERANK_TOP_K": args.rerank_top_k,
        "RERANK_MIN_TOPK": args.rerank_min_topk,
        "RERANK_GAP_ABS": args.gap_abs,
        "RERANK_GAP_RATIO": args.gap_ratio,
        "RERANK_MAX_CANDIDATES": args.rerank_max_candidates,
        "RAG_DYNAMIC_TOPK": (None if args.dynamic_topk is None
                             else args.dynamic_topk == "on"),
    }
    raise SystemExit(asyncio.run(main(args.dataset, args.out, overrides)))
