#!/usr/bin/env python3
# ============================================================
# 答案质量对照（回答"几条证据最好"这个检索指标答不了的问题）
#
#   python scripts/eval_answer_quality.py --sample 18
#   python scripts/eval_answer_quality.py --sample 18 \
#       --config "current:" --config "min3:--rerank-min-topk=3,--gap-ratio=0.4"
#
# 为什么需要它：
#   页级召回率会随"喂给模型的证据条数"单调上升——多给几条必然召回更多页，
#   这是机械的，不能用来判断"几条证据让答案更好"。唯一能回答的是答案侧指标：
#   有据性（答案是否被上下文支撑）与答案相关性。
#
# 指标口径与 TruLens RAG Triad 一致（复用 async_tracker 的裁判 prompt）。
# 刻意避开 eval/run_triad_direct.py：那套题面向另一份语料（汉EV/唐DM-i），
# 在本知识库上全是"无答案题"，测出来没有意义。
# ============================================================

import argparse
import asyncio
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from src.core.config import get_settings
from src.infra.milvus_client import get_milvus_client
from src.knowledge.doc_rag import format_doc_context, search_docs_with_stages
from src.knowledge.prompts import FUSION_PROMPT
from src.rag.evaluation.async_tracker import (
    ANSWER_RELEVANCE_PROMPT,
    GROUNDEDNESS_PROMPT,
)
from src.rag.evaluation.retrieval_metrics import load_eval_dataset

# 取分数：裁判要求"只输出分数"，但实际常带解释，用第一个浮点数兜底
_SCORE_RE = re.compile(r"(\d\.\d+|\d)")


async def _judge(llm, prompt: str) -> float | None:
    try:
        resp = await llm.ainvoke(prompt)
        m = _SCORE_RE.search(resp.content or "")
        if not m:
            return None
        v = float(m.group(1))
        return v if 0.0 <= v <= 1.0 else None
    except Exception:
        return None


async def run_config(name: str, items: list[dict], overrides: dict) -> dict:
    s = get_settings()
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

    grounded, relevant, counts = [], [], []
    print(f"\n{'='*66}\n配置 {name}\n{'='*66}")
    for i, item in enumerate(items, 1):
        _, hits = await search_docs_with_stages(
            item["question"], emb, mc, role="engineer", llm=llm, use_hyde=False)
        counts.append(len(hits))
        context = format_doc_context(hits)
        prompt = FUSION_PROMPT.format(question=item["question"], sources=context, role="engineer")
        try:
            ans = (await llm.ainvoke(prompt)).content or ""
        except Exception as e:
            print(f"  [{i:2d}/{len(items)}] 生成失败: {e}")
            continue

        g = await _judge(llm, GROUNDEDNESS_PROMPT.format(
            question=item["question"], contexts=context[:3000], answer=ans[:2000]))
        r = await _judge(llm, ANSWER_RELEVANCE_PROMPT.format(
            question=item["question"], answer=ans[:2000]))
        if g is not None:
            grounded.append(g)
        if r is not None:
            relevant.append(r)
        print(f"  [{i:2d}/{len(items)}] 证据={len(hits):2d} 有据性={g} 相关性={r} "
              f"{item['question'][:30]}")

    return {
        "config": name,
        "n": len(items),
        "evidence_median": statistics.median(counts) if counts else 0,
        "groundedness": round(statistics.mean(grounded), 4) if grounded else None,
        "answer_relevance": round(statistics.mean(relevant), 4) if relevant else None,
    }


def _parse_overrides(spec: str) -> dict:
    """`--rerank-min-topk=3,--gap-ratio=0.4` → {"RERANK_MIN_TOPK": 3, ...}"""
    mapping = {
        "rerank-min-topk": "RERANK_MIN_TOPK",
        "gap-abs": "RERANK_GAP_ABS",
        "gap-ratio": "RERANK_GAP_RATIO",
        "top-k": "RAG_TOP_K",
        "rerank-top-k": "RAG_RERANK_TOP_K",
        "dynamic-topk": "RAG_DYNAMIC_TOPK",
    }
    out: dict = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        k, _, v = part.partition("=")
        # 允许写成 --rerank-min-topk=5（带 CLI 风格前缀）
        key = mapping.get(k.strip().lstrip("-"))
        if not key:
            raise ValueError(f"未知参数: {k}")
        if key == "RAG_DYNAMIC_TOPK":
            out[key] = v.strip().lower() in ("1", "on", "true", "yes")
        else:
            out[key] = float(v) if "." in v else int(v)
    return out


async def main(dataset: str, sample: int, configs: list[tuple[str, dict]], out: str | None) -> int:
    items = [x for x in load_eval_dataset(dataset) if x["labeled"] and not x["expect_no_answer"]]
    if not items:
        print("[失败] 没有可回答题")
        return 1
    # 等距抽样：覆盖全文档、避免只看开头
    step = max(1, len(items) // sample)
    picked = items[::step][:sample]
    print(f"从 {len(items)} 道可回答题中等距抽取 {len(picked)} 道做答案质量对照")

    results = []
    for name, ov in configs:
        results.append(await run_config(name, picked, ov))

    print(f"\n{'='*66}\n答案质量对照汇总\n{'='*66}")
    print(f"{'配置':<16}{'证据中位':>9}{'有据性':>10}{'答案相关性':>12}")
    for r in results:
        g = r["groundedness"]
        a = r["answer_relevance"]
        print(f"{r['config']:<16}{r['evidence_median']:>9.0f}"
              f"{(g if g is not None else float('nan')):>10.4f}"
              f"{(a if a is not None else float('nan')):>12.4f}")
    if out:
        Path(out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写出 {out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="答案质量对照")
    ap.add_argument("--dataset", default="scripts/eval_dataset_draft.csv")
    ap.add_argument("--sample", type=int, default=18)
    ap.add_argument("--config", action="append", default=None,
                    help="`名称:参数=值,参数=值`，可重复；不给则用 current/min3/min5 三组")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.config:
        cfgs = []
        for c in args.config:
            name, _, spec = c.partition(":")
            cfgs.append((name or "unnamed", _parse_overrides(spec)))
    else:
        cfgs = [
            ("current", {}),
            ("min3", {"RERANK_MIN_TOPK": 3, "RERANK_GAP_RATIO": 0.4}),
            ("min5", {"RERANK_MIN_TOPK": 5, "RERANK_GAP_RATIO": 0.5}),
        ]
    raise SystemExit(asyncio.run(main(args.dataset, args.sample, cfgs, args.out)))
