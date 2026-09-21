#!/usr/bin/env python3
# ============================================================
# RAG 检索通路评测脚本（TruLens 2.x 标准生命周期）
#
# 用法:
#   python scripts/run_rag_experiments.py                        # 跑全部通道
#   python scripts/run_rag_experiments.py --channel doc_rag      # 单通道
#   python scripts/run_rag_experiments.py --list                 # 列出题目
#   python scripts/run_rag_experiments.py --dashboard            # 启动 Dashboard
#
#   # 检索指标评测（不跑生成、不调 LLM 裁判，检索调参的快速回归回路）
#   python scripts/run_rag_experiments.py --retrieval-only
#   python scripts/run_rag_experiments.py --retrieval-only --channel doc_rag
#
# 指标（TruLens RAG Triad）:
#   答案相关性 — Question → Answer
#   上下文相关性 — Question → Context
#   有据性 — Context → Answer
#
# 检索指标（HitRate@K / Recall@K / MRR，零 LLM 成本）:
#   标注集 scripts/eval_dataset.csv，一行一题：
#     question,relevant_doc_ids,note
#   relevant_doc_ids 填相关文档的 doc_name，多个用分号分隔；
#   留空的题目视为未标注，评测时跳过。doc_name 可从检索日志
#   （"DocRAG 召回...来源:"）或 Milvus 中取。
#
# ★ 每个通道作为独立 app_version，Dashboard 可多版本对比
# ============================================================

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from src.core.config import get_settings
from src.infra.milvus_client import get_milvus_client
from src.infra.neo4j_client import get_neo4j_driver
from src.rag.evaluation.feedbacks import build_rag_triad_metrics
from src.rag.evaluation.retrieval_metrics import (
    evaluate_retrieval,
    extract_doc_ids,
    load_eval_dataset,
)
from src.rag.evaluation.tracked_rag import TrackedRAG
from src.rag.evaluation.trulens_config import (
    get_llm_provider,
    get_trulens_session,
    launch_dashboard,
)
from trulens.apps.app import TruApp

# ── 评测问题集（ALM 汽车领域）───────────────────────────────────────────

EVAL_QUESTIONS = [
    "汉EV 2024款的电池管理系统有哪些安全保护机制？",
    "OTA升级失败后如何恢复？",
    "ABS模块故障的诊断流程是什么？",
    "2023年后出厂的车型扭矩标准有哪些变更？",
    "网关模块和BCM模块之间的通信协议是什么？",
    "变更CR-2024-00178对底盘控制系统有什么影响？",
    "近3个月S1级别的问题有多少个？闭环率是多少？",
    "高压系统维修的安全注意事项有哪些？",
    "唐DM-i的电机控制器过热故障怎么排查？",
    "软件版本v3.2.1有哪些已知问题和修复方案？",
]

CHANNELS = ["doc_rag", "graph_rag", "fusion"]
EVAL_KS = (5, 10, 20)
DATASET_PATH = Path(__file__).resolve().parent / "eval_dataset.csv"


# ── 依赖构建 ────────────────────────────────────────────────────────────

def build_tracked_rag(channel: str, use_hyde: bool | None = None,
                      top_k: int | None = None,
                      rerank_top_k: int | None = None) -> TrackedRAG:
    """消融参数为 None 时回落到 Settings.RAG_*（见 TrackedRAG.__init__），
    CLI 显式传值时以 CLI 为准——一次只改一个变量做单因子对比。"""
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.chat_api_key,
        base_url=settings.BASE_URL_CHAT,
        temperature=0,
    )
    embedding_model = DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )
    milvus_client = get_milvus_client()
    neo4j_driver = get_neo4j_driver()

    return TrackedRAG(
        channel=channel,
        llm=llm,
        embedding_model=embedding_model,
        milvus_client=milvus_client,
        neo4j_driver=neo4j_driver,
        role="engineer",
        use_hyde=use_hyde,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
    )


# ── 延迟统计 ─────────────────────────────────────────────────────────────

def print_latency_stats(latencies_ms: list[float], label: str) -> None:
    if not latencies_ms:
        return
    ordered = sorted(latencies_ms)
    avg = sum(ordered) / len(ordered)
    p50 = ordered[len(ordered) // 2]
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    print(f"  >>> {label} 延迟: avg={avg:.0f}ms p50={p50:.0f}ms p95={p95:.0f}ms (n={len(ordered)})")


# ── 检索指标评测（HitRate@K / Recall@K / MRR）────────────────────────────

async def run_retrieval_eval(channel: str, dataset: list[dict],
                             use_hyde: bool | None = None, top_k: int | None = None,
                             rerank_top_k: int | None = None):
    labeled = [item for item in dataset if item["labeled"]]
    skipped = len(dataset) - len(labeled)

    print(f"\n{'='*60}")
    print(f"检索指标评测 | 通道: {channel} | 标注题数: {len(labeled)}"
          + (f"（另 {skipped} 题未标注已跳过）" if skipped else ""))
    print(f"{'='*60}")

    if not labeled:
        print(f"  >>> {channel} 无标注题目，跳过。请在 {DATASET_PATH} 填 relevant_doc_ids")
        return None
    if channel == "graph_rag":
        print("  >>> graph_rag 返回图谱记录而非文档，无 doc 级 ground truth，不适用检索指标")
        return None

    tracked = build_tracked_rag(channel, use_hyde=use_hyde, top_k=top_k,
                                rerank_top_k=rerank_top_k)
    results = []
    latencies = []
    for i, item in enumerate(labeled, 1):
        start = time.perf_counter()
        hits = await tracked.retrieve_docs(item["question"])
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        retrieved = extract_doc_ids(hits)
        results.append((retrieved, item["relevant"]))
        hit = "✓" if set(retrieved) & set(item["relevant"]) else "✗"
        print(f"  [{i}/{len(labeled)}] {hit} {item['question'][:40]}... "
              f"→ {latency_ms:.0f}ms, 召回文档 {len(set(retrieved))} 个")

    summary = evaluate_retrieval(results, ks=EVAL_KS)
    summary["latency_ms_avg"] = sum(latencies) / len(latencies) if latencies else None
    print(f"\n  >>> {channel} 检索指标（{summary['count']} 题）:")
    for k in EVAL_KS:
        print(f"      HitRate@{k}={summary[f'hit_rate@{k}']:.2%}  Recall@{k}={summary[f'recall@{k}']:.2%}")
    print(f"      MRR={summary['mrr']:.3f}")
    print_latency_stats(latencies, channel)
    return summary


# ── 结果落盘 & 历史对比（没有落盘就没法做单因子对比）──────────────────────

RESULTS_PATH = Path(__file__).resolve().parent / "rag_eval_results.jsonl"


def _result_key(ablation: dict) -> tuple:
    return (ablation["use_hyde"], ablation["top_k"], ablation["rerank_top_k"])


def _save_retrieval_results(summaries: dict, ablation: dict, dataset_path: str) -> None:
    """把本次检索指标追加到 jsonl。一行一次运行，含全部消融参数。"""
    if not summaries:
        return
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": Path(dataset_path).name,
        "params": dict(ablation),
        "results": summaries,
    }
    try:
        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\n本次结果已追加到 {RESULTS_PATH.name}（用于跨参数对比）")
    except Exception as e:
        print(f"\n!! 结果落盘失败（不影响本次评测）: {e}")


def _compare_with_history(summaries: dict, ablation: dict) -> None:
    """找出历史里"同数据集、同通道、改了某一个参数"的记录，打单因子对比表。

    这是消融的核心：固定其它变量，只看一个参数变化时指标涨没涨。
    """
    if not summaries or not RESULTS_PATH.exists():
        return
    cur_key = _result_key(ablation)
    history = []
    try:
        for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                history.append(json.loads(line))
    except Exception:
        return

    # 按参数键分组，只保留每组最后一次运行；本次参数对应的记录排除掉
    prev_by_key: dict = {}
    for rec in history:
        key = _result_key(rec.get("params", {}))
        if key != cur_key:
            prev_by_key[key] = rec

    if not prev_by_key:
        print("\n（历史里还没有其它参数组合的记录——换个参数再跑一次就能出对比表）")
        return

    print("\n══════════ 单因子对比（同通道 · 历史 vs 本次）══════════")
    # 单因子优先：优先和"只差一个参数"的历史记录比。若历史里只有
    # 多参数差异的记录，退而取其一并显式标注变了几个变量。
    def _changed(prev: dict) -> list[str]:
        prev_p = prev.get("params", {})
        return [
            f"{name}: {prev_p.get(name)} → {ablation.get(name)}"
            for name in ("use_hyde", "top_k", "rerank_top_k")
            if prev_p.get(name) != ablation.get(name)
        ]

    candidates = [(len(_changed(rec)), rec) for rec in prev_by_key.values()]
    candidates = [(n, rec) for n, rec in candidates if n > 0]
    if not candidates:
        print("\n（历史里还没有其它参数组合的记录——换个参数再跑一次就能出对比表）")
        return
    best_n = min(n for n, _ in candidates)
    if best_n > 1:
        print(f"\n!! 历史里没有「只差一个参数」的记录，下面展示差异最小的那组"
              f"（相差 {best_n} 个参数）——\n"
              f"   严格单因子对比需要固定其它参数再跑一次，否则涨跌归因不准。")
    shown = [(n, rec) for n, rec in candidates if n == best_n]
    if len(shown) >= 1:
        n_single = sum(1 for n, _ in candidates if n == 1)
        if n_single > 1:
            print(f"\n（历史里有 {n_single} 组单因子对比记录，展示其中差异最小的一组）")

    for _, prev in sorted(shown, key=lambda x: _result_key(x[1]["params"])):
        changed = _changed(prev)
        print(f"\n  变更项 → {' | '.join(changed)}")
        print(f"  {'通道':8s} {'指标':12s} {'历史':>9s} {'本次':>9s} {'变化':>9s}")
        for ch, cur in summaries.items():
            old = prev.get("results", {}).get(ch)
            if not old:
                continue
            for metric in sorted(k for k in cur if k not in ("count", "latency_ms_avg")):
                o, n = old.get(metric), cur.get(metric)
                if o is None or n is None:
                    continue
                delta = n - o
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
                print(f"  {ch:8s} {metric:12s} {o:>9.3f} {n:>9.3f} {arrow}{abs(delta):>8.3f}")
            lo, ln = old.get("latency_ms_avg"), cur.get("latency_ms_avg")
            if lo and ln:
                d = ln - lo
                arrow = "↑" if d > 0 else ("↓" if d < 0 else "=")
                print(f"  {ch:8s} {'延迟ms':12s} {lo:>9.0f} {ln:>9.0f} {arrow}{abs(d):>8.0f}")


# ── 单通道评测 ───────────────────────────────────────────────────────────


async def run_channel_eval(channel: str, questions: list[str]):
    session = get_trulens_session()
    provider = get_llm_provider()
    metrics = build_rag_triad_metrics(provider)

    tracked = build_tracked_rag(channel)
    latencies = []
    tru_app = TruApp(
        tracked,
        app_name="ALM-RAG",
        app_version=channel,
        feedbacks=metrics,
    )

    print(f"\n{'='*60}")
    print(f"通道: {channel} | 题目数: {len(questions)}")
    print(f"{'='*60}")

    for i, question in enumerate(questions, 1):
        start = time.perf_counter()
        with tru_app:
            answer = await tracked.query(question)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        print(f"  [{i}/{len(questions)}] {question[:50]}... → {latency_ms:.0f}ms")
        print(f"    回答: {answer[:100]}...")
        print()

    print_latency_stats(latencies, channel)

    # ── TruLens 标准生命周期 ──
    # ★ compute_feedbacks 只是把反馈丢给后台评估器线程，不等算完；
    #   必须轮询等全部指标落库后再 stop_evaluator，否则进程退出会杀掉
    #   未完成的反馈（NaN 根因）。
    print(f"  >>> {channel} 记录完成，计算反馈中...")
    session.force_flush()
    # ★ trulens 2.14 的 compute_feedbacks 与 OTEL 事件入库存在竞态
    #   （偶发 KeyError: record_id / 线程池关闭），重试 3 次并留入库缓冲
    for attempt in range(3):
        try:
            await asyncio.sleep(8 * (attempt + 1))
            tru_app.compute_feedbacks(raise_error_on_no_feedbacks_computed=False)
            break
        except Exception as e:
            print(f"    [{channel}] compute_feedbacks 第{attempt+1}次失败: {type(e).__name__}: {str(e)[:120]}")
    else:
        print(f"    [{channel}] ⚠️ 三次尝试均失败，跳过本通道反馈")

    # ★ 用官方 API 等全部反馈落库（record_ids 从 events 拿）
    recs, _ = session.get_records_and_feedback(
        app_name="ALM-RAG", app_versions=[channel])
    ids = []
    for r in recs:
        rid = getattr(r, "record_id", None) or (r.get("record_id") if isinstance(r, dict) else None)
        if rid:
            ids.append(rid)
    names = [m.name for m in metrics]
    print(f"    [{channel}] 等待 {len(ids)} 条记录 × {len(names)} 指标...")
    try:
        session.wait_for_feedback_results(ids, names, timeout=600, poll_interval=5)
        print(f"    [{channel}] 反馈全部落库 ✓")
    except Exception as e:
        print(f"    [{channel}] 等待反馈超时（部分指标可能缺失）: {e}")

    tru_app.stop_evaluator()
    session.force_flush()

    leaderboard = session.get_leaderboard()
    print(f"  >>> {channel} Leaderboard:\n{leaderboard}")

    usage = tracked.token_usage
    print(f"  >>> {channel} Token: input={usage['input_tokens']} output={usage['output_tokens']} "
          f"cost=${usage['cost_usd']:.6f}")


# ── 主入口 ───────────────────────────────────────────────────────────────

async def main(channel: str | None = None, dashboard: bool = False,
               retrieval_only: bool = False, dataset: str | None = None,
               use_hyde: bool | None = None, top_k: int | None = None,
               rerank_top_k: int | None = None, label: str | None = None):
    if dashboard:
        print("启动 TruLens Dashboard → http://localhost:8501")
        launch_dashboard(port=8501)
        return

    channels = [channel] if channel else CHANNELS

    if retrieval_only:
        path = dataset or str(DATASET_PATH)
        items = load_eval_dataset(path)
        labeled = sum(1 for i in items if i["labeled"])
        print(f"标注集: {path}（共 {len(items)} 题，其中已标注 {labeled} 题）")
        if not labeled:
            print("\n!! 标注集里没有任何 relevant_doc_ids —— 全部题目会被跳过，"
                  "跑不出指标。\n"
                  "   请先给每题填上命中的 doc_name（多个用分号分隔），"
                  "doc_name 可从检索日志 'DocRAG 召回' 或 Milvus 里取。")
            return

        _s = get_settings()
        ablation = {
            "use_hyde": _s.RAG_HYDE_ENABLED if use_hyde is None else use_hyde,
            "top_k": _s.RAG_TOP_K if top_k is None else top_k,
            "rerank_top_k": _s.RAG_RERANK_TOP_K if rerank_top_k is None else rerank_top_k,
            "label": label or "",
        }
        print(f"消融参数: use_hyde={ablation['use_hyde']} top_k={ablation['top_k']} "
              f"rerank_top_k={ablation['rerank_top_k']}"
              + (f" label={ablation['label']}" if ablation["label"] else ""))

        summaries: dict = {}
        for ch in channels:
            summary = await run_retrieval_eval(
                ch, items, use_hyde=use_hyde, top_k=top_k, rerank_top_k=rerank_top_k)
            if summary:
                summaries[ch] = summary

        _save_retrieval_results(summaries, ablation, path)
        _compare_with_history(summaries, ablation)
        print("\n检索指标评测完成!")
        return

    for ch in channels:
        try:
            await run_channel_eval(ch, EVAL_QUESTIONS)
        except Exception as e:
            print(f"\n!! 通道 {ch} 评测失败（继续下一通道）: {type(e).__name__}: {str(e)[:200]}\n")

    print("\n全部通道评测完成!")
    print("启动 Dashboard 查看对比结果:")
    print("  python scripts/run_rag_experiments.py --dashboard")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 检索通路评测（TruLens 2.x）")
    parser.add_argument("--channel", choices=CHANNELS, help="单通道评测")
    parser.add_argument("--list", action="store_true", help="列出评测题目")
    parser.add_argument("--dashboard", action="store_true", help="启动 TruLens Dashboard")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="只跑检索指标（HitRate@K/Recall@K/MRR），不跑生成与 LLM 裁判")
    parser.add_argument("--dataset", default=None,
                        help=f"标注集 CSV 路径（默认 {DATASET_PATH.name}）")
    parser.add_argument("--use-hyde", dest="use_hyde", action="store_true", default=None,
                        help="开启 HyDE（默认取 Settings.RAG_HYDE_ENABLED）")
    parser.add_argument("--no-hyde", dest="use_hyde", action="store_false",
                        help="关闭 HyDE")
    parser.add_argument("--top-k", type=int, default=None,
                        help="初召条数（默认取 Settings.RAG_TOP_K）")
    parser.add_argument("--rerank-top-k", type=int, default=None,
                        help="精排目标条数（默认取 Settings.RAG_RERANK_TOP_K）")
    parser.add_argument("--label", default=None,
                        help="本次运行的标签，写进结果文件便于辨认")
    args = parser.parse_args()

    if args.list:
        print(f"评测通道: {CHANNELS}")
        print(f"评测问题 ({len(EVAL_QUESTIONS)} 条):")
        for q in EVAL_QUESTIONS:
            print(f"  - {q}")
    else:
        asyncio.run(main(
            channel=args.channel,
            dashboard=args.dashboard,
            retrieval_only=args.retrieval_only,
            dataset=args.dataset,
            use_hyde=args.use_hyde,
            top_k=args.top_k,
            rerank_top_k=args.rerank_top_k,
            label=args.label,
        ))
