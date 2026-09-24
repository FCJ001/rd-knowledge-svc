#!/usr/bin/env python3
# ============================================================
# 检索质量回归门禁
#
# 用途：把「这次改动让检索变好了还是变差了」变成一条可自动判定的命令，
#       而不是靠人肉看日志。CI 里作为独立 job 跑。
#
# 用法:
#   python scripts/eval_gate.py --baseline eval/rag_baseline.json
#   python scripts/eval_gate.py --update-baseline          # 人工确认后刷新基线
#
# 退出码:
#   0  通过（或指标优于基线）
#   1  指标回退超过阈值 → 挡住合并
#   2  环境不可用（Milvus/embedding 连不上）→ 跳过
#       ★ 若 CI 变量 RAG_EVAL_REQUIRED=1，则退出码 2 视为失败：
#         发布流水线必须真的跑过，不能让"跳过"变成常态。
#
# 门禁指标：hit_rate@5（主）、mrr（主）、hit_rate@20（防召回退化）
# 阈值默认 5 个百分点——低于此幅度的波动在千级语料上分不出信号。
# ============================================================

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from src.rag.evaluation.retrieval_metrics import (
    evaluate_retrieval,
    extract_doc_ids,
    load_eval_dataset,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "scripts" / "eval_dataset.csv"
DEFAULT_BASELINE = REPO_ROOT / "eval" / "rag_baseline.json"

# 参与门禁的指标及其方向（True = 越大越好）
GATED_METRICS = {
    "hit_rate@5": True,
    "mrr": True,
    "hit_rate@20": True,
}
GATED_KS = (5, 10, 20)

# 黄金集最小规模：低于这个数指标的方差会盖过真实差异，门禁失去意义
MIN_LABELED_QUESTIONS = 30


def _check_dataset_health(dataset: list[dict]) -> tuple[bool, str]:
    """门禁自身的前提校验：标注太少时指标不可信，必须显式失败而不是"通过"。"""
    labeled = [item for item in dataset if item.get("labeled")]
    if len(labeled) < MIN_LABELED_QUESTIONS:
        return False, (
            f"标注题数 {len(labeled)} < 门禁下限 {MIN_LABELED_QUESTIONS}："
            "样本太少时指标波动会盖过真实差异，门禁无意义。"
            "请先补标注（relevant_doc_ids / relevant_pages）再启用门禁。"
        )
    relevant_counts = [len(item.get("relevant") or []) for item in labeled]
    if all(c == len(labeled[0].get("relevant") or []) for c in relevant_counts) and len(set(relevant_counts)) == 1:
        # 不是错误，只是提醒：单一标签数量会让 recall 与 hit_rate 同构
        print(f"  [提示] 每题标签数恒为 {relevant_counts[0]}，"
              "hit_rate 与 recall 会高度同构，建议补 page 级标签提升分辨力")
    return True, ""


def compare_to_baseline(current: dict, baseline: dict, max_regression: float) -> list[str]:
    """返回回退项描述列表（空列表 = 通过）。纯函数，便于单测。"""
    regressions: list[str] = []
    for metric, higher_is_better in GATED_METRICS.items():
        cur = current.get(metric)
        base = (baseline.get("metrics") or {}).get(metric)
        if cur is None or base is None:
            continue
        delta = (cur - base) if higher_is_better else (base - cur)
        if delta < -max_regression:
            regressions.append(
                f"{metric}: {base:.4f} → {cur:.4f}（回退 {abs(delta):.4f}，"
                f"阈值 {max_regression}）"
            )
    return regressions


async def _run_eval(dataset: list[dict]) -> dict | None:
    """跑一次检索评测。基础设施不可用时返回 None（由调用方转成退出码 2）。"""
    from src.rag.evaluation.tracked_rag import TrackedRAG

    labeled = [item for item in dataset if item.get("labeled")]
    try:
        from src.core.config import get_settings
        from src.infra.milvus_client import get_milvus_client
        from src.infra.neo4j_client import get_neo4j_driver

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
        tracked = TrackedRAG(
            channel="doc_rag",
            llm=llm,
            embedding_model=embedding_model,
            milvus_client=get_milvus_client(),
            neo4j_driver=get_neo4j_driver(),
            role="engineer",
        )
    except Exception as e:
        print(f"  [跳过] 评测环境不可用: {type(e).__name__}: {e}")
        return None

    results = []
    for item in labeled:
        hits = await tracked.retrieve_docs(item["question"])
        results.append((extract_doc_ids(hits), item["relevant"]))

    summary = evaluate_retrieval(results, ks=GATED_KS)
    print(f"  本次指标（分母 {summary.get('doc_labeled', summary['count'])} 题）: "
          + "  ".join(f"{m}={summary[m]:.4f}" for m in GATED_METRICS if m in summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="检索质量回归门禁")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="标注数据集 CSV")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE), help="基线 JSON")
    parser.add_argument("--max-regression", type=float, default=0.05,
                        help="允许的最大回退幅度（默认 0.05 = 5 个百分点）")
    parser.add_argument("--update-baseline", action="store_true",
                        help="把本次结果写为新的基线（人工确认后再用）")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[失败] 数据集不存在: {dataset_path}")
        return 1

    dataset = load_eval_dataset(str(dataset_path))
    ok, msg = _check_dataset_health(dataset)
    if not ok:
        print(f"[失败] {msg}")
        return 1

    summary = asyncio.run(_run_eval(dataset))
    if summary is None:
        if os.environ.get("RAG_EVAL_REQUIRED") == "1":
            print("[失败] RAG_EVAL_REQUIRED=1 但评测环境不可用：发布流水线不允许跳过门禁")
            return 1
        print("[跳过] 环境不可用，门禁未执行（设置 RAG_EVAL_REQUIRED=1 可要求必须执行）")
        return 2

    baseline_path = Path(args.baseline)
    if args.update_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps({"metrics": {m: summary[m] for m in GATED_METRICS if m in summary},
                        "dataset": dataset_path.name,
                        "labeled": summary["count"]},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[已更新基线] {baseline_path}")
        return 0

    if not baseline_path.exists():
        print(f"[失败] 基线不存在: {baseline_path}。首次启用请先跑 --update-baseline 并人工确认。")
        return 1

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    regressions = compare_to_baseline(summary, baseline, args.max_regression)
    if regressions:
        print("\n[回退] 以下指标低于基线超过阈值：")
        for r in regressions:
            print(f"  - {r}")
        return 1

    print("\n[通过] 检索指标未回退")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
