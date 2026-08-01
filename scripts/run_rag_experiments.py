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
# 指标（TruLens RAG Triad）:
#   答案相关性 — Question → Answer
#   上下文相关性 — Question → Context
#   有据性 — Context → Answer
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
from trulens.apps.app import TruApp

from src.core.config import get_settings
from src.infra.milvus_client import get_milvus_client
from src.infra.neo4j_client import get_neo4j_driver
from src.rag.evaluation.feedbacks import build_rag_triad_metrics
from src.rag.evaluation.tracked_rag import TrackedRAG
from src.rag.evaluation.trulens_config import (
    get_llm_provider, get_trulens_session, launch_dashboard,
)


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


# ── 依赖构建 ────────────────────────────────────────────────────────────

def build_tracked_rag(channel: str) -> TrackedRAG:
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
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
    )


# ── 单通道评测 ───────────────────────────────────────────────────────────

async def run_channel_eval(channel: str, questions: list[str]):
    session = get_trulens_session()
    provider = get_llm_provider()
    metrics = build_rag_triad_metrics(provider)

    tracked = build_tracked_rag(channel)
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
        with tru_app as recording:
            answer = await tracked.query(question)
        latency_ms = (time.perf_counter() - start) * 1000
        print(f"  [{i}/{len(questions)}] {question[:50]}... → {latency_ms:.0f}ms")
        print(f"    回答: {answer[:100]}...")
        print()

    # ── TruLens 标准生命周期 ──
    # ★ 必须先 compute_feedbacks 再 stop_evaluator，否则指标无法计算
    print(f"  >>> {channel} 记录完成，计算反馈中...")
    session.force_flush()
    tru_app.compute_feedbacks(raise_error_on_no_feedbacks_computed=False)
    tru_app.stop_evaluator()
    session.force_flush()

    leaderboard = session.get_leaderboard()
    print(f"  >>> {channel} Leaderboard:\n{leaderboard}")

    usage = tracked.token_usage
    print(f"  >>> {channel} Token: input={usage['input_tokens']} output={usage['output_tokens']} "
          f"cost=${usage['cost_usd']:.6f}")


# ── 主入口 ───────────────────────────────────────────────────────────────

async def main(channel: str | None = None, dashboard: bool = False):
    if dashboard:
        print("启动 TruLens Dashboard → http://localhost:8501")
        launch_dashboard(port=8501)
        return

    channels = [channel] if channel else CHANNELS
    for ch in channels:
        await run_channel_eval(ch, EVAL_QUESTIONS)

    print("\n全部通道评测完成!")
    print("启动 Dashboard 查看对比结果:")
    print("  python scripts/run_rag_experiments.py --dashboard")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 检索通路评测（TruLens 2.x）")
    parser.add_argument("--channel", choices=CHANNELS, help="单通道评测")
    parser.add_argument("--list", action="store_true", help="列出评测题目")
    parser.add_argument("--dashboard", action="store_true", help="启动 TruLens Dashboard")
    args = parser.parse_args()

    if args.list:
        print(f"评测通道: {CHANNELS}")
        print(f"评测问题 ({len(EVAL_QUESTIONS)} 条):")
        for q in EVAL_QUESTIONS:
            print(f"  - {q}")
    else:
        asyncio.run(main(channel=args.channel, dashboard=args.dashboard))
