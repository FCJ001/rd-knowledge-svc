#!/usr/bin/env python3
# ============================================================
# RAG 消融实验脚本
#
# 用法:
#   python scripts/run_rag_experiments.py              # 跑全部实验
#   python scripts/run_rag_experiments.py --channel doc_rag  # 单通道
#   python scripts/run_rag_experiments.py --list              # 列出实验计划
#
# 对照维度:
#   chunk_strategy: fixed / semantic / parent_child
#   top_k: 10 / 20 / 40
#   use_hyde: true / false
#   use_hybrid: true / false
# ============================================================

import argparse
import asyncio
import json
import time
from pathlib import Path

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI

from src.core.config import get_settings
from src.infra.milvus_client import get_milvus_client
from src.infra.neo4j_client import get_neo4j_driver
from src.rag.evaluation.tracked_rag import TrackedRAG
from src.rag.evaluation.metrics import build_all_metrics
from src.rag.evaluation.trulens_config import get_llm_provider, get_trulens_session


# ── 实验计划 ────────────────────────────────────────────────────────────

EXPERIMENT_PLAN = [
    # 基线
    {"name": "baseline", "chunk_strategy": "fixed", "top_k": 20, "use_hyde": False, "use_hybrid": False},
    # HyDE
    {"name": "hyde_on", "chunk_strategy": "fixed", "top_k": 20, "use_hyde": True, "use_hybrid": False},
    # 混合检索
    {"name": "hybrid_on", "chunk_strategy": "fixed", "top_k": 20, "use_hyde": False, "use_hybrid": True},
    # HyDE + 混合
    {"name": "hyde_hybrid", "chunk_strategy": "fixed", "top_k": 20, "use_hyde": True, "use_hybrid": True},
    # 语义切片
    {"name": "semantic_chunk", "chunk_strategy": "semantic", "top_k": 20, "use_hyde": False, "use_hybrid": False},
    # 父子切片
    {"name": "parent_child", "chunk_strategy": "parent_child", "top_k": 20, "use_hyde": False, "use_hybrid": False},
    # Top-K 变体
    {"name": "topk_10", "chunk_strategy": "fixed", "top_k": 10, "use_hyde": False, "use_hybrid": False},
    {"name": "topk_40", "chunk_strategy": "fixed", "top_k": 40, "use_hyde": False, "use_hybrid": False},
]

# 评测问题集（ALM 汽车领域）
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


def build_tracked_apps(config: dict) -> dict[str, TrackedRAG]:
    """根据实验配置构建追踪 App"""
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

    channels = ["doc_rag", "graph_rag", "fusion"]
    apps = {}
    for channel in channels:
        apps[channel] = TrackedRAG(
            channel=channel,
            llm=llm,
            embedding_model=embedding_model,
            milvus_client=milvus_client,
            neo4j_driver=neo4j_driver,
            role="engineer",
        )
    return apps


async def run_single_experiment(
    app: TrackedRAG, question: str,
) -> dict:
    """对单个 App 跑单条问题，记录延迟"""
    start = time.perf_counter()
    answer = await app.query(question)
    latency_ms = (time.perf_counter() - start) * 1000
    return {"question": question, "answer": answer, "latency_ms": round(latency_ms, 1)}


async def run_experiments(channel: str | None = None, dry_run: bool = False):
    """执行消融实验"""
    channels = [channel] if channel else ["doc_rag", "graph_rag", "fusion"]

    for exp_config in EXPERIMENT_PLAN:
        exp_name = exp_config["name"]

        for ch in channels:
            if dry_run:
                print(f"[DRY RUN] {exp_name} / {ch}")
            else:
                print(f"\n{'='*60}")
                print(f"实验: {exp_name} | 通道: {ch}")
                print(f"配置: {json.dumps(exp_config, ensure_ascii=False)}")
                print(f"{'='*60}")

                apps = build_tracked_apps(exp_config)
                app = apps.get(ch)
                if not app:
                    print(f"  跳过: 通道 {ch} 不可用")
                    continue

                for i, question in enumerate(EVAL_QUESTIONS, 1):
                    result = await run_single_experiment(app, question)
                    print(f"  [{i}/{len(EVAL_QUESTIONS)}] {question[:40]}... "
                          f"→ {result['latency_ms']:.0f}ms")
                    print(f"    回答: {result['answer'][:100]}...")


def main():
    parser = argparse.ArgumentParser(description="RAG 消融实验")
    parser.add_argument("--channel", help="单通道实验: doc_rag / graph_rag / fusion")
    parser.add_argument("--list", action="store_true", help="列出实验计划")
    parser.add_argument("--dry-run", action="store_true", help="仅打印实验列表，不实际执行")
    args = parser.parse_args()

    if args.list:
        print("实验计划:")
        for exp in EXPERIMENT_PLAN:
            print(f"  {exp['name']}: {json.dumps(exp, ensure_ascii=False)}")
        print(f"\n评测问题 ({len(EVAL_QUESTIONS)} 条):")
        for q in EVAL_QUESTIONS:
            print(f"  - {q}")
        return

    asyncio.run(run_experiments(channel=args.channel, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
