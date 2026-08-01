#!/usr/bin/env python3
# ============================================================
# NL2SQL 离线评测脚本（TruLens 2.x 标准生命周期）
#
# 用法:
#   python scripts/run_nl2sql_eval.py                    # 跑全部题目
#   python scripts/run_nl2sql_eval.py --limit 5          # 只跑前 5 题
#   python scripts/run_nl2sql_eval.py --output results.json
#   python scripts/run_nl2sql_eval.py --list             # 列出题目
#
# 三指标（TruLens 自动计算）:
#   SQL可执行率  — SQL 是否可执行（客观，0/1）
#   结果相关性  — LLM 裁判：查询结果与问题相关性（0-1）
#   数据返回率  — 有数据=1.0，空结果=0.5（客观）
#
# ★ 录制结果 & 指标存入 TruLens trulens_eval 库，Dashboard 直接查看
# ============================================================

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from trulens.apps.app import TruApp

from src.core.config import get_settings
from src.infra.db import AsyncSessionLocal
from src.infra.db_readonly import ReadOnlySessionLocal
from src.infra.es_client import get_es_client
from src.infra.milvus_client import get_milvus_client
from src.nl2sql.repositories import PgMetaRepository
from src.rag.evaluation.feedbacks import build_nl2sql_metrics
from src.rag.evaluation.nl2sql_metrics import judge_result_relevance, score_sql_valid
from src.rag.evaluation.tracked_nl2sql import TrackedNL2SQL
from src.rag.evaluation.trulens_config import get_llm_provider, get_trulens_session


# ═══════════════════════════════════════════════════════════════════
# NL2SQL 评测题目（ALM 汽车领域，20 道）
# ═══════════════════════════════════════════════════════════════════

EVAL_QUESTIONS = [
    # ── 问题统计 ──
    "各业务线的问题数量分布",
    "S1 严重级别的问题有多少个",
    "近一个月各状态的问题数量",
    "各责任域的问题数量排名",
    "最近创建的前10个问题",
    "问题来源分布统计",
    "已关闭的问题按业务线统计数量",
    "各车型代码对应的问题数量",

    # ── 变更统计 ──
    "各业务线的变更请求数量",
    "待审批状态的变更请求有哪些",
    "近三个月创建的变更请求数量趋势",
    "各状态的变更请求数量分布",
    "看看最近有哪些变更请求",

    # ── 需求统计 ──
    "各业务线的需求数量",
    "高优先级的需求有多少个",
    "各状态的需求数量分布",
    "最近创建的需求有哪些",

    # ── 跨表查询 ──
    "各责任域关联的问题数量和变更请求数量",
    "需求关联的基线有哪些",

    # ── 配置项查询 ──
    "各分类的配置项数量",
    "安全相关的配置项有哪些",
]


# ═══════════════════════════════════════════════════════════════════
# 评测执行
# ═══════════════════════════════════════════════════════════════════


async def build_tracked_nl2sql() -> TrackedNL2SQL:
    """构建 NL2SQL 实例"""
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
    es_client = await get_es_client()
    knowledge_db = AsyncSessionLocal()
    pg_meta_repo = PgMetaRepository(knowledge_db)
    alm_db = ReadOnlySessionLocal()

    tracked = TrackedNL2SQL(
        llm=llm,
        embedding_model=embedding_model,
        milvus_client=milvus_client,
        es_client=es_client,
        pg_meta_repo=pg_meta_repo,
        dw_db_session=alm_db,
        role="engineer",
    )

    tracked._knowledge_db = knowledge_db
    tracked._alm_db = alm_db
    tracked._es_client = es_client

    return tracked


async def run_evaluation(
    questions: list[str],
    output_path: str | None = None,
) -> list[dict]:
    """执行 NL2SQL 评测 — TruLens 2.x 标准生命周期"""
    settings = get_settings()
    tracked = await build_tracked_nl2sql()

    # ── TruLens 2.x 初始化 ──
    session = get_trulens_session()
    provider = get_llm_provider()
    metrics = build_nl2sql_metrics(provider)

    # 即时 LLM 裁判（独立 ChatOpenAI，不走 TruLens provider）
    judge_llm = ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.BASE_URL_CHAT,
        temperature=0,
    )

    tru_app = TruApp(
        tracked,
        app_name="NL2SQL-Eval",
        app_version="1.0",
        feedbacks=metrics,
    )

    results: list[dict] = []

    print(f"\n{'='*60}")
    print(f"NL2SQL 离线评测 — {len(questions)} 道题 × 3 指标")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型: {settings.CHAT_MODEL}")
    print(f"TruLens: 录制中 → 评测结果在 Dashboard :8501 查看")
    print(f"{'='*60}\n")

    # ── 逐题执行 & TruLens 录制 ──
    for i, question in enumerate(questions, 1):
        print(f"[{i:02d}/{len(questions)}] {question}")

        start = time.perf_counter()
        output = {}

        try:
            with tru_app as recording:
                raw_output = await tracked.query(question)

            output = json.loads(raw_output)
            latency_ms = (time.perf_counter() - start) * 1000
        except Exception as e:
            output = {"error": str(e), "sql": "", "data": [], "summary": ""}
            latency_ms = (time.perf_counter() - start) * 1000

        # ── 即时指标（不等 TruLens compute_feedbacks） ──
        error_msg = output.get("error") if output.get("error") else None
        sql_valid = score_sql_valid(error_msg)
        # 数据返回率: 有数据=1.0, SQL成功但无数据=0.5, SQL失败=0.0
        if error_msg:
            has_data = 0.0
        elif output.get("row_count", 0) > 0:
            has_data = 1.0
        else:
            has_data = 0.5

        # LLM 裁判 — 用自定义 prompt 做更精确的相关性判断
        relevance = 0.0
        relevance_reason = ""
        if sql_valid > 0 and output.get("summary"):
            try:
                relevance, relevance_reason = await judge_result_relevance(
                    question,
                    output.get("summary", ""),
                    output.get("data", []),
                    judge_llm,
                )
            except Exception as e:
                print(f"    ⚠ 相关性评分失败: {e}")

        correctness = (sql_valid + has_data + relevance) / 3.0

        result = {
            "index": i,
            "question": question,
            "sql": output.get("sql", ""),
            "error": output.get("error", ""),
            "row_count": output.get("row_count", 0),
            "summary": output.get("summary", "")[:200],
            "latency_ms": round(latency_ms, 1),
            "sql_valid": sql_valid,
            "has_data": has_data,
            "result_relevance": round(relevance, 2),
            "relevance_reason": relevance_reason,
            "correctness": round(correctness, 2),
        }
        results.append(result)

        # ── 实时打印 ──
        status = "✓" if sql_valid else "✗"
        print(
            f"    {status} sql_valid={sql_valid:.0f} rows={output.get('row_count', 0)} "
            f"relevance={relevance:.2f} correctness={correctness:.2f} "
            f"latency={latency_ms:.0f}ms"
        )
        if output.get("error"):
            print(f"    SQL错误: {str(output['error'])[:120]}")
        if output.get("sql"):
            print(f"    SQL: {output['sql'][:120]}")
        if relevance_reason:
            print(f"    相关性理由: {relevance_reason[:120]}")
        print()

    # ── TruLens 标准生命周期 ──
    # ★ 必须先 compute_feedbacks 再 stop_evaluator，否则指标无法计算
    print("TruLens 反馈计算中...")
    session.force_flush()
    tru_app.compute_feedbacks(raise_error_on_no_feedbacks_computed=False)
    tru_app.stop_evaluator()
    session.force_flush()

    # ── Leaderboard ──
    leaderboard = session.get_leaderboard()
    print(f"\n{'='*60}")
    print("TruLens Leaderboard:")
    print(leaderboard)
    print(f"{'='*60}")

    # ── 汇总 ──
    total = len(results)
    valid_count = sum(1 for r in results if r["sql_valid"] == 1.0)
    avg_relevance = (
        sum(r["result_relevance"] for r in results) / total if total > 0 else 0
    )
    avg_correctness = (
        sum(r["correctness"] for r in results) / total if total > 0 else 0
    )
    avg_latency = (
        sum(r["latency_ms"] for r in results) / total if total > 0 else 0
    )

    summary = {
        "total": total,
        "sql_valid_rate": round(valid_count / total, 2) if total > 0 else 0,
        "avg_relevance": round(avg_relevance, 2),
        "avg_correctness": round(avg_correctness, 2),
        "avg_latency_ms": round(avg_latency, 1),
    }

    print(f"\n评测完成")
    print(f"  SQL 可执行率:  {summary['sql_valid_rate']} ({valid_count}/{total})")
    print(f"  平均相关性:    {summary['avg_relevance']}")
    print(f"  平均正确性:    {summary['avg_correctness']}")
    print(f"  平均延迟:      {summary['avg_latency_ms']}ms")
    print(f"  Dashboard:     http://localhost:8501")
    print(f"{'='*60}\n")

    # ── 保存 JSON ──
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "results": results,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_path}")

    # ── Token 用量 ──
    usage = tracked.token_usage
    print(f"\nToken 用量: input={usage['input_tokens']} output={usage['output_tokens']} "
          f"total={usage['total_tokens']} calls={usage['calls']} "
          f"cost=${usage['cost_usd']:.6f}")

    # ── 清理 ──
    await tracked._knowledge_db.close()
    await tracked._alm_db.close()
    await tracked._es_client.close()

    return results


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="NL2SQL 离线评测（TruLens 2.x）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题")
    parser.add_argument("--output", type=str, default="", help="结果输出 JSON 路径")
    parser.add_argument("--list", action="store_true", help="列出评测题目")
    args = parser.parse_args()

    if args.list:
        print(f"评测题目 ({len(EVAL_QUESTIONS)} 道):")
        for i, q in enumerate(EVAL_QUESTIONS, 1):
            print(f"  {i:02d}. {q}")
        return

    questions = EVAL_QUESTIONS[: args.limit] if args.limit > 0 else EVAL_QUESTIONS
    output_path = (
        args.output
        or f"nl2sql_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    asyncio.run(run_evaluation(questions, output_path))


if __name__ == "__main__":
    main()
