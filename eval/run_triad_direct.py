# ============================================================
# RAG Triad 直连评测（同步、无 TruLens 存储链路）
#
# 背景：trulens 2.14 的 compute_feedbacks / OTEL 入库存在竞态，
# 后台评估器在多通道长跑下产物不可靠（偶发 KeyError、NaN、
# interpreter shutdown 杀线程）。本脚本绕开存储链路，用同一个
# 裁判 Provider（DashScopeLiteLLM，含 JSON 兼容修复）同步计算
# 三指标并就地汇总 —— 指标口径与 TruLens RAG Triad 完全一致。
#
# 用法：
#   python eval/run_triad_direct.py                 # 全通道
#   python eval/run_triad_direct.py --channel fusion
# ============================================================
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

# 结果落盘：一次运行一行 JSON（含逐题分数），供跨通道/跨版本对比
TRIAD_RESULTS_PATH = Path(__file__).resolve().parent / "triad_results.jsonl"


def build_tracked_rag(channel: str):
    """与 scripts/run_rag_experiments.py 的同名函数保持一致。"""
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_openai import ChatOpenAI

    from src.core.config import get_settings
    from src.infra.milvus_client import get_milvus_client
    from src.infra.neo4j_client import get_neo4j_driver
    from src.rag.evaluation.tracked_rag import TrackedRAG

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
    return TrackedRAG(
        channel=channel,
        llm=llm,
        embedding_model=embedding_model,
        milvus_client=get_milvus_client(),
        neo4j_driver=get_neo4j_driver(),
        role="engineer",
    )


async def _judge(provider_fn, *args, key: str, scores: dict, max_retry: int = 3):
    """同步裁判包一层重试；fn 返回 (score, reason) 或 score。"""
    for attempt in range(max_retry):
        try:
            r = provider_fn(*args)
            score = r[0] if isinstance(r, tuple) else r
            scores[key].append(float(score))
            return float(score)
        except Exception as e:
            if attempt == max_retry - 1:
                print(f"    [{key}] {max_retry}次失败: {str(e)[:80]}", flush=True)
            await asyncio.sleep(3)
    return None


async def eval_channel(channel: str, questions: list[str]) -> dict:
    from src.rag.evaluation.tracked_rag import TrackedRAG
    from src.rag.evaluation.trulens_config import get_llm_provider

    tracked = build_tracked_rag(channel)
    provider = get_llm_provider()

    scores: dict[str, list] = {"答案相关性": [], "上下文相关性": [], "有据性": []}
    latencies = []

    for i, q in enumerate(questions, 1):
        start = time.perf_counter()
        contexts = await tracked.retrieve(q)
        answer = await tracked.generate(q, contexts)
        latencies.append((time.perf_counter() - start) * 1000)

        # ★ 与生成侧共用同一个上下文窗口。原来这里写死 8 条、generate 用 10 条，
        #   裁判比生成少看两条 —— 排在列表后面的图谱证据被截掉，导致
        #   「上下文相关性 / 有据性」系统性偏低（出现答案满分却判无据的怪象）。
        ctx_text = ("\n\n".join(contexts[: TrackedRAG.CONTEXT_WINDOW])
                    if contexts else "（无检索结果）")

        a_rel = await _judge(provider.relevance_with_cot_reasons,
                             q, answer, key="答案相关性", scores=scores)
        c_rel = await _judge(provider.context_relevance_with_cot_reasons,
                             q, ctx_text, key="上下文相关性", scores=scores)
        gnd = await _judge(provider.groundedness_measure_with_cot_reasons,
                           ctx_text, answer, key="有据性", scores=scores)

        fmt = lambda x: f"{x:.2f}" if x is not None else "×"
        print(f"  [{i}/{len(questions)}] 答案{fmt(a_rel)} 上下文{fmt(c_rel)} "
              f"有据{fmt(gnd)}  | {q[:34]}...", flush=True)

    # 库内/库外拆分（标注依据语料检查：库内仅 4 份 EV 规范 PDF ——
    # 见 Milvus alm_docs：CATARC高压安全测评 / 充电基础设施指南 /
    # 动力电池梯次利用标准 / 纯电动汽车高压安全技术规范。
    # OTA / 扭矩标准变更 / 特定CR号 / 版本发布说明在库中无对应文档，
    # 扣分属语料覆盖问题而非检索问题，故单列。
    #
    # ★ 用题目原文匹配，不用下标：原来写 OUT_OF_DOMAIN = {1,3,5,9} 配
    #   enumerate(v)，而下标从 0 开始 —— 实际排除的是 Q2/Q4/Q6/Q10，
    #   注释与代码对不上，且改题目顺序就会静默算错。按题面匹配最稳。
    OUT_OF_DOMAIN_QUESTIONS = {
        "OTA升级失败后如何恢复？",
        "2023年后出厂的车型扭矩标准有哪些变更？",
        "变更CR-2024-00178对底盘控制系统有什么影响？",
        "软件版本v3.2.1有哪些已知问题和修复方案？",
    }
    print(f"\n════ {channel} 汇总 ════")
    for k, v in scores.items():
        inn = [x for i, x in enumerate(v)
               if i < len(EVAL_QUESTIONS) and EVAL_QUESTIONS[i] not in OUT_OF_DOMAIN_QUESTIONS]
        if inn:
            print(f"  {k}: 库内均值 {sum(inn)/len(inn):.3f} (n={len(inn)}) / 全量 {sum(v)/len(v):.3f} (n={len(v)})")
    for k, v in scores.items():
        if v:
            print(f"  {k}: 均值 {sum(v)/len(v):.3f} (n={len(v)})")
    if latencies:
        ordered = sorted(latencies)
        print(f"  端到端延迟: avg={sum(ordered)/len(ordered):.0f}ms "
              f"p50={ordered[len(ordered)//2]:.0f}ms")

    # ★ 落盘：之前只 print 不存，跑完结果就丢了（复现消融时无处可查）。
    #   逐题分数一并保存——只看均值会漏掉"低答案相关性配高有据性"
    #   这类只在单题上才看得出的异常。
    per_question = [
        {
            "index": i + 1,
            "question": EVAL_QUESTIONS[i] if i < len(EVAL_QUESTIONS) else "",
            "in_domain": (i < len(EVAL_QUESTIONS)
                          and EVAL_QUESTIONS[i] not in OUT_OF_DOMAIN_QUESTIONS),
            **{k: (v[i] if i < len(v) else None) for k, v in scores.items()},
        }
        for i in range(max((len(v) for v in scores.values()), default=0))
    ]
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "channel": channel,
        "metrics": {k: (sum(v) / len(v) if v else None) for k, v in scores.items()},
        "in_domain_metrics": {
            k: (sum([x for i, x in enumerate(v)
                     if i < len(EVAL_QUESTIONS)
                     and EVAL_QUESTIONS[i] not in OUT_OF_DOMAIN_QUESTIONS])
                / max(1, len([x for i, x in enumerate(v)
                              if i < len(EVAL_QUESTIONS)
                              and EVAL_QUESTIONS[i] not in OUT_OF_DOMAIN_QUESTIONS])))
            for k, v in scores.items() if v
        },
        "latency_ms_avg": (sum(latencies) / len(latencies)) if latencies else None,
        "per_question": per_question,
    }
    try:
        with open(TRIAD_RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  结果已追加到 {TRIAD_RESULTS_PATH.name}")
    except Exception as e:
        print(f"  !! 落盘失败（不影响评测）: {e}")

    return {k: (sum(v) / len(v) if v else None) for k, v in scores.items()}


async def main(channel: str | None):
    channels = [channel] if channel else CHANNELS
    all_results: dict[str, dict] = {}
    for ch in channels:
        try:
            all_results[ch] = await eval_channel(ch, EVAL_QUESTIONS)
        except Exception as e:
            print(f"\n!! 通道 {ch} 失败（继续下一通道）: {type(e).__name__}: {str(e)[:150]}")

    print("\n══════════ 总表（RAG Triad 均值）══════════")
    fmt = lambda x: f"{x:.3f}" if x is not None else "n/a"
    print(f"{'通道':10s} {'答案相关性':>10s} {'上下文相关性':>12s} {'有据性':>8s}")
    for ch, r in all_results.items():
        print(f"{ch:10s} {fmt(r.get('答案相关性')):>10s} "
              f"{fmt(r.get('上下文相关性')):>12s} {fmt(r.get('有据性')):>8s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Triad 直连评测（同步）")
    parser.add_argument("--channel", choices=CHANNELS)
    args = parser.parse_args()
    asyncio.run(main(args.channel))
