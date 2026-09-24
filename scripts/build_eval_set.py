#!/usr/bin/env python3
# ============================================================
# 评测集构建工具
#
# 现状：scripts/eval_dataset_corpus5.csv 只有 10 题、且是 doc 级标签——
# 实测 hit_rate@5/@10/@20 三个 k 完全相同，指标不含排序信息，
# 任何调参都读不出信号。本工具用来把评测集扩到有分辨力的规模。
#
# 两个子命令：
#
#   skeleton  生成骨架 + 覆盖计划（不出网，不需要 LLM）
#       python scripts/build_eval_set.py skeleton --out scripts/eval_skeleton.csv
#     产出：带表头与格式示例的 CSV、每题该覆盖哪个文档/模态的配额表。
#
#   draft     LLM 起草候选题（每块一题，页码由源块自动填好）
#       python scripts/build_eval_set.py draft --per-doc 20 --out scripts/eval_candidates.csv
#     ★ 起草结果**必须逐条人工验证**后才可计入评测集：
#       已验证的流程是"手写约 1/3 + LLM 起草其余 + 全量人工验证"，
#       拒绝率约 5.9%、另有约 20% 需要修改（arXiv:2605.28222）。
#
# 评审通过后合并进 datasets 的流程：
#   1. 人工检查 candidates.csv 每行：问题是否可从标注页回答、类型/模态是否正确、页码是否准确
#   2. 删掉不可以的，把剩下的追加到 scripts/eval_dataset.csv
#   3. python scripts/eval_gate.py --update-baseline   # 确认无回退后刷新基线
#
# 标签粒度：**页级优先**（relevant_pages=文档名.pdf:p12）。页级标签会自动派生
# doc 级，因此不必两处都填；页级指标才反映"召回得准不准"。
# ============================================================

import argparse
import asyncio
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
COLLECTION = "alm_docs"

# 评测集列（顺序即写出的顺序）
COLUMNS = [
    "question", "relevant_doc_ids", "relevant_pages",
    "answer_type", "modality", "expect_no_answer", "note",
]

# 分层维度（与 runbook 的方案一致）
ANSWER_TYPES = ("exact_value", "descriptive", "cross_section")
MODALITIES = ("text", "table", "formula", "image")

# 单页最多取多少块做起草候选（同页重叠块会造成近似重复题）
_MAX_CHUNKS_PER_PAGE = 2


def _detect_modality(text: str) -> str:
    """按块内容判定模态——决定这题考的是哪种内容类型。"""
    if "<table" in text:
        return "table"
    if "$$" in text or "![" in text and "公式原图" in text:
        return "formula"
    if "![" in text:
        return "image"
    return "text"


def _clean_question(q: str) -> str:
    """清洗 LLM 起草的问题：去掉编号/前缀/引号包裹。"""
    q = (q or "").strip()
    q = re.sub(r"^\s*(\d+[\.\)、]|[-*])\s*", "", q)
    q = q.strip().strip('"').strip("“”").strip()
    return q


async def _load_chunks() -> list[dict]:
    """从 Milvus 读取全部可检索块（doc_name / page_number / text）。"""
    from src.infra.milvus_client import get_milvus_client

    client = get_milvus_client()
    rows: list[dict] = []
    try:
        rows = await asyncio.to_thread(
            client.query,
            collection_name=COLLECTION,
            filter="",
            output_fields=["doc_name", "page_number", "text"],
            limit=16384,
        )
    except Exception as e:
        print(f"[失败] 无法读取 Milvus: {type(e).__name__}: {e}")
        return []
    return [r for r in rows if r.get("doc_name") and r.get("text")]


def _warn_if_pages_missing(chunks: list[dict]) -> None:
    """页码为 0 时提示后果——页级标签与引用页码都依赖它。"""
    if any(isinstance(c.get("page_number"), int) and c["page_number"] > 0 for c in chunks):
        return
    print("⚠️  全部 chunk 的 page_number 为 0（页码未知）")
    print("    → 页级标签（relevant_pages）不可用，本次只能产出 doc 级标签")
    print("    → 答案引用也不会显示页码")
    print("    修复需重新入库（依赖 MinerU 服务），详见 `build_eval_set.py pages`")
    print()


def _plan_coverage(chunks: list[dict], target: int) -> list[tuple[str, int]]:
    """按文档分配题量配额。

    ★ 不按块占比分配：实测 81% 的块来自同一本维修手册，按比例分配会让
    评测集几乎只考那一本，另外 4 份文档形同未测。这里对每份文档先保底，
    余额再按块数加权，保证每份文档都有足够题目暴露问题。
    """
    by_doc = Counter(c["doc_name"] for c in chunks)
    docs = sorted(by_doc, key=lambda d: -by_doc[d])
    if not docs:
        return []

    floor = max(3, target // (len(docs) * 2))   # 每份文档的保底题数
    quota = {d: min(floor, by_doc[d]) for d in docs}
    remaining = target - sum(quota.values())
    if remaining > 0:
        total = sum(by_doc.values())
        for d in docs:
            add = int(round(remaining * by_doc[d] / total))
            quota[d] = min(quota[d] + add, by_doc[d])
    return sorted(quota.items(), key=lambda kv: -kv[1])


def _sample_for_draft(chunks: list[dict], per_doc: int) -> list[dict]:
    """按文档 + 模态分层抽样，供起草用。

    优先取表格/公式/图片块——这些是检索最容易出问题的内容类型，
    纯文本题的价值低得多。
    """
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        c = dict(c)
        c["modality"] = _detect_modality(c["text"])
        by_doc[c["doc_name"]].append(c)

    picked: list[dict] = []
    for doc, items in by_doc.items():
        by_mod: dict[str, list[dict]] = defaultdict(list)
        for it in items:
            by_mod[it["modality"]].append(it)

        # 每页最多取 _MAX_CHUNKS_PER_PAGE 块，避免同页重叠块造出近似重复题
        per_page = Counter()
        for mod in ("table", "formula", "image", "text"):
            bucket = by_mod.get(mod) or []
            random.shuffle(bucket)
            for it in bucket:
                if len([p for p in picked if p["doc_name"] == doc]) >= per_doc:
                    break
                page = it.get("page_number")
                if per_page[page] >= _MAX_CHUNKS_PER_PAGE:
                    continue
                # 太短的块问不出有价值的问题
                if len(it["text"]) < 120:
                    continue
                per_page[page] += 1
                picked.append(it)
    return picked


_DRAFT_PROMPT = """你是汽车研发知识库的评测题出题人。
下面是一段来自文档的片段，请针对它出一道**能从该片段找到答案**的问题。

要求：
- 问题必须是真实工程师会问的，不要出现"根据以下片段"这类元话术
- 答案必须完整落在片段内；片段信息不足就别出题，返回空
- 问题用中文，一句话

片段（来自《{doc_name}》第 {page} 页）：
<片段>
{text}
</片段>

只输出一行 JSON，不要任何解释：
{{"question": "问题文本", "answer_type": "exact_value|descriptive|cross_section"}}

answer_type 含义：
- exact_value：答案是一个具体数值/型号/规格（如扭矩 25 N·m）
- descriptive：答案是说明性文字（如为什么会出现某故障）
- cross_section：需要跨章节/跨部件综合才能回答
"""


async def cmd_draft(per_doc: int, out_path: Path, seed: int) -> int:
    random.seed(seed)
    from langchain_openai import ChatOpenAI
    from src.core.config import get_settings

    chunks = await _load_chunks()
    if not chunks:
        return 1
    print(f"读取到 {len(chunks)} 个块，分布: {dict(Counter(c['doc_name'] for c in chunks))}")
    _warn_if_pages_missing(chunks)

    picked = _sample_for_draft(chunks, per_doc)
    print(f"分层抽样得到 {len(picked)} 个候选块（按模态优先取表格/公式/图片）")

    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.chat_api_key,
        base_url=settings.BASE_URL_CHAT,
        temperature=0.3,   # 出题需要一点多样性，但不能太发散
    )

    rows = []
    for i, chunk in enumerate(picked, 1):
        prompt = _DRAFT_PROMPT.format(
            doc_name=chunk["doc_name"],
            page=chunk.get("page_number", "?"),
            text=chunk["text"][:2000],
        )
        try:
            resp = await llm.ainvoke(prompt)
            content = (resp.content or "").strip()
            if "```" in content:
                content = content.split("```")[1].lstrip("json").strip()
            data = json.loads(content)
        except Exception as e:
            logger.warning(f"[{i}/{len(picked)}] 起草失败，跳过: {type(e).__name__}: {e}")
            continue

        question = _clean_question(data.get("question", ""))
        if not question:
            continue
        answer_type = data.get("answer_type", "").strip()
        if answer_type not in ANSWER_TYPES:
            answer_type = "descriptive"

        doc_name = chunk["doc_name"]
        page = chunk.get("page_number")
        # ★ 0 = 页码未知（入库时解析器没给锚点），不能当成"第 0 页"写进标签
        page_ref = f"{doc_name}:p{page}" if page not in ("", None, 0, "0") else ""
        rows.append({
            "question": question,
            "relevant_doc_ids": doc_name,
            "relevant_pages": page_ref,
            "answer_type": answer_type,
            "modality": chunk["modality"],
            "expect_no_answer": "",
            "note": "LLM 起草，待人工验证" + ("" if page_ref else "（页码未知，需人工补页级标签）"),
        })
        print(f"  [{i}/{len(picked)}] {chunk['modality']:7s} {question[:44]}")

    _write_csv(out_path, rows)
    print(f"\n已写出 {len(rows)} 条候选到 {out_path}")
    print("★ 下一步：逐条人工验证。检查点——")
    print("   1) 问题是否真能从标注页（relevant_pages）回答")
    print("   2) 页码是否准确（源块页码由入库流程写入，但跨页块可能偏一页）")
    print("   3) answer_type / modality 是否正确")
    print("   4) 与已有题目是否重复")
    print("   验证通过后把 note 改成空或'已验证'，再合并进 scripts/eval_dataset.csv")
    return 0


def cmd_skeleton(out_path: Path, target: int) -> int:
    chunks = asyncio.run(_load_chunks())
    if not chunks:
        return 1

    _warn_if_pages_missing(chunks)
    plan = _plan_coverage(chunks, target)
    total_planned = sum(q for _, q in plan)

    print("=" * 64)
    print(f"评测集覆盖计划（目标 {target} 题，实际配额 {total_planned} 题）")
    print("=" * 64)
    print(f"{'文档':<44}{'块数':>8}{'建议题数':>10}")
    by_doc = Counter(c["doc_name"] for c in chunks)
    for doc, quota in plan:
        print(f"{doc:<44}{by_doc[doc]:>8}{quota:>10}")
    print()
    print("★ 不按块占比分配：实测块最多的文档占 81%，按比例会让评测集几乎只考它一本。")
    print("  这里对每份文档先保底，余额再按块数加权。")
    print()

    # 模态分布 → 按实际占比给出建议配额（不要硬编码"表格占 15%"这类说法：
    # 语料里公式块可能只有个位数，强行凑配额只会造出没有意义的题）
    mods = Counter(_detect_modality(c["text"]) for c in chunks)
    total_chunks = sum(mods.values())
    print("语料模态分布与建议题量：")
    for mod in MODALITIES:
        n = mods.get(mod, 0)
        share = n / total_chunks if total_chunks else 0
        if n == 0:
            print(f"  {mod:8s} {n:>6} 块   → 语料中没有，不设该模态的题")
        elif n < 10:
            print(f"  {mod:8s} {n:>6} 块   → 过少，合并进 text 类，不单独设配额")
        else:
            # 建议题量按占比算但设下限，保证低占比模态也能被考到
            suggest = max(3, int(target * share))
            print(f"  {mod:8s} {n:>6} 块（{share:.0%}） → 建议 {suggest} 题")
    print()

    print("分层维度（每题在 note 列标出实际取值）：")
    print(f"  answer_type: {' / '.join(ANSWER_TYPES)}")
    print(f"  modality   : {' / '.join(MODALITIES)}")
    print()

    # 骨架：格式示例 + 拒答题占位，而不是 150 行空白
    examples = [
        {
            "question": "（示例·精确值）EV160 前轮毂轴承的紧固扭矩是多少？",
            "relevant_doc_ids": "<文档名>.pdf",
            "relevant_pages": "<文档名>.pdf:p123; <文档名>.pdf:p124",
            "answer_type": "exact_value",
            "modality": "table",
            "expect_no_answer": "",
            "note": "格式示例，请替换为真实题目",
        },
        {
            "question": "（示例·描述性）电池包绝缘电阻偏低可能由哪些原因导致？",
            "relevant_doc_ids": "<文档名>.pdf",
            "relevant_pages": "<文档名>.pdf:p88",
            "answer_type": "descriptive",
            "modality": "text",
            "expect_no_answer": "",
            "note": "格式示例，请替换为真实题目",
        },
        {
            "question": "（示例·跨章节）更换动力电池后需要同步校验哪些系统？",
            "relevant_doc_ids": "<文档名>.pdf;<另一文档>.pdf",
            "relevant_pages": "",
            "answer_type": "cross_section",
            "modality": "text",
            "expect_no_answer": "",
            "note": "格式示例；跨文档题可只填 doc 级",
        },
        {
            "question": "（示例·拒答题）我们的知识库里有没有 2030 款某车型的维修手册？",
            "relevant_doc_ids": "",
            "relevant_pages": "",
            "answer_type": "",
            "modality": "",
            "expect_no_answer": "1",
            "note": "拒答题：库里确实没有答案，用于标定'该拒答'的分数阈值",
        },
    ]
    _write_csv(out_path, examples)
    print(f"骨架已写出到 {out_path}（含 {len(examples)} 行格式示例，非空白行）")
    print()
    print("★ 页级标签怎么写：")
    print("   relevant_pages 填 `文档名.pdf:p页码`，多页用分号分隔。")
    print("   页码可从此处取：")
    print("     python scripts/build_eval_set.py pages           # 列出每份文档的页码范围")
    print("   或直接在 Milvus 里按 doc_name 查 chunk 的 page_number。")
    print("   ★ 页级标签会自动派生 doc 级，不必两处都填。")
    print()
    print("★ 拒答题至少 15 题（占总题量 10% 左右）——它是幻觉治理唯一可量化的验证。")
    return 0


def cmd_pages() -> int:
    chunks = asyncio.run(_load_chunks())
    if not chunks:
        return 1
    by_doc: dict[str, set] = defaultdict(set)
    for c in chunks:
        by_doc[c["doc_name"]].add(c.get("page_number"))

    known = {doc: {p for p in pages if isinstance(p, int) and p > 0}
             for doc, pages in by_doc.items()}
    if not any(known.values()):
        print("=" * 64)
        print("⚠️  所有文档的 page_number 都是 0 —— 页码信息缺失")
        print("=" * 64)
        print("0 是「页码未知」的哨兵（解析器未提供页码锚点时入库会写 0）。")
        print()
        print("影响：")
        print("  1) 答案引用不显示页码（不会显示错误的'第0页'）")
        print("  2) 无法用页级标签，页级检索指标不可用")
        print()
        print("修复：本仓库已修正页码锚点的提取与 chunk→页映射逻辑，")
        print("      但已入库的数据需要**重新入库**才会带上页码。")
        print("      重入库依赖 MinerU 服务（MINERU_API_URL），当前未必在运行：")
        print("        curl -s \"$MINERU_API_URL/health\" || echo 'MinerU 不可达'")
        print()
        print("在此之前：评测集用 doc 级标签（relevant_doc_ids），")
        print("          页级列留空即可——load_eval_dataset 会自动降级到 doc 级。")
        return 0

    print("=" * 64)
    print("各文档的页码范围（用于填 relevant_pages）")
    print("=" * 64)
    for doc in sorted(by_doc):
        nums = sorted(known[doc])
        if nums:
            print(f"{doc}")
            print(f"    页数 {len(nums)}，范围 p{nums[0]}~p{nums[-1]}")
        else:
            print(f"{doc}: 无页码信息（该文档入库时未拿到页码锚点）")
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="评测集构建工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sk = sub.add_parser("skeleton", help="生成骨架与覆盖计划（不出网）")
    p_sk.add_argument("--out", default=str(REPO_ROOT / "scripts" / "eval_skeleton.csv"))
    p_sk.add_argument("--target", type=int, default=150, help="目标题数（默认 150）")

    p_dr = sub.add_parser("draft", help="LLM 起草候选题（结果需人工验证）")
    p_dr.add_argument("--out", default=str(REPO_ROOT / "scripts" / "eval_candidates.csv"))
    p_dr.add_argument("--per-doc", type=int, default=20, help="每份文档起草多少题")
    p_dr.add_argument("--seed", type=int, default=42, help="抽样随机种子（保证可复现）")

    sub.add_parser("pages", help="列出各文档页码范围")

    args = parser.parse_args()
    if args.command == "skeleton":
        return cmd_skeleton(Path(args.out), args.target)
    if args.command == "draft":
        return asyncio.run(cmd_draft(args.per_doc, Path(args.out), args.seed))
    if args.command == "pages":
        return cmd_pages()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
