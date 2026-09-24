#!/usr/bin/env python3
# ============================================================
# 重入库驱动脚本（含文档名修正）
#
#   python scripts/reingest.py --list                 # 只看映射与现状，不动数据
#   python scripts/reingest.py                        # 执行重入库
#   python scripts/reingest.py --only 维修手册          # 只重入库匹配的条目
#
# 为什么需要它：
#   1) 页码链路修复（MinerU v2 协议的页码锚点 + chunk→页 n-gram 映射）
#      只对新入库的数据生效，存量数据 page_number 全为 0；
#   2) 表格/公式块保护切片、检索用上下文前缀同样只在新入库时生效；
#   3) 实测有 2 份文档文件名与内容完全不符（见 docs/runbook.md 4.0.1）。
#
# ★ 不改动源文件：把源 PDF 复制成「正确文件名」的临时副本再入库，
#   源目录保持原样（data/pdfs 里的名字是别人放的，不该由本脚本单方面改）。
#   要改掉的旧文档按 md5(旧名)[:16] 精确定位后删除。
# ============================================================

import argparse
import asyncio
import hashlib
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = REPO_ROOT / "data" / "pdfs"


@dataclass
class Entry:
    source: str          # 磁盘上的文件名（可能名不符实）
    doc_name: str        # 入库使用的正确名称
    doc_type: str
    model_code: str
    note: str
    acl_roles: str = "engineer,business,aftersales,customer"  # 可见角色（检索前过滤）


# 文件名 → 实际内容的对照（依据：逐份打开 PDF 读首页 + 全文关键词核验）
ENTRIES: list[Entry] = [
    Entry("北汽新能源EV160_200维修手册.pdf", "北汽新能源EV160_200维修手册.pdf",
          "repair_manual", "EV160", "名称与内容相符"),
    Entry("CATARC_电动汽车高压安全测评.pdf", "CATARC_GB18384.3修改单宣贯讲义.pdf",
          "spec_doc", "", "原名含机构前缀，改为内容名（GB/T 18384.3 修改单宣贯讲义）"),
    Entry("纯电动汽车高压安全技术规范.pdf", "纯电动汽车出厂安全技术规范_第1部分_高压部件.pdf",
          "spec_doc", "", "内容为 T/ZJSAE 团体标准正文，名称为其简称"),
    Entry("电动汽车动力电池梯次利用标准.pdf", "纯电动汽车出厂安全技术规范_编制说明.pdf",
          "spec_doc", "", "★ 原名与内容完全不符：实为该标准的报批稿编制说明"),
    Entry("电动汽车充电基础设施发展指南.pdf", "电动乘用车安全设计规范.pdf",
          "spec_doc", "", "★ 原名与内容完全不符：实为电动乘用车安全设计规范"),
]


def doc_id_of(name: str) -> str:
    return hashlib.md5(name.encode()).hexdigest()[:16]


def _milvus_doc_names() -> dict[str, int]:
    """当前 collection 里各 doc_name 的块数。"""
    from src.infra.milvus_client import get_milvus_client

    client = get_milvus_client()
    rows = client.query("alm_docs", filter="", output_fields=["doc_name"], limit=16384)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["doc_name"]] = counts.get(r["doc_name"], 0) + 1
    return counts


def show_plan() -> None:
    existing = _milvus_doc_names()
    print("=" * 76)
    print("重入库计划")
    print("=" * 76)
    for e in ENTRIES:
        src = PDF_DIR / e.source
        old_id = doc_id_of(e.source)
        print(f"\n源文件: {e.source}  ({'存在' if src.exists() else '★ 缺失'})")
        if e.doc_name != e.source:
            print(f"  入库名: {e.source}  →  {e.doc_name}")
        else:
            print(f"  入库名: {e.doc_name}（不变）")
        print(f"  类型/车型: {e.doc_type} / {e.model_code or '—'}")
        print(f"  说明: {e.note}")
        print(f"  现有数据: 旧名 {existing.get(e.source, 0)} 块"
              + (f"（doc_id={old_id}，将删除）" if existing.get(e.source) else "（无）"))
    print()
    print(f"现 collection 共 {sum(existing.values())} 块，"
          f"{len(existing)} 份文档")
    print("★ 重入库依赖 MinerU，且会重新解析全部 PDF（并发=1，需要若干分钟）")


async def reingest(only: str | None) -> int:
    from src.infra.milvus_client import escape_milvus_string, get_milvus_client
    from src.rag.ingestion.pipeline import DocMetadata, get_ingestion_pipeline

    client = get_milvus_client()
    pipeline = get_ingestion_pipeline(client)

    targets = [e for e in ENTRIES if not only or only in e.source]
    if not targets:
        print(f"[失败] 没有匹配 {only!r} 的条目")
        return 1

    existing = _milvus_doc_names()
    ok, failed = 0, 0
    for e in targets:
        src = PDF_DIR / e.source
        if not src.exists():
            print(f"[跳过] 源文件缺失: {src}")
            failed += 1
            continue

        # 删除旧名文档（doc_id 由名字决定，改名等于换文档）
        old_id = doc_id_of(e.source)
        if existing.get(e.source):
            await asyncio.to_thread(
                client.delete, collection_name="alm_docs",
                filter=f'doc_id == "{escape_milvus_string(old_id)}"',
            )
            print(f"  已删除旧文档: {e.source}（doc_id={old_id}）")

        # 复制成正确名字的临时副本再入库，源文件不动
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / e.doc_name
            shutil.copy2(src, tmp)
            meta = DocMetadata(
                doc_name=e.doc_name,
                doc_type=e.doc_type,
                model_code=e.model_code,
                acl_roles=[r for r in e.acl_roles.split(",") if r],
            )
            print(f"\n▶ 入库 {e.doc_name} …")
            try:
                new_id = await pipeline.ingest(str(tmp), meta)
                ok += 1
                print(f"  ✓ 完成 doc_id={new_id}")
            except Exception as ex:
                failed += 1
                logger.exception(f"入库失败: {e.doc_name}: {ex}")
                print(f"  ✗ 失败: {type(ex).__name__}: {ex}")

    # 结果核对
    print("\n" + "=" * 76)
    print("重入库结果")
    print("=" * 76)
    after = _milvus_doc_names()
    print(f"成功 {ok} / 失败 {failed}；collection 现 {sum(after.values())} 块")
    for name, n in sorted(after.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {n} 块")

    # 页码覆盖率
    try:
        rows = await asyncio.to_thread(
            client.query, collection_name="alm_docs", filter="",
            output_fields=["doc_name", "page_number"], limit=16384,
            consistency_level="Strong",   # Bounded 下刚写入的行可能还不可见
        )
        total = len(rows)
        with_page = sum(1 for r in rows if isinstance(r.get("page_number"), int)
                        and r["page_number"] > 0)
        print(f"\n页码覆盖率: {with_page}/{total}")
        if with_page:
            by_doc: dict[str, list[int]] = {}
            for r in rows:
                by_doc.setdefault(r["doc_name"], []).append(r.get("page_number") or 0)
            for name, pgs in sorted(by_doc.items()):
                ok = sum(1 for v in pgs if v > 0)
                print(f"  {'✓' if ok == len(pgs) else '△'} {name}: {ok}/{len(pgs)} 有页码"
                      f"（范围 p{min(p for p in pgs if p) if ok else 0}~"
                      f"p{max(pgs) if pgs else 0}）")
        else:
            print("  ★ 仍为 0：检查 MinerU 是否返回 content_list（return_content_list=true）")
    except Exception as ex:
        print(f"页码核对失败: {ex}")

    print("\n★ 若文档名有变化，记得同步更新 scripts/eval_dataset_draft.csv 的 relevant_doc_ids")
    return 0 if failed == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="重入库（含文档名修正）")
    ap.add_argument("--list", action="store_true", help="只看计划，不动数据")
    ap.add_argument("--only", default=None, help="只处理源文件名包含该子串的条目")
    args = ap.parse_args()
    if args.list:
        show_plan()
        return 0
    return asyncio.run(reingest(args.only))


if __name__ == "__main__":
    raise SystemExit(main())
