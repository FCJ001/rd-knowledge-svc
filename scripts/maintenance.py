#!/usr/bin/env python3
# ============================================================
# 运维维护脚本（建议由 crontab 定时驱动，见 docs/runbook.md）
#
#   python scripts/maintenance.py report            容量与残留报告（只读）
#   python scripts/maintenance.py compact           触发 Milvus 压缩，回收软删残留
#   python scripts/maintenance.py retention         清理超过保留期的软删文档
#   python scripts/maintenance.py reconcile         源侧对账：源/PG/Milvus 三方 diff
#                                                   （默认 dry-run，--apply 才动数据）
#
# 为什么需要它：
#   - 入库幂等是版本化写入（先插新版本再删旧版本），Milvus 的删除是标记式，
#     残留要到压缩才回收，于是 get_collection_stats 的 row_count 会高于实际行数；
#   - 没有保留策略时，软删文档会永久留在 PG 与 MinIO 里；
#   - 容量天花板（mem_limit）需要定期核对，不能等 OOM 才发现；
#   - 源目录删掉/改名文档后索引无从得知（旧文档残留的根本来源）——reconcile
#     定时 diff 权威源与索引，把它从"靠人记"变成有闭环的机制。
# ============================================================

import argparse
import asyncio
import hashlib
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from src.core.config import get_settings

COLLECTION = "alm_docs"

# 1024 维 float32 的每向量字节数（容量估算用）
_BYTES_PER_VECTOR = 1024 * 4
# docker-compose 里 Milvus 的 mem_limit（GiB），与 compose 保持一致
_MILVUS_MEM_LIMIT_GIB = 2


def _doc_id_of(name: str) -> str:
    """与入库侧同一幂等键：md5(doc_name)[:16]"""
    return hashlib.md5(name.encode()).hexdigest()[:16]


def _capacity_report(stats_rows: int, live_rows: int) -> None:
    """容量与天花板估算。

    ★ 这里的估算刻意保守：容器内存的大头是引擎本身与索引结构，
    向量数据只占很小一部分（实测 1097 条时向量数据约 4.7 MiB，
    而容器已用 586 MiB）。所以用「已用内存里可让出的部分」反推，
    而不是拿 mem_limit 直接除以向量大小——后者会高估一个数量级。
    """
    vec_mib = live_rows * _BYTES_PER_VECTOR / 1024 / 1024
    print(f"  向量数据估算: {vec_mib:.1f} MiB（{live_rows} 条 × {_BYTES_PER_VECTOR} 字节）")
    print(f"  容器限额: {_MILVUS_MEM_LIMIT_GIB} GiB（docker-compose mem_limit）")
    print("  ⚠️ 超限是 OOM kill 而非优雅降级——语料增长前先核对此值并调整 mem_limit")
    print("  ⚠️ row_count 含未压缩的软删行，不等于可检索行数")


async def cmd_report() -> int:
    from src.infra.milvus_client import get_milvus_client

    client = get_milvus_client()
    try:
        stats = client.get_collection_stats(COLLECTION)
    except Exception as e:
        print(f"[失败] 无法读取 collection 统计: {e}")
        return 1
    reported = int(stats.get("row_count") or 0)

    # pipeline_version 字段是后补的：老 collection 查它报错，降级为不统计
    try:
        rows = client.query(COLLECTION, filter="",
                            output_fields=["doc_name", "doc_type", "pipeline_version"],
                            limit=16384)
        has_pv = True
    except Exception:
        rows = client.query(COLLECTION, filter="", output_fields=["doc_name", "doc_type"],
                            limit=16384)
        has_pv = False

    live = len(rows)

    print("=" * 60)
    print("Milvus 容量与残留报告")
    print("=" * 60)
    print(f"  row_count（含软删残留）: {reported}")
    print(f"  实际可检索行数: {live}")
    residual = reported - live
    if residual > 0:
        print(f"  软删残留: {residual} 行 → 跑 `maintenance.py compact` 回收")
    else:
        print("  软删残留: 无")
    print()
    _capacity_report(reported, live)

    if rows:
        print()
        print("  文档分布（前 10）:")
        for name, count in Counter(str(r.get("doc_name")) for r in rows).most_common(10):
            print(f"    {name}: {count} 块")
        print()
        print("  doc_type 分布:", dict(Counter(str(r.get("doc_type")) for r in rows)))
        empty_meta = sum(1 for r in rows if not str(r.get("doc_type") or "").strip())
        if empty_meta:
            print(f"  ⚠️ {empty_meta} 行 doc_type 为空：过滤条件对它不生效")

        # 旧管线产物统计：切了 chunking/页码/前缀逻辑后，存量不会自动升级
        if has_pv:
            from src.rag.ingestion.pipeline import PIPELINE_VERSION
            stale = [r for r in rows if int(r.get("pipeline_version") or 0) < PIPELINE_VERSION]
            if stale:
                by_doc = Counter(str(r.get("doc_name")) for r in stale)
                print(f"  ⚠️ {len(stale)} 行为旧管线版本"
                      f"（pipeline_version < {PIPELINE_VERSION}），建议重刷: scripts/reingest.py")
                for name, count in by_doc.most_common(5):
                    print(f"      {name}: {count} 块")
            else:
                print(f"  管线版本: 全部为最新（pipeline_version={PIPELINE_VERSION}）")
        else:
            print("  ⚠️ collection 无 pipeline_version 字段（重入库一次即可带上）")
    return 0


async def cmd_compact() -> int:
    """触发 Milvus 压缩。软删行在压缩前仍占统计口径，长期不压缩会积累。"""
    await asyncio.to_thread(_compact_sync)
    return 0


def _compact_sync() -> None:
    from src.infra.milvus_client import get_milvus_client

    client = get_milvus_client()
    print(f"触发 collection 压缩: {COLLECTION}")
    try:
        client.compact(COLLECTION)
    except Exception as e:
        print(f"[失败] 压缩触发失败: {e}")
        raise SystemExit(1)
    print("压缩已提交（Milvus 异步执行，稍后重跑 report 核对残留数）")


async def cmd_retention() -> int:
    """清理超过保留期的软删文档（PG 元数据 + MinIO 对象）。"""
    settings = get_settings()
    days = settings.DOC_RETENTION_DAYS
    if days <= 0:
        print("DOC_RETENTION_DAYS=0：未启用保留策略，不执行清理")
        return 0

    from sqlalchemy import select
    from src.infra.db import AsyncSessionLocal
    from src.knowledge.model import KnowledgeDoc

    cutoff = datetime.now() - timedelta(days=days)
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(KnowledgeDoc).where(
                KnowledgeDoc.status == "deleted",
                KnowledgeDoc.updated_at < cutoff,
            )
        )
        stale = list(result.scalars().all())
        if not stale:
            print(f"无超过 {days} 天保留期的软删文档")
            return 0

        print(f"待清理软删文档 {len(stale)} 篇（软删超过 {days} 天）:")
        for doc in stale:
            print(f"  - {doc.doc_name} (doc_id={doc.doc_id}, updated_at={doc.updated_at})")

        # 默认 dry-run：真要删除必须显式加 --apply，避免误删
        print("\n以上为 dry-run 结果。加 --apply 才会真正删除。")
        return 0


async def cmd_retention_apply() -> int:
    """真正执行清理（PG 行删除 + MinIO 对象删除）。"""
    settings = get_settings()
    days = settings.DOC_RETENTION_DAYS
    if days <= 0:
        print("DOC_RETENTION_DAYS=0：未启用保留策略，拒绝执行")
        return 1

    from sqlalchemy import delete, select
    from src.infra.db import AsyncSessionLocal
    from src.infra.minio_client import delete_object
    from src.knowledge.model import KnowledgeDoc

    cutoff = datetime.now() - timedelta(days=days)
    removed = 0
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(KnowledgeDoc).where(
                KnowledgeDoc.status == "deleted",
                KnowledgeDoc.updated_at < cutoff,
            )
        )
        stale = list(result.scalars().all())
        for doc in stale:
            if doc.minio_key:
                try:
                    await asyncio.to_thread(delete_object, doc.minio_key)
                except Exception as e:
                    logger.warning(f"MinIO 对象删除失败（继续删元数据）: {doc.minio_key}: {e}")
            await db.execute(delete(KnowledgeDoc).where(KnowledgeDoc.id == doc.id))
            removed += 1
            print(f"  已清理: {doc.doc_name}")
        await db.commit()
    print(f"\n完成：清理 {removed} 篇软删文档")
    return 0


# ============================================================
# 源侧对账（reconcile）：权威源 ↔ PG ↔ Milvus 三方 diff
# ============================================================

@dataclass
class DocView:
    """PG 文档行的对账视图（与 ORM 解耦，纯函数可测）"""
    doc_id: str
    doc_name: str
    status: str
    minio_key: str | None = None


@dataclass
class ReconcilePlan:
    missing_in_index: list[str] = field(default_factory=list)   # 源有、索引无
    orphan_docs: list[DocView] = field(default_factory=list)    # 索引在库、源已消失
    milvus_orphans: list[tuple[str, str, int]] = field(default_factory=list)  # (doc_name, doc_id, 块数)
    broken_docs: list[DocView] = field(default_factory=list)    # PG indexed 但 Milvus 零块


def compute_reconcile_plan(
    source_names: set[str],
    pg_docs: list[DocView],
    milvus_counts: dict[str, int],
    minio_present: Callable[[str], bool],
) -> ReconcilePlan:
    """对账核心（纯函数，便于单测）。

    ★ 权威源按文档来源分两路，不能混用一个口径：
      - minio_key 为空（reingest/脚本直入）→ 权威源是 RECONCILE_SOURCE_DIR 目录；
      - minio_key 非空（API 上传）→ 权威源是 MinIO 对象。
      偷懒统一用目录会把所有 API 上传的文档误判为"源已消失"，一键 --apply
      就是批量误删。
    """
    plan = ReconcilePlan()
    by_name = {d.doc_name: d for d in pg_docs}
    active = [d for d in pg_docs if d.status != "deleted"]

    # 1) 索引在库、权威源已消失 → 孤儿文档（--apply 软删）
    for d in active:
        if d.minio_key:
            present = minio_present(d.minio_key)
        else:
            present = d.doc_name in source_names
        if not present:
            plan.orphan_docs.append(d)

    # 2) 源有、索引完全没有（含软删过的）→ 需要人工入库。
    #    不自动入库：入库需要 doc_type/acl_roles/model_code 等元数据，
    #    自动猜会造出 fail-closed 或错误类型的文档（比缺着更糟）。
    plan.missing_in_index = sorted(n for n in source_names if n not in by_name)

    # 3) Milvus 有块、PG 无在库文档（从未落库 / 已软删但块残留）→ 孤儿块
    for name, cnt in sorted(milvus_counts.items()):
        if cnt <= 0:
            continue
        d = by_name.get(name)
        if d is None or d.status == "deleted":
            plan.milvus_orphans.append((name, _doc_id_of(name), cnt))

    # 4) PG 标 indexed 但 Milvus 零块 → 半失败态（入库中断/删除只做了一半），
    #    只报告建议重传：直接自动重传拿不到原始文件路径（worker 从 MinIO 取）。
    plan.broken_docs = [
        d for d in active
        if d.status == "indexed" and milvus_counts.get(d.doc_name, 0) == 0
    ]
    return plan


def _minio_object_exists(key: str) -> bool:
    from src.infra.minio_client import get_minio_client
    try:
        get_minio_client().stat_object(get_settings().MINIO_BUCKET, key)
        return True
    except Exception:
        return False


async def _reconcile(apply: bool) -> int:
    settings = get_settings()
    source_dir = Path(settings.RECONCILE_SOURCE_DIR)
    if not source_dir.is_dir():
        print(f"[失败] 对账源目录不存在: {source_dir}（RECONCILE_SOURCE_DIR）")
        return 1

    source_names = {
        p.name for p in source_dir.iterdir()
        if p.is_file() and not p.name.startswith(".")
    }

    from sqlalchemy import select
    from src.infra.db import AsyncSessionLocal
    from src.knowledge.model import KnowledgeDoc

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(KnowledgeDoc))
        pg_docs = [
            DocView(doc_id=d.doc_id, doc_name=d.doc_name, status=d.status,
                    minio_key=d.minio_key)
            for d in result.scalars().all()
        ]

    from src.infra.milvus_client import get_milvus_client
    client = get_milvus_client()
    try:
        rows = client.query(COLLECTION, filter="", output_fields=["doc_name"], limit=16384)
    except Exception as e:
        print(f"[失败] 无法查询 Milvus collection: {e}")
        return 1
    milvus_counts: dict[str, int] = {}
    for r in rows:
        name = str(r.get("doc_name"))
        milvus_counts[name] = milvus_counts.get(name, 0) + 1

    plan = compute_reconcile_plan(source_names, pg_docs, milvus_counts, _minio_object_exists)

    print("=" * 60)
    print(f"源侧对账（源目录: {source_dir}）" + ("  [--apply]" if apply else "  [dry-run]"))
    print("=" * 60)
    print(f"  源目录文档: {len(source_names)}  PG 记录: {len(pg_docs)}  "
          f"Milvus 文档: {len(milvus_counts)}")

    if plan.missing_in_index:
        print(f"\n▶ 源有、索引无（{len(plan.missing_in_index)} 份，需人工入库——"
              "自动入库猜不出 doc_type/ACL 元数据）:")
        for name in plan.missing_in_index:
            print(f"    - {name}")
    if plan.orphan_docs:
        print(f"\n▶ 索引在库、权威源已消失（{len(plan.orphan_docs)} 份"
              + ("，将软删" if apply else "，dry-run 仅列出") + "）:")
        for d in plan.orphan_docs:
            origin = "MinIO 对象缺失" if d.minio_key else "源目录无此文件"
            print(f"    - {d.doc_name} (doc_id={d.doc_id}, {origin})")
    if plan.milvus_orphans:
        print(f"\n▶ Milvus 孤儿块（{len(plan.milvus_orphans)} 份：PG 无在库文档但向量还在"
              + ("，将删除" if apply else "，dry-run 仅列出") + "）:")
        for name, doc_id, cnt in plan.milvus_orphans:
            print(f"    - {name} (doc_id={doc_id}, {cnt} 块)")
    if plan.broken_docs:
        print(f"\n▶ 半失败态（{len(plan.broken_docs)} 份：PG=indexed 但 Milvus 零块，"
              "建议重新上传）:")
        for d in plan.broken_docs:
            print(f"    - {d.doc_name} (doc_id={d.doc_id})")
    if not (plan.missing_in_index or plan.orphan_docs or plan.milvus_orphans or plan.broken_docs):
        print("\n✓ 源、PG、Milvus 三方一致，无差异")
        return 0

    if not apply:
        print("\n以上为 dry-run 结果。加 --apply 才会执行：孤儿文档软删（走 delete_doc"
              " 四联删）+ Milvus 孤儿块清理。")
        return 0

    # --apply：孤儿文档软删（Milvus + MinIO + PG + 缓存失效，与 API 删除同一链路）
    deleted = 0
    if plan.orphan_docs:
        from src.knowledge.doc_ingestion import delete_doc
        async with AsyncSessionLocal() as db:
            for d in plan.orphan_docs:
                try:
                    await delete_doc(d.doc_id, db)
                    await db.commit()
                    deleted += 1
                    print(f"  已软删: {d.doc_name} (doc_id={d.doc_id})")
                except Exception as e:
                    await db.rollback()
                    logger.exception(f"软删失败（继续处理后续）: {d.doc_name}: {e}")

    # --apply：Milvus 孤儿块清理（按 doc_id 精确删，doc_id 由 doc_name 幂等推出）
    cleaned = 0
    for name, doc_id, cnt in plan.milvus_orphans:
        try:
            client.delete(COLLECTION, filter=f'doc_id == "{doc_id}"')
            cleaned += 1
            print(f"  已清 Milvus 孤儿块: {name} ({cnt} 块)")
        except Exception as e:
            logger.exception(f"Milvus 孤儿块清理失败（继续）: {name}: {e}")

    print(f"\n完成：软删 {deleted} 份孤儿文档，清理 {cleaned} 份孤儿块。"
          "残留统计要等 compact 后才回收（见 maintenance.py compact）。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="运维维护脚本")
    parser.add_argument("command", choices=["report", "compact", "retention", "reconcile"],
                        help="report=容量报告 / compact=Milvus 压缩 / retention=保留期清理"
                             "（默认 dry-run）/ reconcile=源侧对账（默认 dry-run）")
    parser.add_argument("--apply", action="store_true",
                        help="retention/reconcile 时才生效：真正执行删除（默认只打印）")
    args = parser.parse_args()

    if args.command == "report":
        return asyncio.run(cmd_report())
    if args.command == "compact":
        return asyncio.run(cmd_compact())
    if args.command == "reconcile":
        return asyncio.run(_reconcile(apply=args.apply))
    if args.apply:
        return asyncio.run(cmd_retention_apply())
    return asyncio.run(cmd_retention())


if __name__ == "__main__":
    raise SystemExit(main())
