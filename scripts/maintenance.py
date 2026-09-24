#!/usr/bin/env python3
# ============================================================
# 运维维护脚本（建议由 crontab 定时驱动，见 docs/runbook.md）
#
#   python scripts/maintenance.py report            容量与残留报告（只读）
#   python scripts/maintenance.py compact           触发 Milvus 压缩，回收软删残留
#   python scripts/maintenance.py retention         清理超过保留期的软删文档
#
# 为什么需要它：
#   - 入库幂等是「先删同 doc_id 再插」，Milvus 的删除是标记式，
#     残留要到压缩才回收，于是 get_collection_stats 的 row_count 会高于实际行数；
#   - 没有保留策略时，软删文档会永久留在 PG 与 MinIO 里；
#   - 容量天花板（mem_limit）需要定期核对，不能等 OOM 才发现。
# ============================================================

import argparse
import asyncio
import sys
from collections import Counter
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

    try:
        rows = client.query(COLLECTION, filter="", output_fields=["doc_name", "doc_type"],
                            limit=16384)
    except Exception as e:
        print(f"[失败] 无法查询 collection: {e}")
        return 1
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


def main() -> int:
    parser = argparse.ArgumentParser(description="运维维护脚本")
    parser.add_argument("command", choices=["report", "compact", "retention"],
                        help="report=容量报告 / compact=Milvus 压缩 / retention=保留期清理（默认 dry-run）")
    parser.add_argument("--apply", action="store_true",
                        help="retention 时才生效：真正执行删除（默认只打印）")
    args = parser.parse_args()

    if args.command == "report":
        return asyncio.run(cmd_report())
    if args.command == "compact":
        return asyncio.run(cmd_compact())
    if args.apply:
        return asyncio.run(cmd_retention_apply())
    return asyncio.run(cmd_retention())


if __name__ == "__main__":
    raise SystemExit(main())
