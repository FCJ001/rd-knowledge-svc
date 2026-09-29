# ============================================================
# 文档入库服务层 — 拆成"入队"与"处理"两段，配合 Redis Stream worker
#
#   enqueue（API 进程）：
#     compute_doc_id → create_ingest_record（落 queued 记录）→ 投递 Redis Stream
#   process（worker 进程）：
#     process_ingestion（跑管线 + 更新既有 job 进度/结果）
#
# 幂等：doc_id = md5(doc_name)[:16]，同名重复上传覆盖旧数据（版本化写入：
# 每次处理 version+1，先插新版本 chunk 再删旧版本，见 pipeline.ingest）。
# ============================================================

import asyncio
import hashlib
from datetime import datetime, time

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.infra.milvus_client import (
    assert_valid_doc_id,
    escape_milvus_string,
    get_milvus_client,
)
from src.knowledge.model import DocIngestJob, KnowledgeDoc
from src.rag.ingestion.pipeline import DocMetadata, get_ingestion_pipeline


def compute_doc_id(doc_name: str) -> str:
    """doc_id = md5(doc_name)[:16]，确定性幂等键"""
    return hashlib.md5(doc_name.encode()).hexdigest()[:16]


def parse_expire_date(raw: str | None) -> int:
    """失效日期 YYYY-MM-DD → 当日 23:59:59（本地时区）的 epoch 秒。

    空/None → 0（永久有效）；格式非法抛 ValueError（上传侧 400）。
    取当日末尾而非零点：同日设置的过期文档在当天仍可检索，次日起淡出。
    """
    if not raw or not raw.strip():
        return 0
    try:
        d = datetime.strptime(raw.strip(), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"expire_date 格式非法（应为 YYYY-MM-DD）: {raw!r}")
    return int(datetime.combine(d, time(23, 59, 59)).timestamp())


async def create_ingest_record(
    db: AsyncSession,
    doc_name: str,
    doc_type: str,
    category: str = "",
    business_line: str = "",
    model_code: str = "",
    acl_roles: str = "",
    chunk_strategy: str = "fixed",
    parser: str = "mineru",
    content_hash: str = "",
    expire_date: str = "",
) -> tuple[str, str]:
    """幂等 upsert KnowledgeDoc + 创建 queued 的 DocIngestJob。返回 (doc_id, job_id)。

    acl_roles：API 层已解析/校验过的可见角色逗号串（""=仅 admin），这里原样
    落库不再二次解析——解析点唯一，避免两处口径不一致。
    content_hash：上传文件 SHA-256（内容指纹，跳过未变更重传/查重复入库）。
    expire_date：失效日期 YYYY-MM-DD（""=永久有效），worker 处理时换算 expire_ts。
    """
    doc_id = compute_doc_id(doc_name)

    result = await db.execute(select(KnowledgeDoc).where(KnowledgeDoc.doc_name == doc_name))
    existing = result.scalar_one_or_none()
    if existing:
        existing.doc_id = doc_id
        existing.doc_type = doc_type
        existing.category = category or None
        existing.business_line = business_line or None
        existing.model_code = model_code or None
        existing.acl_roles = acl_roles
        existing.chunk_strategy = chunk_strategy
        existing.content_hash = content_hash or None
        existing.expire_date = expire_date or None
        existing.status = "queued"
        await db.flush()
    else:
        doc = KnowledgeDoc(
            doc_id=doc_id, doc_name=doc_name, doc_type=doc_type,
            category=category or None,
            business_line=business_line or None,
            model_code=model_code or None,
            acl_roles=acl_roles,
            chunk_strategy=chunk_strategy,
            content_hash=content_hash or None,
            expire_date=expire_date or None,
            status="queued",
        )
        db.add(doc)
        await db.flush()

    job = DocIngestJob(doc_id=doc_id, stage="queued", progress=0, parser=parser)
    db.add(job)
    await db.flush()

    logger.info(f"入库任务已入队: {doc_name} doc_id={doc_id} job_id={job.id}")
    return doc_id, str(job.id)


async def process_ingestion(
    db: AsyncSession,
    job_id: str,
    file_path: str,
    doc_name: str,
    doc_type: str,
    category: str = "",
    business_line: str = "",
    model_code: str = "",
    acl_roles: str = "",
    chunk_strategy: str = "fixed",
    parser: str = "mineru",
) -> str:
    """worker 调用：跑完整管线，更新既有 job 的进度与 KnowledgeDoc 结果。

    失败抛出异常（worker 负责标记 failed / 重试 / 记指标）。
    ★ 版本化写入：处理前取 KnowledgeDoc.version+1 传给管线。每次尝试（含
      失败重试）都递增——重试拿新版本号，绝不与在库/残留 chunk 撞主键；
      版本号只需单调，有空洞无碍。旧版本由管线在新版本写入成功后清理。
    """
    milvus = get_milvus_client()
    pipeline = get_ingestion_pipeline(milvus)
    from src.rag.config import ChunkingConfig
    pipeline.chunking_config = ChunkingConfig(strategy=chunk_strategy)
    pipeline.parser.parser = parser

    # 元数据以 PG 为单一事实源：expire_date → expire_ts，版本号前置递增
    result = await db.execute(select(KnowledgeDoc).where(KnowledgeDoc.doc_name == doc_name))
    doc = result.scalar_one_or_none()
    version = (doc.version or 0) + 1 if doc else 1
    expire_ts = parse_expire_date(doc.expire_date) if doc else 0
    if doc:
        doc.version = version
        await db.flush()

    meta = DocMetadata(
        doc_name=doc_name, doc_type=doc_type,
        category=category, business_line=business_line, model_code=model_code,
        acl_roles=[r for r in (acl_roles or "").split(",") if r],
        version=version,
        expire_ts=expire_ts,
    )

    # 标记运行中（stage=parse 起点）
    await _set_job(db, job_id, stage="parse", progress=5)

    result_doc_id = await pipeline.ingest(file_path, meta)

    # 更新 KnowledgeDoc 结果（doc 通常已在上面取到；兜底按 doc_id 再查一次）
    if doc is None:
        result = await db.execute(select(KnowledgeDoc).where(KnowledgeDoc.doc_id == result_doc_id))
        doc = result.scalar_one_or_none()
    if doc:
        doc.status = "indexed"
        doc.chunk_count = await _count_chunks(result_doc_id)

    # 更新 job 完成（★ 终态必须是 completed：前端/轮询方以 stage==completed 判定入库结束）
    await _set_job(db, job_id, stage="completed", progress=100, doc_id=result_doc_id)

    logger.info(f"文档入库完成: {doc_name} → doc_id={result_doc_id} v={version}")
    return result_doc_id


async def _set_job(
    db: AsyncSession,
    job_id: str,
    stage: str,
    progress: int,
    doc_id: str | None = None,
    error: str | None = None,
) -> None:
    job = await db.get(DocIngestJob, int(job_id))
    if job is None:
        logger.warning(f"入库 job 不存在: {job_id}")
        return
    if doc_id is not None:
        job.doc_id = doc_id
    job.stage = stage
    job.progress = progress
    if error is not None:
        job.error_msg = error


async def _count_chunks(doc_id: str) -> int:
    """查 Milvus 统计 chunk 数（同步 gRPC 调用放线程池，不阻塞事件循环）。

    ★ consistency_level=Strong：刚 insert 的数据在默认 Bounded 一致性下可能读不到，
      会把 chunk_count 落成 0。"""
    assert_valid_doc_id(doc_id)
    milvus = get_milvus_client()
    results = await asyncio.to_thread(
        milvus.query,
        collection_name="alm_docs",
        filter=f'doc_id == "{escape_milvus_string(doc_id)}"',
        output_fields=["id"],
        consistency_level="Strong",
    )
    return len(results)


async def delete_doc(doc_id: str, db: AsyncSession | None = None) -> None:
    """删除文档：Milvus 向量 + MinIO 原文件/图片 + PG 软删 + 查询缓存失效。

    ★ 旧实现只删 Milvus + 标记 PG，MinIO 对象与缓存残留：
      删除后 TTL 内缓存仍返回已删文档的答案，MinIO 对象永久泄漏存储。"""
    assert_valid_doc_id(doc_id)
    milvus = get_milvus_client()

    milvus_ok = True
    try:
        await asyncio.to_thread(
            milvus.delete,
            collection_name="alm_docs",
            filter=f'doc_id == "{escape_milvus_string(doc_id)}"',
        )
    except Exception as e:
        milvus_ok = False
        logger.error(f"Milvus 删除失败 doc_id={doc_id}: {e}")

    # MinIO：原始文件 + 图片目录，尽力而为（失败只记日志，不影响主流程）
    minio_deleted = 0
    if db:
        result = await db.execute(
            select(KnowledgeDoc).where(KnowledgeDoc.doc_id == doc_id)
        )
        doc = result.scalar_one_or_none()
        if doc:
            from src.infra.minio_client import delete_prefix
            if doc.minio_key:
                try:
                    await asyncio.to_thread(
                        _delete_minio_object_safe, doc.minio_key,
                    )
                    minio_deleted += 1
                except Exception as e:
                    logger.warning(f"MinIO 原文件删除失败 {doc.minio_key}: {e}")
            try:
                minio_deleted += await asyncio.to_thread(
                    delete_prefix, f"images/{doc_id}/",
                )
            except Exception as e:
                logger.warning(f"MinIO 图片目录删除失败 images/{doc_id}/: {e}")

            doc.status = "deleted"

    # 查询缓存：删除后立即失效，不等 TTL（300s 内继续返回已删文档的答案是事故）
    from src.core.cache import invalidate_search_cache
    await invalidate_search_cache()

    logger.info(
        f"文档已删除: doc_id={doc_id} milvus={'ok' if milvus_ok else 'FAILED'} "
        f"minio_objects={minio_deleted}"
    )


def _delete_minio_object_safe(object_name: str) -> None:
    from src.infra.minio_client import delete_object
    delete_object(object_name)
