# ============================================================
# 文档入库服务层 — 拆成"入队"与"处理"两段，配合 Redis Stream worker
#
#   enqueue（API 进程）：
#     compute_doc_id → create_ingest_record（落 queued 记录）→ 投递 Redis Stream
#   process（worker 进程）：
#     process_ingestion（跑管线 + 更新既有 job 进度/结果）
#
# 幂等：doc_id = md5(doc_name)[:16]，同名重复上传覆盖旧数据。
# ============================================================

import hashlib

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.infra.milvus_client import get_milvus_client
from src.knowledge.model import DocIngestJob, KnowledgeDoc
from src.rag.ingestion.pipeline import DocMetadata, get_ingestion_pipeline


def compute_doc_id(doc_name: str) -> str:
    """doc_id = md5(doc_name)[:16]，确定性幂等键"""
    return hashlib.md5(doc_name.encode()).hexdigest()[:16]


async def create_ingest_record(
    db: AsyncSession,
    doc_name: str,
    doc_type: str,
    category: str = "",
    business_line: str = "",
    model_code: str = "",
    chunk_strategy: str = "fixed",
    parser: str = "mineru",
) -> tuple[str, str]:
    """幂等 upsert KnowledgeDoc + 创建 queued 的 DocIngestJob。返回 (doc_id, job_id)。"""
    doc_id = compute_doc_id(doc_name)

    result = await db.execute(select(KnowledgeDoc).where(KnowledgeDoc.doc_name == doc_name))
    existing = result.scalar_one_or_none()
    if existing:
        existing.doc_id = doc_id
        existing.doc_type = doc_type
        existing.category = category or None
        existing.business_line = business_line or None
        existing.model_code = model_code or None
        existing.chunk_strategy = chunk_strategy
        existing.status = "queued"
        await db.flush()
    else:
        doc = KnowledgeDoc(
            doc_id=doc_id, doc_name=doc_name, doc_type=doc_type,
            category=category or None,
            business_line=business_line or None,
            model_code=model_code or None,
            chunk_strategy=chunk_strategy,
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
    chunk_strategy: str = "fixed",
    parser: str = "mineru",
) -> str:
    """worker 调用：跑完整管线，更新既有 job 的进度与 KnowledgeDoc 结果。

    失败抛出异常（worker 负责标记 failed / 重试 / 记指标）。
    """
    milvus = get_milvus_client()
    pipeline = get_ingestion_pipeline(milvus)
    from src.rag.config import ChunkingConfig
    pipeline.chunking_config = ChunkingConfig(strategy=chunk_strategy)
    pipeline.parser.parser = parser

    meta = DocMetadata(
        doc_name=doc_name, doc_type=doc_type,
        category=category, business_line=business_line, model_code=model_code,
    )

    # 标记运行中（stage=parse 起点）
    await _set_job(db, job_id, stage="parse", progress=5)

    result_doc_id = await pipeline.ingest(file_path, meta)

    # 更新 KnowledgeDoc 结果
    result = await db.execute(select(KnowledgeDoc).where(KnowledgeDoc.doc_id == result_doc_id))
    doc = result.scalar_one_or_none()
    if doc:
        doc.status = "indexed"
        doc.chunk_count = _count_chunks(result_doc_id)

    # 更新 job 完成
    await _set_job(db, job_id, stage="index", progress=100, doc_id=result_doc_id)

    logger.info(f"文档入库完成: {doc_name} → doc_id={result_doc_id}")
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


def _count_chunks(doc_id: str) -> int:
    """查 Milvus 统计 chunk 数"""
    milvus = get_milvus_client()
    results = milvus.query(
        collection_name="alm_docs",
        filter=f'doc_id == "{doc_id}"',
        output_fields=["id"],
    )
    return len(results)


async def delete_doc(doc_id: str, db: AsyncSession | None = None) -> None:
    """删除文档（Milvus + PG 元数据）"""
    milvus = get_milvus_client()
    milvus.delete(collection_name="alm_docs", filter=f'doc_id == "{doc_id}"')

    if db:
        result = await db.execute(
            select(KnowledgeDoc).where(KnowledgeDoc.doc_id == doc_id)
        )
        doc = result.scalar_one_or_none()
        if doc:
            doc.status = "deleted"

    logger.info(f"文档已删除: doc_id={doc_id}")
