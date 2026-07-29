# ============================================================
# 文档入库服务层 — 调 IngestionPipeline，落元数据表
# ============================================================

from pathlib import Path

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.infra.milvus_client import get_milvus_client
from src.knowledge.model import KnowledgeDoc, DocIngestJob
from src.rag.ingestion.pipeline import DocMetadata, get_ingestion_pipeline


async def ingest_and_index(
    file_path: str,
    doc_name: str,
    doc_type: str,
    category: str = "",
    business_line: str = "",
    model_code: str = "",
    chunk_strategy: str = "fixed",
    parser: str = "mineru",
    db: AsyncSession | None = None,
) -> str:
    """
    上传文档 → 解析 → 切片 → 嵌入 → 入库 Milvus → 写元数据表。

    幂等：相同 doc_name 重复上传会覆盖旧数据。
    """
    milvus = get_milvus_client()
    pipeline = get_ingestion_pipeline(milvus)

    # 更新 chunking 策略
    from src.rag.config import ChunkingConfig
    pipeline.chunking_config = ChunkingConfig(strategy=chunk_strategy)
    pipeline.parser.parser = parser

    file_name = Path(file_path).name
    meta = DocMetadata(
        doc_name=doc_name,
        doc_type=doc_type,
        category=category,
        business_line=business_line,
        model_code=model_code,
    )

    # 幂等：同名文档复用旧记录，避免 doc_id 唯一索引冲突
    doc_id, job_id = None, None
    if db:
        result = await db.execute(
            select(KnowledgeDoc).where(KnowledgeDoc.doc_name == doc_name)
        )
        existing = result.scalar_one_or_none()
        if existing:
            existing.doc_type = doc_type
            existing.category = category or None
            existing.business_line = business_line or None
            existing.model_code = model_code or None
            existing.chunk_strategy = chunk_strategy
            existing.status = "ingesting"
            await db.flush()
            doc_id = str(existing.id)
        else:
            doc = KnowledgeDoc(
                doc_id="", doc_name=doc_name, doc_type=doc_type,
                category=category or None,
                business_line=business_line or None,
                model_code=model_code or None,
                chunk_strategy=chunk_strategy,
                status="ingesting",
            )
            db.add(doc)
            await db.flush()
            doc_id = str(doc.id)

        job = DocIngestJob(doc_id="", stage="parse", progress=0, parser=parser)
        db.add(job)
        await db.flush()
        job_id = str(job.id)

    try:
        result_doc_id = await pipeline.ingest(file_path, meta)
    except Exception as e:
        logger.error(f"入库失败: {doc_name} - {e}")
        if db and job_id:
            job = await db.get(DocIngestJob, int(job_id))
            if job:
                job.error_msg = str(e)
        raise

    # 更新元数据
    if db and doc_id:
        doc = await db.get(KnowledgeDoc, int(doc_id))
        if doc:
            doc.doc_id = result_doc_id
            doc.status = "indexed"
            doc.chunk_count = _count_chunks(result_doc_id)
        if job_id:
            job = await db.get(DocIngestJob, int(job_id))
            if job:
                job.doc_id = result_doc_id
                job.stage = "index"
                job.progress = 100

    logger.info(f"文档入库完成: {doc_name} → doc_id={result_doc_id}")
    return result_doc_id


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
