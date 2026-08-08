# ============================================================
# 文档入库 API
#
# POST   /api/v1/ingest/upload       上传文档并入库
# DELETE /api/v1/ingest/docs/{doc_id} 删除文档
# GET    /api/v1/ingest/jobs/{doc_id} 查询入库任务进度
# ============================================================

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.base_schema import ResponseSchema
from src.core.deps import UserContext, get_current_user
from src.core.logger import logger
from src.core.metrics import INGESTION_JOBS
from src.core.rate_limit import check_rate_limit
from src.infra.db import get_db
from src.knowledge.doc_ingestion import (
    create_ingest_record,
    delete_doc as delete_doc_service,
)
from src.knowledge.model import DocIngestJob, KnowledgeDoc

router = APIRouter(prefix="/api/v1/ingest", tags=["文档入库"])


# ── Request / Response models ────────────────────────────────────────────

class IngestResponse(BaseModel):
    doc_id: str
    doc_name: str
    status: str


class JobStatusResponse(BaseModel):
    doc_id: str
    stage: str
    progress: int
    parser: str
    error_msg: str | None


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/upload", response_model=ResponseSchema[IngestResponse])
async def upload_doc(
    file: UploadFile = File(...),
    doc_type: str = "spec_doc",
    category: str = "",
    business_line: str = "",
    model_code: str = "",
    chunk_strategy: str = "fixed",
    parser: str = "mineru",
    rate_limit: None = Depends(check_rate_limit),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """上传文档并投递异步入库任务（worker 进程处理解析/切片/嵌入/索引）"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名为空")

    # 保存到临时文件（1MB 分块流式写，大文件不整块驻留内存）
    suffix = Path(file.filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        while chunk := await file.read(1024 * 1024):
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        # 1. 上传到 MinIO（原始文档留存，worker 从中取）
        from src.core.config import get_settings
        from src.infra.minio_client import get_minio_client
        minio_client = get_minio_client()
        minio_key = f"{doc_type}/{model_code or 'common'}/{file.filename}"
        minio_client.fput_object(get_settings().MINIO_BUCKET, minio_key, tmp_path)
        logger.info(f"MinIO 上传完成: {minio_key}")

        # 2. 落 queued 记录（幂等 upsert KnowledgeDoc + DocIngestJob）
        doc_id, job_id = await create_ingest_record(
            db,
            doc_name=file.filename,
            doc_type=doc_type,
            category=category,
            business_line=business_line,
            model_code=model_code,
            chunk_strategy=chunk_strategy,
            parser=parser,
        )
        result = await db.execute(select(KnowledgeDoc).where(KnowledgeDoc.doc_id == doc_id))
        doc = result.scalar_one_or_none()
        if doc:
            doc.minio_key = minio_key

        # 3. 投递 Redis Stream（worker 消费）
        from src.rag.ingestion.queue import enqueue_ingest_job
        ok = await enqueue_ingest_job({
            "job_id": job_id,
            "doc_id": doc_id,
            "object_name": minio_key,
            "doc_name": file.filename,
            "doc_type": doc_type,
            "category": category,
            "business_line": business_line,
            "model_code": model_code,
            "chunk_strategy": chunk_strategy,
            "parser": parser,
        })
        if not ok:
            await db.rollback()
            raise HTTPException(status_code=503, detail="入库队列暂不可用，请稍后重试")

        await db.commit()
        INGESTION_JOBS.labels(status="queued").inc()
        return ResponseSchema(data=IngestResponse(
            doc_id=doc_id,
            doc_name=file.filename,
            status="queued",
        ))

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"文档上传失败: {file.filename}")
        raise HTTPException(status_code=500, detail=f"上传失败: {e}")

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@router.delete("/docs/{doc_id}", response_model=ResponseSchema[dict])
async def remove_doc(
    doc_id: str,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """删除文档（Milvus + PG 元数据）"""
    await delete_doc_service(doc_id, db)
    await db.commit()
    return ResponseSchema(data={"doc_id": doc_id, "status": "deleted"})


@router.get("/jobs/{doc_id}", response_model=ResponseSchema[JobStatusResponse])
async def get_ingest_job(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
):
    """查询文档入库任务进度"""
    result = await db.execute(
        select(DocIngestJob)
        .where(DocIngestJob.doc_id == doc_id)
        .order_by(DocIngestJob.id.desc())
        .limit(1)
    )
    job = result.scalar_one_or_none()
    if not job:
        return ResponseSchema(data=JobStatusResponse(
            doc_id=doc_id,
            stage="unknown",
            progress=0,
            parser="",
            error_msg="未找到入库任务",
        ))

    return ResponseSchema(data=JobStatusResponse(
        doc_id=str(job.doc_id),
        stage=str(job.stage),
        progress=int(job.progress),
        parser=str(job.parser),
        error_msg=job.error_msg,
    ))
