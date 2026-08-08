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
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.base_schema import ResponseSchema
from src.core.deps import UserContext, get_current_user
from src.core.exceptions import ERR_INGEST_FAILED, BizException
from src.core.logger import logger
from src.core.rate_limit import check_rate_limit
from src.infra.db import get_db
from src.knowledge.doc_ingestion import delete_doc as delete_doc_service
from src.knowledge.doc_ingestion import ingest_and_index
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
    """上传文档并触发入库管线（解析→切片→嵌入→索引）"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名为空")

    # 保存到临时文件（1MB 分块流式写，大文件不整块驻留内存）
    suffix = Path(file.filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        while chunk := await file.read(1024 * 1024):
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        # 上传到 MinIO
        minio_key = None
        try:
            from src.core.config import get_settings
            from src.infra.minio_client import get_minio_client
            minio_client = get_minio_client()
            minio_key = f"{doc_type}/{model_code or 'common'}/{file.filename}"
            minio_client.fput_object(get_settings().MINIO_BUCKET, minio_key, tmp_path)
            logger.info(f"MinIO 上传完成: {minio_key}")
        except Exception as e:
            logger.warning(f"MinIO 上传失败（继续入库）: {e}")

        doc_id = await ingest_and_index(
            file_path=tmp_path,
            doc_name=file.filename,
            doc_type=doc_type,
            category=category,
            business_line=business_line,
            model_code=model_code,
            chunk_strategy=chunk_strategy,
            parser=parser,
            db=db,
        )

        # 更新 MinIO key
        if minio_key:
            result = await db.execute(
                select(KnowledgeDoc).where(KnowledgeDoc.doc_id == doc_id)
            )
            doc = result.scalar_one_or_none()
            if doc:
                doc.minio_key = minio_key

        await db.commit()
        return ResponseSchema(data=IngestResponse(
            doc_id=doc_id,
            doc_name=file.filename,
            status="indexed",
        ))

    except Exception as e:
        logger.exception(f"文档入库失败: {file.filename}")
        raise HTTPException(status_code=500, detail=f"入库失败: {e}")

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
