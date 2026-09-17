# ============================================================
# 文档入库 API
#
# POST   /api/v1/ingest/upload       上传文档并入库
# DELETE /api/v1/ingest/docs/{doc_id} 删除文档
# GET    /api/v1/ingest/jobs/{doc_id} 查询入库任务进度
# ============================================================

import asyncio
import os
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.base_schema import ResponseSchema
from src.core.config import get_settings
from src.core.deps import UserContext, get_current_user
from src.core.logger import logger
from src.core.metrics import INGESTION_JOBS
from src.core.rate_limit import check_rate_limit
from src.infra.db import get_db
from src.knowledge.doc_ingestion import (
    create_ingest_record,
)
from src.knowledge.doc_ingestion import (
    delete_doc as delete_doc_service,
)
from src.knowledge.model import DocIngestJob, KnowledgeDoc

router = APIRouter(prefix="/api/v1/ingest", tags=["文档入库"])

# 路径参数 doc_id 白名单：字母/数字/下划线/连字符（doc_id 是 md5 hex[:16]）
_DOC_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _sanitize_filename(name: str) -> str:
    """清洗文件名：只保留白名单字符，防 ../ 越级与特殊字符破坏 MinIO key"""
    name = Path(name).name  # 去掉任何路径分量
    cleaned = re.sub(r"[^\w.\-\u4e00-\u9fff]", "_", name)  # 中英文/数字/点/横线/下划线
    return cleaned or "unnamed"


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
    settings = get_settings()
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名为空")

    # ★ 扩展名白名单：入库只吃文档类文件
    raw_name = Path(file.filename).name
    suffix = Path(raw_name).suffix.lower()
    allowed = [s.strip().lower() for s in settings.ALLOWED_UPLOAD_SUFFIXES.split(",") if s.strip()]
    if suffix and suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {suffix}，允许: {' '.join(allowed)}")

    doc_name = _sanitize_filename(raw_name)

    # 保存到临时文件（1MB 分块流式写 + 总量上限，防大文件磁盘耗尽）
    max_bytes = settings.UPLOAD_MAX_MB * 1024 * 1024
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".pdf") as tmp:
        written = 0
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(status_code=413, detail=f"文件超过大小上限 {settings.UPLOAD_MAX_MB}MB")
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        # 1. 上传到 MinIO（原始文档留存，worker 从中取）
        # ★ 同步上传放线程池：大文件上传期间不冻结事件循环
        from src.infra.minio_client import get_minio_client
        minio_client = get_minio_client()
        minio_key = f"{doc_type}/{model_code or 'common'}/{doc_name}"
        await asyncio.to_thread(
            minio_client.fput_object, settings.MINIO_BUCKET, minio_key, tmp_path,
        )
        logger.info(f"MinIO 上传完成: {minio_key} size={written}")

        # 2. 落 queued 记录（幂等 upsert KnowledgeDoc + DocIngestJob）
        doc_id, job_id = await create_ingest_record(
            db,
            doc_name=doc_name,
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

        # 3. 投递 Redis Stream（worker 消费；带 trace_id 便于跨进程链路关联）
        from src.core.logger import trace_id_var
        from src.rag.ingestion.queue import enqueue_ingest_job
        ok = await enqueue_ingest_job({
            "job_id": job_id,
            "doc_id": doc_id,
            "object_name": minio_key,
            "doc_name": doc_name,
            "doc_type": doc_type,
            "category": category,
            "business_line": business_line,
            "model_code": model_code,
            "chunk_strategy": chunk_strategy,
            "parser": parser,
            "trace_id": trace_id_var.get(),
        })
        if not ok:
            await db.rollback()
            # 入队失败补偿：删掉刚传的 MinIO 对象，不留孤儿
            try:
                from src.infra.minio_client import delete_object
                await asyncio.to_thread(delete_object, minio_key)
            except Exception as e:
                logger.warning(f"入队失败后清理 MinIO 对象失败 {minio_key}: {e}")
            raise HTTPException(status_code=503, detail="入库队列暂不可用，请稍后重试")

        await db.commit()
        INGESTION_JOBS.labels(status="queued").inc()
        return ResponseSchema(data=IngestResponse(
            doc_id=doc_id,
            doc_name=doc_name,
            status="queued",
        ))

    except HTTPException:
        raise
    except Exception:
        # ★ 原始异常只进日志；返回体不带底层细节（MinIO/网络信息不外泄）
        logger.exception(f"文档上传失败: {doc_name}")
        raise HTTPException(status_code=500, detail="上传失败，请稍后重试")

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
    """删除文档（Milvus + MinIO + PG 元数据 + 查询缓存）"""
    if not _DOC_ID_RE.match(doc_id or ""):
        raise HTTPException(status_code=400, detail="非法 doc_id")
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
