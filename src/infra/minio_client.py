# ============================================================
# MinIO 对象存储 — 原始文档留存
#
# bucket = knowledge-docs（与 P1 的 alm-reports 隔离）
# ============================================================

import io

from minio import Minio
from loguru import logger

from src.core.config import get_settings

settings = get_settings()

_minio_client = Minio(
    settings.MINIO_ENDPOINT,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=settings.MINIO_SECURE,
)


def get_minio_client() -> Minio:
    """FastAPI Depends 注入用"""
    return _minio_client


def ensure_bucket_exists() -> None:
    """确保 bucket 存在，不存在则创建。在应用启动时调用。"""
    exists = _minio_client.bucket_exists(settings.MINIO_BUCKET)
    logger.info(f"检查 MinIO bucket {settings.MINIO_BUCKET} 是否存在：{exists}")
    if not exists:
        _minio_client.make_bucket(bucket_name=settings.MINIO_BUCKET)
        logger.info(f"创建 MinIO bucket: {settings.MINIO_BUCKET}")


def upload_file(object_name: str, data: bytes,
                content_type: str = "application/octet-stream") -> str:
    """上传文件到 MinIO bucket"""
    _minio_client.put_object(
        bucket_name=settings.MINIO_BUCKET,
        object_name=object_name,
        data=io.BytesIO(data),
        content_type=content_type,
        length=len(data),
    )
    return object_name


def download_file(object_name: str) -> bytes:
    """从 MinIO bucket 下载文件"""
    resp = _minio_client.get_object(
        bucket_name=settings.MINIO_BUCKET,
        object_name=object_name,
    )
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()


def delete_object(object_name: str) -> None:
    """删除 MinIO bucket 中的文件"""
    _minio_client.remove_object(
        bucket_name=settings.MINIO_BUCKET,
        object_name=object_name,
    )
    logger.info(f"删除 MinIO 文件: {object_name}")
