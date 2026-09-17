# ============================================================
# MinIO 对象存储 — 原始文档留存
#
# bucket = knowledge-docs（与 P1 的 alm-reports 隔离）
# ============================================================

import io
import json

from loguru import logger
from minio import Minio

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


_bucket_ready = False


def ensure_bucket_exists() -> None:
    """确保 bucket 存在并设置访问策略。进程内只执行一次（旧实现每张图片上传都跑两轮 RTT）。"""
    global _bucket_ready
    if _bucket_ready:
        return
    exists = _minio_client.bucket_exists(settings.MINIO_BUCKET)
    logger.info(f"检查 MinIO bucket {settings.MINIO_BUCKET} 是否存在：{exists}")
    if not exists:
        _minio_client.make_bucket(bucket_name=settings.MINIO_BUCKET)
        logger.info(f"创建 MinIO bucket: {settings.MINIO_BUCKET}")

    # 公共读策略：检索命中的图片 URL 是 http://endpoint/bucket/obj 的裸地址，
    # 前端 <img> 需匿名 GET 才能直接显示。
    # ★ 生产建议 MINIO_PUBLIC_READ=false 关闭，改走预签名 URL 或鉴权代理。
    if settings.MINIO_PUBLIC_READ:
        try:
            _minio_client.set_bucket_policy(
                settings.MINIO_BUCKET,
                json.dumps({
                    "Version": "2012-10-17",
                    "Statement": [{
                        "Effect": "Allow",
                        "Principal": {"AWS": ["*"]},
                        "Action": ["s3:GetObject"],
                        "Resource": [f"arn:aws:s3:::{settings.MINIO_BUCKET}/*"],
                    }],
                }),
            )
        except Exception as e:
            logger.warning(f"设置 bucket 公共读策略失败（图片可能无法直接访问）: {e}")

    _bucket_ready = True


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


def delete_prefix(prefix: str) -> int:
    """删除指定前缀下所有对象（文档删除时清理原始文件与 images/{doc_id}/ 目录）。

    尽力而为：单个对象删除失败只记日志，不中断。返回成功删除数。"""
    count = 0
    try:
        objects = list(_minio_client.list_objects(settings.MINIO_BUCKET, prefix=prefix, recursive=True))
    except Exception as e:
        logger.warning(f"列举 MinIO 对象失败 prefix={prefix}: {e}")
        return 0
    for obj in objects:
        try:
            _minio_client.remove_object(settings.MINIO_BUCKET, obj.object_name)
            count += 1
        except Exception as e:
            logger.warning(f"删除 MinIO 对象失败 {obj.object_name}: {e}")
    return count
