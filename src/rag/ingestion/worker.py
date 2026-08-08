# ============================================================
# 入库后台 worker（独立进程）
#
# 启动：python -m src.rag.ingestion.worker
#
# 消费 Redis Stream（alm_ingest:jobs）：
#   取消息 → MinIO 下载原始文件 → process_ingestion（解析/切片/嵌入/索引）
#   → 更新 doc_ingest_jobs 进度 → XACK
#
# ★ 重试：处理失败按 INGEST_MAX_RETRIES 次退避重试，仍失败则标记 failed；
# ★ 未 XACK 的消息在崩溃后由同组其他 consumer 重新认领（XREADGROUP PEL）；
# ★ 幂等：pipeline 先删后插，重复处理不会产生脏数据。
# ============================================================

from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile
import time
from pathlib import Path

from loguru import logger

from src.core.config import get_settings
from src.core.metrics import INGESTION_JOBS
from src.infra.db import AsyncSessionLocal
from src.infra.minio_client import download_file

settings = get_settings()


def _client():
    import redis.asyncio as aioredis

    from src.infra.redis_cache import redis_pool

    return aioredis.Redis(connection_pool=redis_pool)


def _decode(field):
    """decode_responses 开关下字段可能是 bytes 或 str，统一解码"""
    return field.decode("utf-8") if isinstance(field, bytes) else field


async def _mark_failed(job_id: str, error: str) -> None:
    async with AsyncSessionLocal() as db:
        try:
            from src.knowledge.model import DocIngestJob
            job = await db.get(DocIngestJob, int(job_id))
            if job:
                job.stage = "failed"
                job.error_msg = str(error)[:2000]
            await db.commit()
        except Exception:
            logger.exception("标记入库 job 失败时出错")


async def _do_ingest(payload: dict) -> str:
    """单次入库：下载 → 管线 → 更新 DB（每次调用开新 session，便于重试）。"""
    object_name = payload["object_name"]
    file_name = Path(payload["doc_name"]).name
    suffix = Path(file_name).suffix or ".pdf"

    # MinIO 下载
    data = await asyncio.to_thread(download_file, object_name)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        from src.knowledge.doc_ingestion import process_ingestion

        async with AsyncSessionLocal() as db:
            try:
                doc_id = await process_ingestion(
                    db,
                    job_id=payload["job_id"],
                    file_path=tmp_path,
                    doc_name=payload["doc_name"],
                    doc_type=payload["doc_type"],
                    category=payload.get("category", ""),
                    business_line=payload.get("business_line", ""),
                    model_code=payload.get("model_code", ""),
                    chunk_strategy=payload.get("chunk_strategy", "fixed"),
                    parser=payload.get("parser", "mineru"),
                )
                await db.commit()
            except Exception:
                await db.rollback()
                raise
        return doc_id
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


async def _run_ingest_with_retry(payload: dict) -> None:
    last_exc: Exception | None = None
    for attempt in range(settings.INGEST_MAX_RETRIES + 1):
        try:
            await _do_ingest(payload)
            return
        except Exception as e:
            last_exc = e
            logger.warning(f"入库处理失败 attempt={attempt + 1}/{settings.INGEST_MAX_RETRIES + 1}: {e}")
            if attempt < settings.INGEST_MAX_RETRIES:
                await asyncio.sleep(min(8.0, 1.0 * (2 ** attempt)) * (0.5 + 0.5 * (time.time() % 1)))
    raise last_exc


async def process_one(client, msg_id: str, fields: dict) -> None:
    """处理单条消息：成功/最终失败都 XACK。"""
    try:
        payload = json.loads(_decode(fields.get("payload", "{}")))
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"消息载荷解析失败: {e}，直接 XACK 丢弃")
        await client.xack(settings.INGEST_STREAM, settings.INGEST_CONSUMER_GROUP, msg_id)
        return

    job_id = payload.get("job_id", "")
    INGESTION_JOBS.labels(status="started").inc()

    try:
        await _run_ingest_with_retry(payload)
    except Exception as e:
        logger.error(f"入库任务最终失败 job_id={job_id}: {e}")
        await _mark_failed(job_id, str(e))
        INGESTION_JOBS.labels(status="failed").inc()
    else:
        INGESTION_JOBS.labels(status="succeeded").inc()
    finally:
        await client.xack(settings.INGEST_STREAM, settings.INGEST_CONSUMER_GROUP, msg_id)


async def run_forever() -> None:
    from src.rag.ingestion.queue import ensure_group

    client = _client()
    await ensure_group(client)
    consumer = f"{socket.gethostname()}-{os.getpid()}"
    logger.info(f"入库 worker 启动: group={settings.INGEST_CONSUMER_GROUP} consumer={consumer}")

    while True:
        try:
            resp = await client.xreadgroup(
                settings.INGEST_CONSUMER_GROUP,
                consumer,
                {settings.INGEST_STREAM: ">"},
                count=10,
                block=5000,
            )
        except Exception as e:
            logger.warning(f"XREADGROUP 异常，2s 后重试: {e}")
            await asyncio.sleep(2)
            continue

        if not resp:
            continue

        for _stream, messages in resp:
            for msg_id, fields in messages:
                try:
                    await process_one(client, msg_id, fields)
                except Exception:
                    logger.exception("处理入库消息时发生未捕获异常")
                    await asyncio.sleep(1)


def main() -> None:
    asyncio.run(run_forever())


if __name__ == "__main__":
    main()
