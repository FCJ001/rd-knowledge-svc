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
# ★ 崩溃恢复：工人带存活心跳（Redis SET EX），定期认领「空闲超阈值 且
#   原持有人心跳已失效」的 PEL 遗留消息——慢文档（心跳在续）不会被误抢，
#   文档不再卡死在 processing；
# ★ 对账兜底：queued 记录超过 INGEST_LOST_JOB_HOURS 仍无进展 → 标记 failed，
#   把 maxlen 裁剪 / worker 长期不可用造成的静默丢任务变成显式失败；
# ★ 优雅停机：SIGTERM/SIGINT 后停止取新消息，处理完在途消息再退出；
# ★ 幂等：pipeline 先删后插，重复处理不会产生脏数据。
# ============================================================

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import tempfile
import time
from datetime import timedelta
from pathlib import Path

from loguru import logger
from sqlalchemy import func, select

from src.core.config import get_settings
from src.core.logger import setup_logger
from src.core.metrics import INGEST_QUEUE_DEPTH, INGESTION_JOBS
from src.infra.db import AsyncSessionLocal
from src.infra.minio_client import download_file

settings = get_settings()

# 消息空闲多久才进入「疑似遗留」候选（与存活心跳配合判定，见 _claim_stale）。
# ★ 心跳兜底后，本阈值不再承担「必须大于单条最长处理时间」的约束——
#   它只决定「持有人死后多久才认领」，取值只影响恢复速度，不影响正确性。
PEL_MIN_IDLE_MS = settings.INGEST_PEL_MIN_IDLE_S * 1000
# PEL 回收 / 对账巡检周期
PEL_RECLAIM_INTERVAL_S = 60
# XREADGROUP 阻塞时长（必须小于 REDIS_SOCKET_TIMEOUT，否则每次读都超时）
READ_BLOCK_MS = 3000
# 工人存活心跳 TTL：主循环每轮刷新（轮次 ≤ READ_BLOCK_MS=3s），崩溃后
# 最多 TTL 过期才被判定失联，遗留消息才会被其他 worker 认领
ALIVE_TTL_S = 60


def _client():
    import redis.asyncio as aioredis

    from src.infra.redis_cache import redis_pool

    return aioredis.Redis(connection_pool=redis_pool)


def _decode(field):
    """decode_responses 开关下字段可能是 bytes 或 str，统一解码"""
    return field.decode("utf-8") if isinstance(field, bytes) else field


def _alive_key(owner: str) -> str:
    return f"{settings.INGEST_STREAM}:alive:{owner}"


async def _heartbeat(client, consumer: str) -> None:
    """刷新本工人存活心跳。失败不抛：下轮循环会再刷，TTL 60s 余量充足。"""
    try:
        await client.set(_alive_key(consumer), "1", ex=ALIVE_TTL_S)
    except Exception as e:
        logger.warning(f"刷新存活心跳失败（下轮重试）: {e}")


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
                    acl_roles=payload.get("acl_roles", ""),
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
    trace_hint = payload.get("trace_id") or "-"
    logger.info(f"开始处理入库消息 msg_id={msg_id} job_id={job_id} trace_id={trace_hint}")

    try:
        await _run_ingest_with_retry(payload)
    except Exception as e:
        logger.error(f"入库任务最终失败 job_id={job_id} trace_id={trace_hint}: {e}")
        await _mark_failed(job_id, str(e))
        INGESTION_JOBS.labels(status="failed").inc()
    else:
        INGESTION_JOBS.labels(status="succeeded").inc()
    finally:
        await client.xack(settings.INGEST_STREAM, settings.INGEST_CONSUMER_GROUP, msg_id)


async def _update_queue_metrics(client) -> None:
    """队列深度 / 待确认消息数 → Prometheus Gauge（堆积无告警是最痛的盲区）"""
    try:
        INGEST_QUEUE_DEPTH.labels(kind="stream").set(await client.xlen(settings.INGEST_STREAM))
        pending = await client.xpending(settings.INGEST_STREAM, settings.INGEST_CONSUMER_GROUP)
        INGEST_QUEUE_DEPTH.labels(kind="pending").set(pending.get("pending", 0) if pending else 0)
    except Exception as e:
        logger.warning(f"更新队列指标失败: {e}")


async def _claim_stale(client, consumer: str) -> list[tuple[str, dict]]:
    """认领「原持有人已失联」的 PEL 遗留消息，返回列表交主循环并发派发。

    判据 = 空闲超过 PEL_MIN_IDLE_MS **且** 原 consumer 的存活心跳已过期。
    ★ 心跳是承重的：单看空闲时长无法区分「持有人死了」和「一份大文档还在
      正常处理」（并发/多 worker 下误抢 = 同一文档重复处理，白烧一遍解析）。
      有了心跳就不再依赖「阈值必须大于单条最长处理时间」的人为约定。
    用 XPENDING(idle 过滤) + 逐条 XCLAIM，而不是 XAUTOCLAIM 一把梭：
    认领前能按 owner 跳过活人，XCLAIM 自带的 min_idle 二次校验防竞态。
    """
    claimed: list[tuple[str, dict]] = []
    try:
        pending = await client.xpending_range(
            settings.INGEST_STREAM,
            settings.INGEST_CONSUMER_GROUP,
            min="-",
            max="+",
            count=50,
            idle=PEL_MIN_IDLE_MS,
        )
        for entry in pending:
            owner = entry.get("consumer", "")
            msg_id = entry.get("message_id")
            if not msg_id:
                continue
            try:
                if await client.exists(_alive_key(owner)):
                    continue  # 原持有人还活着（可能正在处理大文档），不抢
                msgs = await client.xclaim(
                    settings.INGEST_STREAM,
                    settings.INGEST_CONSUMER_GROUP,
                    consumer,
                    min_idle_time=PEL_MIN_IDLE_MS,
                    message_ids=[msg_id],
                )
            except Exception as e:
                logger.warning(f"认领消息 {msg_id} 失败（下个周期重试）: {e}")
                continue
            for m_id, fields in msgs:
                logger.warning(f"认领遗留消息 msg_id={m_id}（原持有人 {owner} 心跳已失效）")
                claimed.append((m_id, fields))
    except Exception as e:
        logger.warning(f"PEL 回收失败（下个周期重试）: {e}")
    return claimed


async def _sweep_lost_jobs() -> int:
    """对账兜底：queued 记录超过 INGEST_LOST_JOB_HOURS 仍无进展 → 标记 failed。

    maxlen 裁剪会无声吞掉「已落库、从未投递」的任务，DB 记录永远停在
    queued——对账是唯一能把它变成显式失败的机制。只扫 queued 不扫
    processing：后者的活账由 PEL 回收兜底（有消息侧凭证），queued 没有
    凭证只能按时间判死；阈值（默认 2h）远大于正常排队时长，误判面极小。
    时间比较全部走服务端（func.now() - interval）：列是 naive DateTime，
    避免 Python 侧时区语义与 DB 不一致。
    """
    cutoff_h = max(1, settings.INGEST_LOST_JOB_HOURS)
    try:
        from src.knowledge.model import DocIngestJob

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(DocIngestJob).where(
                    DocIngestJob.stage == "queued",
                    DocIngestJob.updated_at < func.now() - timedelta(hours=cutoff_h),
                )
            )
            jobs = list(result.scalars().all())
            if not jobs:
                return 0
            for job in jobs:
                job.stage = "failed"
                job.error_msg = (
                    f"任务丢失：queued 超过 {cutoff_h}h 无进展"
                    f"（队列积压被裁剪或 worker 长期不可用），请重新上传"
                )
            await db.commit()
            for job in jobs:
                logger.warning(f"对账标记丢失任务 job_id={job.id}（queued 超时）")
            return len(jobs)
    except Exception as e:
        logger.warning(f"丢失任务对账失败（下个周期重试）: {e}")
        return 0


async def run_forever() -> None:
    from src.rag.ingestion.queue import ensure_group

    client = _client()
    try:
        await ensure_group(client)
    except Exception as e:
        logger.error(f"Redis 不可用，worker 无法启动: {e}")
        raise
    consumer = f"{socket.gethostname()}-{os.getpid()}"
    logger.info(
        f"入库 worker 启动: group={settings.INGEST_CONSUMER_GROUP} consumer={consumer} "
        f"concurrency={settings.INGEST_CONCURRENCY} "
        f"pel_reclaim={PEL_RECLAIM_INTERVAL_S}s/{PEL_MIN_IDLE_MS}ms(且心跳失效) "
        f"alive_ttl={ALIVE_TTL_S}s lost_sweep={settings.INGEST_LOST_JOB_HOURS}h"
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass  # 非 POSIX 平台退化为进程被直接杀死

    last_reclaim = 0.0

    # 并发消费：信号量限实际并发，任务数有背压上限；
    # 消息读入即进 PEL，任务完成时各自 XACK，崩溃后由 PEL 回收兜底
    concurrency = max(1, settings.INGEST_CONCURRENCY)
    sem = asyncio.Semaphore(concurrency)
    in_flight: set[asyncio.Task] = set()

    async def _guarded(msg_id: str, fields: dict) -> None:
        async with sem:
            try:
                await process_one(client, msg_id, fields)
            except Exception:
                logger.exception("处理入库消息时发生未捕获异常")

    while not stop.is_set():
        # 每轮刷新存活心跳（轮次 ≤ 3s，TTL 60s 余量充足）
        await _heartbeat(client, consumer)

        # 周期性队列指标 + PEL 回收 + 丢失任务对账
        now = time.monotonic()
        if now - last_reclaim >= PEL_RECLAIM_INTERVAL_S:
            last_reclaim = now
            await _update_queue_metrics(client)
            # 认领的遗留消息走与正常消费同一条 _guarded 派发路径：
            # 同受信号量约束，且不阻塞主循环（旧实现串行内联，一次认领
            # 多条时会把主循环饿住几分钟）
            for msg_id, fields in await _claim_stale(client, consumer):
                task = asyncio.create_task(_guarded(msg_id, fields))
                in_flight.add(task)
                task.add_done_callback(in_flight.discard)
            await _sweep_lost_jobs()

        # 背压：在途任务满时先等一个完成，避免消息被读进 PEL 却排队干等
        while len(in_flight) >= concurrency * 2 and not stop.is_set():
            done, _ = await asyncio.wait(
                in_flight, timeout=READ_BLOCK_MS / 1000,
                return_when=asyncio.FIRST_COMPLETED,
            )
            in_flight -= done
        if stop.is_set():
            break

        try:
            resp = await client.xreadgroup(
                settings.INGEST_CONSUMER_GROUP,
                consumer,
                {settings.INGEST_STREAM: ">"},
                count=concurrency,
                block=READ_BLOCK_MS,
            )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"XREADGROUP 异常，2s 后重试: {e}")
            await asyncio.sleep(2)
            continue

        if not resp:
            continue

        for _stream, messages in resp:
            for msg_id, fields in messages:
                if stop.is_set():
                    # 优雅停机：不 XACK 未处理的消息，留给下个 worker（走 PEL 回收）
                    logger.info(f"停机中，消息 {msg_id} 留给后续 worker 处理")
                    break
                task = asyncio.create_task(_guarded(msg_id, fields))
                in_flight.add(task)
                task.add_done_callback(in_flight.discard)

    # 优雅停机：等在途任务收尾（完成时各自 XACK）
    if in_flight:
        logger.info(f"等待 {len(in_flight)} 个在途入库任务收尾...")
        await asyncio.gather(*in_flight, return_exceptions=True)

    logger.info("入库 worker 已退出")


def main() -> None:
    setup_logger()
    asyncio.run(run_forever())


if __name__ == "__main__":
    main()
