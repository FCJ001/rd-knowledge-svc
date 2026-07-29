# ============================================================
# 日志（loguru）
#
# 三个 sink：控制台 + 按日文件 + 审计独立文件（180 天留存）
# trace_id 全链路透传
#
# 用法：
#   from src.core.logger import logger, setup_logger, trace_id_var
#   setup_logger()
# ============================================================

import sys
import uuid
from contextvars import ContextVar
from pathlib import Path

from loguru import logger

from src.core.config import get_settings

# ------------------------------------------------------------
# trace_id 上下文变量
# ------------------------------------------------------------
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def _patcher(record) -> None:
    record["extra"].setdefault("trace_id", trace_id_var.get())


_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<magenta>{extra[trace_id]}</magenta> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def setup_logger() -> None:
    settings = get_settings()

    logger.remove()
    logger.configure(patcher=_patcher)

    # 控制台
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format=_LOG_FORMAT,
        colorize=True,
    )

    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 应用日志：按天切，留 30 天，旧文件压缩
    logger.add(
        log_dir / "{time:YYYY-MM-DD}.log",
        level=settings.LOG_LEVEL,
        format=_LOG_FORMAT,
        rotation="00:00",
        retention="30 days",
        compression="gz",
        encoding="utf-8",
    )

    # ★ 审计日志独立文件：只记录 audit=True 的日志
    logger.add(
        log_dir / "audit_{time:YYYY-MM-DD}.log",
        level="INFO",
        format=_LOG_FORMAT,
        rotation="00:00",
        retention=settings.AUDIT_LOG_RETENTION,
        compression="gz",
        encoding="utf-8",
        filter=lambda record: record["extra"].get("audit") is True,
    )


__all__ = ["logger", "setup_logger", "trace_id_var", "new_trace_id"]
