# ============================================================
# 请求日志中间件 + trace_id 全链路透传
# ============================================================

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from src.core.logger import logger, new_trace_id, trace_id_var


class TraceLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("X-Trace-Id") or new_trace_id()
        token = trace_id_var.set(trace_id)

        start = time.perf_counter()
        method, path = request.method, request.url.path
        user_hint = request.headers.get("X-User-Id", "-")
        logger.info(f"--> {method} {path} user={user_hint}")

        try:
            response = await call_next(request)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"<-- {method} {path} {response.status_code} {elapsed:.1f}ms")
            response.headers["X-Trace-Id"] = trace_id
            return response
        except Exception:
            elapsed = (time.perf_counter() - start) * 1000
            logger.exception(f"<-- {method} {path} 异常 {elapsed:.1f}ms")
            raise
        finally:
            trace_id_var.reset(token)
