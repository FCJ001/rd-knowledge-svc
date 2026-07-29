# ============================================================
# 业务异常 + 全局异常处理
# ============================================================

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.core.logger import logger, trace_id_var

# 错误码
ERR_BAD_REQUEST = 40001
ERR_NOT_FOUND = 40004
ERR_PERMISSION_DENIED = 40203
ERR_KNOWLEDGE_FAILED = 40401
ERR_NL2SQL_FAILED = 40402
ERR_INGEST_FAILED = 40403
ERR_INTERNAL = 50000


class BizException(Exception):
    def __init__(self, message: str, code: int = ERR_BAD_REQUEST):
        self.code = code
        self.message = message
        super().__init__(message)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(BizException)
    async def biz_exception_handler(request: Request, exc: BizException):
        logger.warning(f"业务异常 code={exc.code} path={request.url.path} msg={exc.message}")
        return JSONResponse(
            status_code=200,
            content={
                "code": exc.code,
                "message": exc.message,
                "data": None,
                "trace_id": trace_id_var.get(),
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"未捕获异常 path={request.url.path}")
        return JSONResponse(
            status_code=500,
            content={
                "code": ERR_INTERNAL,
                "message": "服务内部错误，请联系管理员",
                "data": None,
                "trace_id": trace_id_var.get(),
            },
        )
