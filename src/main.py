# ============================================================
# 应用入口
#
# 启动：uvicorn src.main:app --port 8002
# 文档：http://localhost:8002/docs（APP_ENV=prod 时自动关闭）
#
# 生命周期：lifespan 统一负责资源的初始化与释放 ——
#   启动：日志 / MinIO bucket 预检
#   关闭：评估器 → Neo4j → Milvus → DB 引擎 → Redis 连接池
# ============================================================

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.core.base_schema import ResponseSchema
from src.core.config import get_settings
from src.core.exceptions import register_exception_handlers
from src.core.logger import logger, setup_logger
from src.core.metrics import PrometheusMiddleware
from src.middlewares.logging import TraceLoggingMiddleware

settings = get_settings()

BASE_DIR = Path(__file__).resolve().parent  # src/
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger()
    if settings.APP_ENV == "prod" and settings.AUTH_MODE != "jwt":
        logger.warning("AUTH_MODE=header 运行在生产环境：身份依赖网关剥离 X-User-* 头，请确认网关已配置！")
    logger.info(f"{settings.APP_NAME} 启动 env={settings.APP_ENV} port=8002")

    # MinIO bucket 预检（旧实现推迟到首张图片上传时，失败发现晚）
    try:
        import asyncio

        from src.infra.minio_client import ensure_bucket_exists
        await asyncio.to_thread(ensure_bucket_exists)
    except Exception as e:
        logger.warning(f"MinIO bucket 预检失败（图片上传时会重试）: {e}")

    yield

    # ── 优雅停机：集中释放所有外部资源 ──
    logger.info(f"{settings.APP_NAME} 关闭中：释放资源...")
    from src.infra.db import engine
    from src.infra.milvus_client import close_milvus_client
    from src.infra.neo4j_client import close_neo4j_driver
    from src.infra.redis_cache import redis_pool

    try:
        from src.rag.evaluation.async_tracker import AsyncEvaluator
        evaluator = AsyncEvaluator._instance
        if evaluator is not None and hasattr(evaluator, "close"):
            await evaluator.close()
    except Exception as e:
        logger.warning(f"评估器关闭失败: {e}")

    try:
        await close_neo4j_driver()
    except Exception as e:
        logger.warning(f"Neo4j driver 关闭失败: {e}")

    try:
        close_milvus_client()
    except Exception as e:
        logger.warning(f"Milvus 连接关闭失败: {e}")

    try:
        await engine.dispose()
    except Exception as e:
        logger.warning(f"DB 引擎释放失败: {e}")

    try:
        await redis_pool.disconnect()
    except Exception as e:
        logger.warning(f"Redis 连接池释放失败: {e}")

    logger.info(f"{settings.APP_NAME} 已关闭")


app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.APP_DEBUG,
    lifespan=lifespan,
    # 生产环境关闭交互式文档与 OpenAPI schema（不对外暴露接口清单）
    docs_url=None if settings.APP_ENV == "prod" else "/docs",
    redoc_url=None if settings.APP_ENV == "prod" else "/redoc",
    openapi_url=None if settings.APP_ENV == "prod" else "/openapi.json",
)

app.add_middleware(TraceLoggingMiddleware)
app.add_middleware(PrometheusMiddleware)

# ★ CORS：origin 白名单从配置注入（CORS_ORIGINS，逗号分隔）。
#   通配 origin 与 allow_credentials 组合既不安全也不合规范，禁止同时出现。
_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials="*" not in _origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
register_exception_handlers(app)

# ── 注册路由 ────────────────────────────────────────────────────────────

from src.api.routers.eval import router as eval_router
from src.api.routers.feedback import router as feedback_router
from src.api.routers.ingest import router as ingest_router
from src.api.routers.knowledge import router as knowledge_router

app.include_router(knowledge_router)
app.include_router(ingest_router)
app.include_router(eval_router)
app.include_router(feedback_router)


@app.get("/health", response_model=ResponseSchema[dict])
async def health():
    """liveness：进程活着即返回 200，不探测依赖"""
    return ResponseSchema(data={"app": settings.APP_NAME, "env": settings.APP_ENV})


@app.get("/ready")
async def readiness():
    """readiness：探测核心依赖，任一不可用返回 HTTP 503，编排层据此摘流量"""
    import asyncio

    from fastapi.responses import JSONResponse

    checks: dict[str, bool] = {}

    async def _pg() -> bool:
        from sqlalchemy import text

        from src.infra.db import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True

    async def _redis() -> bool:
        from src.infra.redis_cache import get_shared_redis_client
        await asyncio.wait_for(get_shared_redis_client().ping(), timeout=2)
        return True

    async def _milvus() -> bool:
        from src.infra.milvus_client import get_milvus_client
        await asyncio.wait_for(asyncio.to_thread(get_milvus_client().list_collections), timeout=3)
        return True

    async def _neo4j() -> bool:
        from src.infra.neo4j_client import get_neo4j_driver
        await asyncio.wait_for(get_neo4j_driver().verify_connectivity(), timeout=3)
        return True

    probes = {"postgres": _pg, "redis": _redis, "milvus": _milvus, "neo4j": _neo4j}
    for name, probe in probes.items():
        try:
            checks[name] = bool(await probe())
        except Exception as e:
            logger.warning(f"/ready 探测 {name} 失败: {e}")
            checks[name] = False

    ok = all(checks.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"ready": ok, "checks": checks},
    )


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus 指标暴露端点（默认 /metrics 文本格式）。

    ★ 生产应通过网关/网络策略限制为内网采集器可访问。"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── 静态文件 & SPA（绝对路径，不依赖进程 CWD）──────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/eval")
async def eval_page():
    return FileResponse(STATIC_DIR / "eval.html")
