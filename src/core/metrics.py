# ============================================================
# Prometheus 指标注册 + HTTP 指标中间件
#
# 暴露端点：GET /metrics（main.py 注册）
# 业务指标在此集中定义，业务代码里只 import 并 inc/observe。
# 告警规则见 deploy/prometheus/alerts.yml
# ============================================================

from __future__ import annotations

import time

from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# ── HTTP 层 ───────────────────────────────────────────────────

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "HTTP 请求总数（按方法/路径/状态码）",
    ["method", "path", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP 请求耗时（秒）",
    ["method", "path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, float("inf")),
)

# ── LLM 用量 ─────────────────────────────────────────────────

LLM_TOKENS = Counter(
    "llm_tokens_total",
    "LLM Token 用量累计（kind=input/output）",
    ["model", "kind"],
)

LLM_CALLS = Counter(
    "llm_calls_total",
    "LLM 调用次数",
    ["model"],
)

LLM_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "LLM 单次调用耗时（秒），p95 由此计算",
    ["model"],
    buckets=(0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, float("inf")),
)

# ── 检索通道 ─────────────────────────────────────────────────

RETRIEVAL_REQUESTS = Counter(
    "retrieval_requests_total",
    "多通道检索请求（channel=doc_rag/graph_rag/chatbi；status=ok/empty/failed/skipped）",
    ["channel", "status"],
)

RETRIEVAL_LATENCY = Histogram(
    "retrieval_channel_duration_seconds",
    "检索通道单次执行耗时（秒），含重试",
    ["channel"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, float("inf")),
)

RETRIEVAL_OUTCOME = Counter(
    "knowledge_retrieval_outcome_total",
    "检索最终结果分类：ok=有结果 / empty=检索成功但无内容 / degraded=全部通道故障",
    ["outcome"],
)
# ★ empty 与 degraded 必须分开统计：前者是知识库覆盖问题（该补文档），
#   后者是服务依赖故障（该查 Milvus/embedding）。混在一起会让
#   "空答案率"告警分不清是内容缺口还是系统故障。

# ── LLM 成本 ─────────────────────────────────────────────────

LLM_COST_USD = Counter(
    "llm_cost_usd_total",
    "LLM 调用成本估算（美元，按 Settings.MODEL_PRICING_* 折算）",
    ["model"],
)

# ── 限流 ─────────────────────────────────────────────────────

RATE_LIMIT_REJECTED = Counter(
    "rate_limit_rejected_total",
    "限流拒绝次数（按端点）",
    ["endpoint"],
)

QUOTA_REJECTED = Counter(
    "token_quota_rejected_total",
    "Token 配额拒绝次数（scope=user）",
    ["scope"],
)

# ── 熔断器 ───────────────────────────────────────────────────

CIRCUIT_BREAKER_CHANGES = Counter(
    "circuit_breaker_state_changes_total",
    "熔断器状态切换次数（state=closed/open/half_open）",
    ["target", "state"],
)

CIRCUIT_BREAKER_STATE = Gauge(
    "circuit_breaker_state",
    "熔断器当前状态（0=closed 1=half_open 2=open），可对单值直接告警",
    ["target"],
)

# ── 入库任务 ─────────────────────────────────────────────────

INGESTION_JOBS = Counter(
    "ingestion_jobs_total",
    "入库任务（status=queued/started/succeeded/failed）",
    ["status"],
)

INGEST_QUEUE_DEPTH = Gauge(
    "ingest_queue_depth",
    "入库队列深度（Redis Stream XLEN）与待确认消息数（PEL）",
    ["kind"],
)

# 异步任务重试
ASYNC_TASK_RETRIES = Counter(
    "async_task_retries_total",
    "异步任务重试次数（按任务类型）",
    ["task"],
)

_CIRCUIT_STATE_CODE = {"closed": 0, "half_open": 1, "open": 2}


class PrometheusMiddleware(BaseHTTPMiddleware):
    """记录每个 HTTP 请求的计数与耗时（QPS / p95 延迟来源）。

    ★ path 标签用路由模板（/api/v1/ingest/jobs/{doc_id}）而非原始 URL，
      否则 doc_id/trace_id 会把标签基数撑爆，拖垮 Prometheus。"""

    async def dispatch(self, request: Request, call_next):
        method = request.method
        route = request.scope.get("route")
        path = getattr(route, "path", None) or request.url.path
        start = time.perf_counter()
        status = 500  # 异常时按 5xx 记录
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - start
            HTTP_REQUEST_DURATION.labels(method=method, path=path).observe(elapsed)
            HTTP_REQUESTS.labels(method=method, path=path, status=status).inc()
