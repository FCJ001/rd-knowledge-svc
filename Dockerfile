# ============================================================
# rd-knowledge-svc 生产镜像（多阶段）
# 构建：docker build -t rd-knowledge-svc:latest .
# ============================================================

FROM python:3.13-slim AS builder

WORKDIR /build
COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
    && python -c "import fitz" || pip install --no-cache-dir --prefix=/install pymupdf

FROM python:3.13-slim

# 健康检查与运行所需的最小工具
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
WORKDIR /app
COPY src ./src
COPY alembic ./alembic
COPY alembic.ini pytest.ini requirements.txt ./

# 非root运行
RUN useradd -m svc && chown -R svc:svc /app
USER svc

ENV PYTHONUNBUFFERED=1 LOG_DIR=/tmp/logs

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -sf http://localhost:8002/ready || exit 1

# APP_ENV=prod 会在启动时强制校验危险默认值（debug/CORS/JWT 密钥）
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "2"]
