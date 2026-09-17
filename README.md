# rd-knowledge-svc

面向车企研发场景的 RAG 知识问答服务：文档检索（DocRAG）+ 知识图谱（GraphRAG）多通道融合，nl2sql 问数由独立服务 [rd-chatBI] 承担，本服务通过 HTTP 调用并透传行级权限。

## 架构

```
前端 ──▶ API (FastAPI :8002) ──▶ Redis Stream ──▶ 入库 Worker（独立进程）
              │
              ├─ doc_rag    Milvus（dense + BM25 混合检索 + Rerank）
              ├─ graph_rag  Neo4j（NL2Cypher，只读强制）
              └─ nl2sql     HTTP → rd-chatBI（熔断/重试/超时，失败不影响知识通道）
```

存储：PostgreSQL（元数据/反馈/评测）· Milvus（向量）· Neo4j（图谱）· Redis（缓存/限流/熔断状态/任务队列）· MinIO（原文与图片）。

## 快速开始（Docker Compose）

```bash
cp .env.example .env          # 填入 DASHSCOPE_API_KEY 等真实密钥
docker compose up -d --build  # API + worker + PG/Redis/MinIO/Milvus/Neo4j
docker compose exec api alembic upgrade head   # 建表
```

## 本地开发

```bash
pip install -r requirements-dev.txt
uvicorn src.main:app --port 8002 --reload
python -m src.rag.ingestion.worker             # 另开终端
pytest -q                                      # 单元测试
ruff check src scripts tests                   # lint
```

## 生产清单（上线前必查）

| 项 | 说明 |
|---|---|
| `APP_ENV=prod` | 启动时强制校验：禁 debug、CORS 必须收敛、JWT 密钥必须配置 |
| `AUTH_MODE=jwt` | 配 `JWT_SECRET`（HS256）或 `JWT_PUBLIC_KEY`（RS256）。若走网关 header 透传，网关必须剥离外部 `X-User-*` 头 |
| 轮换密钥 | 历史 HS256 密钥与 DashScope/DeepSeek key 曾入库/明文落盘，必须轮换 |
| `CORS_ORIGINS` | 显式 origin 列表，禁止 `*` |
| `MINIO_PUBLIC_READ=false` | 生产关闭桶公共读，改预签名 URL |
| `/health` vs `/ready` | liveness 用 `/health`；readiness 用 `/ready`（探测 PG/Redis/Milvus/Neo4j） |
| `/metrics` | 仅内网采集器可访问（网关/网络策略限制） |
| ChatBI 行级权限 | 本服务透传 role/owner_domain_id/business_line，rd-chatBI 侧必须同样验签 |

## 关键配置

全部配置见 `.env.example`（含注释）。核心开关：

- 韧性：`RETRIEVAL_CHANNEL_TIMEOUT/RETRIES`、`CIRCUIT_*`（熔断状态外置 Redis 多副本共享）、`QUERY_REWRITE_*`
- 检索：`RAG_DYNAMIC_TOPK`（断崖截断）、`RAG_HYDE_ENABLED`、`FORMULA_IMAGE_ENABLED`/`TABLE_ORIGINALS_ENABLED`（原图对照）
- 评测：`TRULENS_ENABLED`、`EVAL_SAMPLE_RATE`

## 可观测性

- 日志：本地 `logs/`（按天轮转 + gz），trace_id 全链路贯穿（含 ChatBI 透传与入库任务）
- 指标：`/metrics`（HTTP/LLM token 与延迟/通道延迟/熔断状态/队列深度/限流），告警规则见 `deploy/prometheus/alerts.yml`
- 审计：`audit.py` 查询审计日志（180 天保留）

## 工程约定

- Python 3.13 · FastAPI · SQLAlchemy async · Alembic 迁移
- pymilvus/minio 等**同步客户端在 async 代码里必须 `asyncio.to_thread`**，否则会阻塞事件循环
- LLM 调用必须有超时（`LLM_REQUEST_TIMEOUT`/`GENERATION_TIMEOUT`），检索通道走熔断→重试→超时三层保护
- 新增写操作端点：身份一律取 `get_current_user` 上下文，禁止信任请求体自报身份
