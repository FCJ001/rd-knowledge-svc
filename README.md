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

**完整的容量模型、故障处置、变更流程与已知缺口见 [docs/runbook.md](docs/runbook.md)。**

| 项 | 说明 |
|---|---|
| `APP_ENV=prod` | 启动时强制校验：禁 debug、CORS 必须收敛、JWT 密钥必须配置、禁开发默认密码（DB/Neo4j/MinIO）、**`MINIO_PUBLIC_READ` 必须 false**、`MINERU_API_URL` 必须显式提供 |
| `AUTH_MODE=jwt` | 配 `JWT_SECRET`（HS256）或 `JWT_PUBLIC_KEY`（RS256）。若走网关 header 透传，网关必须剥离外部 `X-User-*` 头并显式设 `GATEWAY_TRUSTED=true` |
| `DOC_DELETE_REQUIRED_ROLES` | 删除文档要求的角色白名单（默认 `admin`）。留空 = 不校验，仅限开发环境 |
| `RAG_REQUIRE_MODEL_CODE` | 车型隔离场景设为 true：请求不带 `model_code` 直接 400，避免跨车型串味 |
| `USER_DAILY_TOKEN_QUOTA` | 按用户/天的 token 配额（0 = 不限额）。与请求数限流互补，防「低频高耗」烧预算 |
| `DOC_RETENTION_DAYS` | 软删文档保留天数（0 = 永久保留）。大于 0 时由 maintenance 脚本清理 |
| 轮换密钥 | 历史 HS256 密钥与 DashScope/DeepSeek key 曾入库/明文落盘，必须轮换；已泄露密钥进 `JWT_REVOKED_SECRETS` 黑名单（部署侧注入） |
| 数据备份 | `./scripts/backup.sh`：PG 在线逻辑备份 + MinIO 对象镜像；Neo4j/Milvus 数据卷需 `COLD_BACKUP=1` 停机冷备，配 crontab 定时执行 |
| 定时维护 | `python scripts/maintenance.py report\|compact\|retention`（crontab 建议见 runbook） |
| 低置信拒答 | `RAG_REFUSE_THRESHOLD`：文档通道 top-1 分数低于阈值时直接答"没有"、不调生成 LLM。默认 0（关）。**阈值必须在本项目 reranker 尺度上标定**（实测 τ=0.95），换 reranker 要重标 |
| 重入库 | `python scripts/reingest.py --list` 看待办：改文档名、页码链路变更、切片策略变更都需重入库 |
| `CORS_ORIGINS` | 显式 origin 列表，禁止 `*` |
| `/health` vs `/ready` | liveness 用 `/health`；readiness 用 `/ready`（探测 PG/Redis/Milvus/Neo4j） |
| `/metrics` | 仅内网采集器可访问（网关/网络策略限制） |
| 证据条数 | `RAG_DYNAMIC_TOPK=false`（默认）+ `RAG_RERANK_TOP_K=5` = 固定喂 5 条证据。实测证据条数与答案质量**非单调**、峰值在 5 条，改前必读 runbook 4.4.3；`RERANK_MIN_TOPK`/`GAP_*` 仅在开启动态截断时生效 |
| 检索质量门禁 | `python scripts/eval_gate.py`。**当前标注集仅 10 题、低于门禁下限 30 题，门禁尚未生效**——需先补标注 |
| ChatBI 行级权限 | 本服务透传 role/owner_domain_id/business_line，rd-chatBI 侧必须同样验签 |
| 文档级 ACL | 权限谓词并进 Milvus 查询（**检索前过滤**，无权内容不进候选集），admin 绕过、词表外角色 fail-closed。上传时 `acl_roles` 声明可见角色；存量授权用 `scripts/backfill_acl.py`。图谱通道无节点级 ACL，非 admin 默认拒绝（runbook 五） |

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
