# ============================================================
# 全局配置
#
# 所有外部依赖的连接信息、模型密钥统一从这里读，来源是 .env。
# ★ 绝不在业务代码里硬编码密钥
#
# 用法：
#   from src.core.config import get_settings
#   settings = get_settings()        # lru_cache，全进程只解析一次 .env
# ============================================================

from functools import lru_cache
from urllib.parse import quote_plus

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ---------------- 应用 ----------------
    APP_NAME: str = "rd-knowledge-svc"
    APP_ENV: str = "dev"                  # dev / test / prod
    APP_DEBUG: bool = False               # ★ 生产安全默认：debug 关（SQL echo、FastAPI debug 随之关）

    # ---------------- 认证 ----------------
    # header：开发期/网关可信模式，直接信任 X-User-* 请求头（网关必须剥离外部同名头）
    # jwt：验签 Authorization: Bearer <token>，密钥从环境注入
    AUTH_MODE: str = "header"
    JWT_SECRET: str = ""                  # HS256 验签密钥（AUTH_MODE=jwt 时必填，禁止硬编码）
    JWT_PUBLIC_KEY: str = ""              # RS256 公钥（PEM，上游网关签发时用）
    JWT_ALGORITHM: str = "HS256"          # HS256 / RS256
    GATEWAY_TRUSTED: bool = False         # header 模式下声明"网关已剥离外部 X-User-* 同名头"（prod 必须显式 true）
    JWT_REVOKED_SECRETS: str = ""         # 已泄露历史密钥黑名单（逗号分隔；部署侧注入，源码不落密钥字面量）

    # 删除文档（含 MinIO 原文）是不可逆操作，要求指定角色才放行。
    # 逗号分隔的 role 白名单；空字符串表示不校验（仅限开发环境）。
    DOC_DELETE_REQUIRED_ROLES: str = "admin"

    # ---------------- 跨域 ----------------
    CORS_ORIGINS: str = "*"               # 逗号分隔的显式 origin 列表；生产必须收敛

    # ---------------- PostgreSQL（共享实例，独立库）----------------
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "rdagent"
    DB_PASSWORD: str = "rdagent123"
    DB_NAME: str = "rd_knowledge"

    # ---------------- Redis Stack ----------------
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0
    REDIS_SOCKET_TIMEOUT: float = 5.0     # 单命令读写超时（worker 的 XREADGROUP block 需小于此值）
    REDIS_MAX_CONNECTIONS: int = 100      # 连接池上限，防无界增长

    # ---------------- MinIO ----------------
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "knowledge-docs"
    MINIO_SECURE: bool = False
    MINIO_PUBLIC_READ: bool = True        # 桶公共读策略（前端 <img> 直连）；生产建议改预签名 URL 后关闭

    # ---------------- PostgreSQL 连接池 ----------------
    DB_POOL_ENABLED: bool = True          # False 回退 NullPool（每请求新建连接）
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10

    # ---------------- Milvus ----------------
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_TIMEOUT: float = 10.0          # Milvus 客户端请求超时（秒），兜底所有 gRPC 调用

    # ---------------- Neo4j ----------------
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "rdagent123"
    NEO4J_CONNECTION_TIMEOUT: float = 5.0   # 建连超时（秒）
    NEO4J_MAX_POOL_SIZE: int = 50           # driver 连接池上限
    NEO4J_QUERY_TIMEOUT: float = 10.0       # 单条 Cypher 执行超时（秒）

    # ---------------- 模型 ----------------
    # ★ 主链路（生成/改写/HyDE/GraphRAG/幻觉检测/VL 摘要/LLM 重排/评测裁判）统一走
    #   DeepSeek 的 OpenAI 兼容端点；DashScope 保留给向量化（embedding/语义分块）
    #   与 qwen3-rerank 精排（RERANK_PROVIDER=dashscope，默认）。
    DASHSCOPE_API_KEY: str = ""           # 向量化（EMBEDDING_MODEL）+ qwen3-rerank 精排使用
    BASE_URL_CHAT: str = "https://api.deepseek.com/v1"
    CHAT_MODEL: str = "deepseek-chat"     # 可换 deepseek-reasoner（推理模型，延迟更高）
    EMBEDDING_MODEL: str = "text-embedding-v3"  # DashScope 向量化，不随主链路切换

    DEEPSEEK_API_KEY: str = ""            # 主链路密钥（CHAT_API_KEY 为空时回退到它）
    CHAT_API_KEY: str = ""                # 显式指定主链路密钥（换 OpenAI 兼容供应商时用）

    # ---------------- MinerU ----------------
    # ★ 解析服务地址必须由部署环境显式提供，代码不设业务默认值（prod 校验强制）
    MINERU_API_URL: str = "http://localhost:8000"
    MINERU_BACKEND: str = "hybrid-auto-engine"
    MINERU_TIMEOUT: int = 300
    # ---------------- RAG ----------------
    # ★ 下面几项是检索调参旋钮。RAG_TOP_K / RAG_RERANK_TOP_K 现在同时被
    #   在线链路（doc_rag.search_docs_raw 的 None 默认值回落到这里）和离线
    #   消融脚本消费——改 .env 对两边同时生效，离线/在线口径天然一致。
    # 初召条数。★ 2026-09-22 实测（固定取 5 条证据）：
    #   深度的最终页级召回是非单调的，**峰值就在 20** ——
    #   10→89.8% / 15→90.7% / 20→93.5% / 30→92.2% / 40→91.1%。
    #   加深确实抬高召回天花板（raw@30 从 94.9% 到 97.0%），但 LLM 清单式重排
    #   在更长的候选列表上精度下降，最终证据反而变差。**别为了 raw 数字提高它。**
    #   详见 docs/runbook.md 4.4.4。
    RAG_TOP_K: int = 20
    RAG_RERANK_TOP_K: int = 5      # 精排目标条数（动态截断关闭时的固定值）
    RAG_HYDE_ENABLED: bool = False # HyDE：多一次 LLM 调用换语义鸿沟跨越
    RAG_HYBRID_ENABLED: bool = False  # 保留开关；当前 hybrid(RRF) 恒开，置 false 不回退单路
    # Rerank 后按分数断崖动态截断（最多 10 条）。
    # ★ 默认关闭：2026-09-22 实测（n=36，同一批题）——
    #   开：证据中位 2（33% 的查询只剩 1 条），页级召回 89.1%，有据性 0.949
    #   关：证据固定 5 条，              页级召回 93.5%，有据性 0.965
    #   "证据条数与答案质量"是非单调关系，峰值在 5 条附近（2 条 0.949 / 5 条 0.965 /
    #   8 条 0.925 / 10 条 0.910）。而 RAG_RERANK_TOP_K 恰好就是 5，本来就是最优值，
    #   动态截断反而把它砍坏了。详见 docs/runbook.md 4.4.3。
    RAG_DYNAMIC_TOPK: bool = False
    RERANK_TIMEOUT: float = 10.0   # 精排调用超时（秒），超时回退向量距离排序
    # ★ 2026-09-24 默认切至 dashscope（qwen3-rerank 专用模型）：pointwise 分数确定性、
    #   可复现，拒答阈值才可标定；LLM 清单式重排保留为可选后端（分数有漂移，且整表
    #   进 prompt 更慢更贵）。此前调参结论（RAG_TOP_K 峰值 20 / τ=0.95）均基于
    #   deepseek 重排实测，切换后需用 scripts/eval_report.py 重新对照验证。
    RERANK_PROVIDER: str = "dashscope"  # dashscope=qwen3-rerank 专用模型（默认）/ deepseek=LLM 清单式重排 / off=RRF 序直通

    # 断崖截断参数（决定"喂给生成的证据条数"）：
    #   相邻分差 ≥ RERANK_GAP_ABS 或相对降幅 ≥ RERANK_GAP_RATIO 即截断于断崖前。
    # ★ 这些是影响答案质量的关键旋钮，但两套检索指标都测不出它的影响
    #   （doc 级到顶、页级被返回条数卡住），所以必须配 eval_report.py 的
    #   两阶段对照来调——见 docs/runbook.md 4.4.2。
    RERANK_MIN_TOPK: int = 1       # 断崖再早也至少保留几条证据
    # LLM 清单式重排一次最多看多少个候选。★ 必须 ≥ RAG_TOP_K，
    #   否则超出部分的候选 LLM 从未见过，会被打成 0 分排在末尾——
    #   表现为"提高了 RAG_TOP_K 但召回没变"。
    RERANK_GAP_ABS: float = 0.5
    RERANK_GAP_RATIO: float = 0.25
    RERANK_MAX_CANDIDATES: int = 20

    # 返回给前端的 contexts 单条字符上限。同时是前端引用展示、缓存载荷与
    # 在线评测 context_relevance 的输入，截太狠会让三处都薄于实际生成上下文。
    RAG_CONTEXT_MAX_CHARS: int = 1500

    # 低置信拒答阈值：文档通道 top-1 分数低于此值时直接答"没有"，
    # 不调生成 LLM（省 token + 防硬编）。0 = 不启用。
    # ★ 必须在本项目 reranker 的分数尺度上标定，不能照搬他人数值。
    #   2026-09 实测（deepseek 精排，已非默认 provider）：可回答题 top-1 中位 1.000、
    #   拒答题中位 0.600，最优阈值 τ=0.95（平衡准确率 0.95）。
    #   ★ 默认 provider 已切至 dashscope，该 τ 值在 qwen3-rerank 分数尺度上
    #     **未经标定、不可沿用**；启用拒答前必须用 scripts/eval_report.py 重标。
    RAG_REFUSE_THRESHOLD: float = 0.0

    # 生产安全：为 true 时请求必须携带 model_code，否则 400。
    # 车型隔离是安全事故而非体验问题——过滤条件为空时静默放开比拒绝更危险。
    RAG_REQUIRE_MODEL_CODE: bool = False

    # ---------------- 文档级 ACL（权限在检索之前生效）----------------
    # ACL 谓词（array_contains_any(acl_roles, [role])）并进 Milvus 检索表达式，
    # 无权内容不进候选集——不做"检索 top_k 再按权限丢弃"（理由见 knowledge/acl.py）。
    # ★ 关闭仅用于演练/排障：prod 下拒绝启动。
    DOC_ACL_ENABLED: bool = True
    # 入库未显式指定 acl_roles 时的默认可见角色（逗号分隔，取值见 acl.ASSIGNABLE_ROLES）。
    # ★ 默认空 = 仅 admin 可见（fail-closed）：宁可让上传者显式声明，
    #   也不要把"忘了填"变成"所有人都能看"。
    DOC_ACL_DEFAULT_ROLES: str = ""
    # 图谱通道节点级 ACL 未实现（Neo4j Community 无细粒度权限，Cypher 由 LLM 生成
    # 无法结构化校验）。非 admin 用户默认被拒（fail-closed）；置 true 即对非 admin
    # 开放该通道且不施加节点级过滤，属显式承担风险，仅限开发环境。
    GRAPH_ACL_ALLOW_NON_ADMIN: bool = False

    # ---------------- LLM 超时 ----------------
    LLM_REQUEST_TIMEOUT: float = 120.0    # 单次 LLM HTTP 请求超时（openai 客户端层）
    GENERATION_TIMEOUT: float = 120.0     # 最终答案生成整体超时（秒），防 LLM 端挂起拖死请求
    HALLUCINATION_TIMEOUT: float = 30.0   # 幻觉检测超时（秒），fail-open

    # ---------------- 图片 VL 摘要 ----------------
    # DeepSeek 视觉模型 deepseek-flash：OpenAI 兼容 image_url 协议，
    # 图片只允许出现在 user 消息里，单图 ≤32MiB（入库页图裁剪远小于此）
    VL_MODEL: str = "deepseek-flash"
    VL_BASE_URL: str = "https://api.deepseek.com/v1"
    VL_API_KEY: str = ""                  # 空则回退主链路密钥（chat_api_key）
    VL_TIMEOUT: float = 60.0              # 单张图片 VL 调用超时（秒），超时按无摘要处理
    VL_CONCURRENCY: int = 2               # VL 并发调用上限（批量入库时限流，防上游 429）
    EMBED_CONCURRENCY: int = 2            # embedding 并发调用上限（批量入库时限流）
    IMAGE_SUMMARIZE_ENABLED: bool = True  # 入库时对每张图片调用 VL 生成描述写入 markdown

    # ---------------- 公式原图对照（双通道）----------------
    # MinerU formula_enable 是二选一：True=LaTeX 文本（可检索），False=公式原图（保真）。
    # 开启后入库时额外跑一遍 formula_enable=false，取公式原图嵌入 markdown 供人眼对照，
    # 防 LaTeX OCR 识别不准确。代价：解析时间约 2 倍（异步 worker 可接受）。
    FORMULA_IMAGE_ENABLED: bool = True

    # ---------------- 表格原图对照（bbox 裁剪）----------------
    # MinerU content_list 返回表格 bbox + page_idx，用 PyMuPDF 渲染对应页裁剪出表格原图，
    # 嵌入 markdown 表格下方供人眼对照，防复杂表格（colspan/rowspan）OCR 串行/丢列。
    # 跨页表格每个页片段一块，续页块（table_body=None）归并到上一块。
    TABLE_ORIGINALS_ENABLED: bool = True
    # 入库时对每张表格原图调用 VL：生成一句话语义摘要（写入 chunk 可检索文本，
    # 让"拧多紧"这类语义化提问能召回表格）+ 净化 Markdown 转录（与 MinerU HTML
    # 并排互查）；失败 fail-open 保留原样，单表失败不影响其他表
    TABLE_VL_ENABLED: bool = True

    # ---------------- API 限流 ----------------
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_BACKEND: str = "redis"     # redis（ZSET 跨 worker）/ memory（进程内 deque）
    RATE_LIMIT_MAX_REQUESTS: int = 20     # 每个 user_id 每个滑动窗口内最多请求数
    RATE_LIMIT_WINDOW_SECONDS: int = 60   # 滑动窗口时长（秒）

    # 按用户/天的 token 配额（成本维度止损）。0 = 不限额。
    # 与 RATE_LIMIT 互补：请求数限流挡不住「低频但每次烧几万 token」的调用，
    # 而一次知识检索本身就要 4~5 次 LLM 调用。
    USER_DAILY_TOKEN_QUOTA: int = 0

    # ---------------- 数据生命周期 ----------------
    # 软删文档（status=deleted）超过该天数后被 maintenance.py retention 清理。
    # 0 = 永久保留（默认：合规场景常常要求可追溯，不要默认删数据）。
    DOC_RETENTION_DAYS: int = 0

    # ---------------- 查询缓存 ----------------
    QUERY_CACHE_ENABLED: bool = True       # 知识检索结果缓存（文档/图谱相对静态，安全）
    QUERY_CACHE_TTL: int = 300             # 缓存有效期（秒）

    # ---------------- 上传 ----------------
    UPLOAD_MAX_MB: int = 200               # 上传文件大小上限（MB），防磁盘耗尽
    ALLOWED_UPLOAD_SUFFIXES: str = ".pdf,.doc,.docx,.md,.txt"  # 扩展名白名单（逗号分隔）

    # ---------------- 入库异步任务（Redis Stream + worker）----------------
    INGEST_STREAM: str = "alm_ingest:jobs"        # 入库任务 Stream
    INGEST_CONSUMER_GROUP: str = "alm_ingest_workers"  # 消费者组
    INGEST_STREAM_MAX_LEN: int = 1000             # Stream 最大保留消息数
    INGEST_MAX_RETRIES: int = 2                   # worker 处理失败重试次数
    INGEST_CONCURRENCY: int = 1                   # worker 单进程并发处理消息数（横向扩容=多起 worker 进程）
    INGEST_PEL_MIN_IDLE_S: int = 1800             # PEL 回收认领的最小空闲时长（秒）；另需原持有人心跳已失效才认领
    INGEST_LOST_JOB_HOURS: int = 2                # queued 记录超该时长仍无进展 → 对账标记 failed（防 maxlen 裁剪静默丢任务）

    # ---------------- 韧性（超时/重试/熔断）----------------
    RETRIEVAL_CHANNEL_TIMEOUT: int = 20   # 单检索通道超时（秒），超时按失败降级
    RETRIEVAL_CHANNEL_RETRIES: int = 2    # 通道临时失败重试次数（指数退避）
    CIRCUIT_FAILURE_THRESHOLD: int = 5    # 熔断阈值：连续失败 N 次打开熔断器
    CIRCUIT_RESET_TIMEOUT: int = 30       # 熔断复位窗口（秒），过后放一个探针
    CIRCUIT_REDIS_ENABLED: bool = True    # 熔断状态外置 Redis（多副本共享；Redis 异常自动降级进程内）

    # ---------------- Query 改写（检索主链路第一步）----------------
    QUERY_REWRITE_ENABLED: bool = True    # 口语→术语 + 子查询拆分，失败降级原问题
    QUERY_REWRITE_TIMEOUT: float = 8.0    # 改写 LLM 调用超时（秒）
    QUERY_REWRITE_MAX_SUB_QUERIES: int = 3  # 子查询数量上限（防 LLM 拆分失控放大召回成本）

    # ---------------- ChatBI（nl2sql 通道走独立服务）----------------
    CHATBI_URL: str = "http://localhost:8004"
    CHATBI_PROJECT_ID: str = "rd_agent"  # 多数据源路由（bi_datasources.code）

    # ---------------- TruLens ----------------
    TRULENS_ENABLED: bool = True

    # ---------------- 在线评测采样 ----------------
    EVAL_SAMPLE_RATE: float = 0.1  # 在线 LLM-as-Judge 采样率，0.1 = 10%
    JUDGE_MODEL: str = ""          # 裁判模型；空则跟随 CHAT_MODEL（换独立模型消除自评偏差）

    # ---------------- Guardrails ----------------
    GUARDRAILS_ENABLED: bool = True
    GUARDRAILS_BLOCK_DDL: bool = True  # 拦截 DROP/TRUNCATE/ALTER
    GUARDRAILS_BLOCK_DML_WITHOUT_WHERE: bool = True  # 拦截无 WHERE 的 DELETE/UPDATE

    # ---------------- 模型定价（USD/1M tokens，成本核算用）----------------
    # deepseek-flash 峰时参考价（输入未命中缓存 $0.30 / 输出 $1.20）；
    # 谷时减半、缓存命中更低，精确计费以价目页为准：
    # https://api-docs.deepseek.com/quick_start/pricing
    MODEL_PRICING_INPUT: float = 0.30
    MODEL_PRICING_OUTPUT: float = 1.20

    # ---------------- 日志 ----------------
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    AUDIT_LOG_RETENTION: str = "180 days"

    # ---------------- 项目一（跨服务调知识库）----------------
    KNOWLEDGE_SVC_URL: str = "http://localhost:8001"

    @model_validator(mode="after")
    def _retrieval_consistency(self) -> "Settings":
        """检索参数一致性：重排候选上限必须 ≥ 初召条数。

        ★ 否则超出的候选 LLM 从未见过、被打 0 分，且动态截断会被静默跳过——
        表现为"改了 RAG_TOP_K，检索行为变得莫名其妙"。
        实测：top_k=30 + 候选上限 20 时，固定 5 条证据的页级召回 92.2%，
        而两者配齐（30/30）时反而掉到 85.7%，两者都不是调参者预期的结果。
        """
        if self.RERANK_MAX_CANDIDATES < self.RAG_TOP_K:
            raise ValueError(
                f"RERANK_MAX_CANDIDATES({self.RERANK_MAX_CANDIDATES}) 必须 ≥ "
                f"RAG_TOP_K({self.RAG_TOP_K})：否则超出上限的候选 LLM 看不到，"
                "会被打 0 分并让动态截断静默失效"
            )
        return self

    @model_validator(mode="after")
    def _prod_safety_check(self) -> "Settings":
        """生产环境防呆：危险默认值在 prod 下直接拒绝启动。"""
        if self.APP_ENV == "prod":
            problems = []
            if self.APP_DEBUG:
                problems.append("APP_DEBUG 必须为 false")
            if self.AUTH_MODE == "jwt" and not (self.JWT_SECRET or self.JWT_PUBLIC_KEY):
                problems.append("AUTH_MODE=jwt 时必须配置 JWT_SECRET 或 JWT_PUBLIC_KEY")
            revoked = {s.strip() for s in self.JWT_REVOKED_SECRETS.split(",") if s.strip()}
            if self.JWT_SECRET and self.JWT_SECRET in revoked:
                problems.append("JWT_SECRET 命中已泄露密钥黑名单（JWT_REVOKED_SECRETS），必须更换")
            if self.AUTH_MODE == "header" and not self.GATEWAY_TRUSTED:
                problems.append(
                    "AUTH_MODE=header 依赖网关剥离外部 X-User-* 同名头；"
                    "确认网关已剥离后显式设置 GATEWAY_TRUSTED=true，否则改用 AUTH_MODE=jwt"
                )
            if (
                self.RERANK_PROVIDER.strip().lower() == "dashscope"
                and not self.DASHSCOPE_API_KEY
            ):
                # 精排在检索关键路径上：无密钥时每个查询都会先打一次注定失败的
                # API 再回退 RRF 序（多付一次超时等待 + 日志噪音），不如启动即拒。
                # 仅 prod 强制；dev/CI 允许无密钥启动（运行期降级链兜底）。
                problems.append(
                    "RERANK_PROVIDER=dashscope 时必须配置 DASHSCOPE_API_KEY"
                )
            if self.CORS_ORIGINS.strip() == "*":
                problems.append("CORS_ORIGINS 必须收敛为显式 origin 列表")
            if self.DB_PASSWORD == "rdagent123" or self.NEO4J_PASSWORD == "rdagent123":
                problems.append("DB_PASSWORD/NEO4J_PASSWORD 不能使用开发默认密码")
            if self.MINIO_ACCESS_KEY == "minioadmin" or self.MINIO_SECRET_KEY == "minioadmin":
                problems.append("MINIO_ACCESS_KEY/MINIO_SECRET_KEY 不能使用开发默认值")
            if self.MINIO_PUBLIC_READ:
                problems.append(
                    "MINIO_PUBLIC_READ 必须为 false：公共读会让原文/图片 URL 可被任意人访问，"
                    "等于绕过文档级权限（改预签名 URL 或走鉴权代理）"
                )
            if "MINERU_API_URL" not in self.model_fields_set:
                problems.append("MINERU_API_URL 必须由部署环境显式提供（不入代码默认值）")
            if not self.DOC_ACL_ENABLED:
                problems.append(
                    "DOC_ACL_ENABLED 必须为 true：关掉等于所有角色都能检索到全部文档"
                    "（权限过滤不进检索请求，见 knowledge/acl.py）"
                )
            if self.GRAPH_ACL_ALLOW_NON_ADMIN:
                problems.append(
                    "GRAPH_ACL_ALLOW_NON_ADMIN 必须为 false：图谱节点级 ACL 未实现，"
                    "对非 admin 开放等于无权限隔离"
                )
            if problems:
                raise ValueError(f"生产配置校验失败: {'; '.join(problems)}")
        return self

    @property
    def DATABASE_URL(self) -> str:
        """本服务自有库 rd_knowledge（密码 URL 编码，特殊字符不破坏连接串）"""
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{quote_plus(self.DB_PASSWORD)}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def REDIS_URL(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def chat_api_key(self) -> str:
        """主链路（对话/生成/VL/重排/裁判）密钥：CHAT_API_KEY 优先，回退 DEEPSEEK_API_KEY"""
        return self.CHAT_API_KEY or self.DEEPSEEK_API_KEY

    @property
    def vl_api_key(self) -> str:
        """VL 密钥：VL_API_KEY 优先，回退主链路密钥"""
        return self.VL_API_KEY or self.chat_api_key

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
