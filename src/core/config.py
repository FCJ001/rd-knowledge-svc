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

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ---------------- 应用 ----------------
    APP_NAME: str = "rd-knowledge-svc"
    APP_ENV: str = "dev"
    APP_DEBUG: bool = True

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

    # ---------------- MinIO ----------------
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "knowledge-docs"
    MINIO_SECURE: bool = False

    # ---------------- Milvus ----------------
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # ---------------- Neo4j ----------------
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "rdagent123"

    # ---------------- 模型 ----------------
    DASHSCOPE_API_KEY: str = ""
    BASE_URL_CHAT: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    CHAT_MODEL: str = "qwen-max"
    EMBEDDING_MODEL: str = "text-embedding-v3"

    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # ---------------- MinerU ----------------
    MINERU_API_URL: str = "http://117.50.195.135:8000"
    MINERU_BACKEND: str = "hybrid-auto-engine"
    MINERU_TIMEOUT: int = 300

    # ---------------- RAG ----------------
    RAG_TOP_K: int = 20
    RAG_RERANK_TOP_K: int = 5
    RAG_HYDE_ENABLED: bool = False
    RAG_HYBRID_ENABLED: bool = False

    # ---------------- NL2SQL（查项目一业务库 rd_agent）----------------
    ALM_DB_USER: str = "rdagent"
    ALM_DB_PASSWORD: str = "rdagent123"
    ALM_DB_NAME: str = "rd_agent"

    # ---------------- TruLens ----------------
    TRULENS_ENABLED: bool = True

    # ---------------- 在线评测采样 ----------------
    EVAL_SAMPLE_RATE: float = 0.1  # 在线 LLM-as-Judge 采样率，0.1 = 10%

    # ---------------- Guardrails ----------------
    GUARDRAILS_ENABLED: bool = True
    GUARDRAILS_BLOCK_DDL: bool = True  # 拦截 DROP/TRUNCATE/ALTER
    GUARDRAILS_BLOCK_DML_WITHOUT_WHERE: bool = True  # 拦截无 WHERE 的 DELETE/UPDATE

    # ---------------- 模型定价（USD/1M tokens）----------------
    MODEL_PRICING_INPUT: float = 0.4   # qwen-max 输入 $0.4/1M
    MODEL_PRICING_OUTPUT: float = 1.2  # qwen-max 输出 $1.2/1M

    # ---------------- 日志 ----------------
    LOG_LEVEL: str = "DEBUG"
    LOG_DIR: str = "logs"
    AUDIT_LOG_RETENTION: str = "180 days"

    # ---------------- 项目一（跨服务调知识库）----------------
    KNOWLEDGE_SVC_URL: str = "http://localhost:8001"

    @property
    def DATABASE_URL(self) -> str:
        """本服务自有库 rd_knowledge"""
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def ALM_DATABASE_URL(self) -> str:
        """项目一业务库 rd_agent（NL2SQL 查询目标，只读）"""
        return (
            f"postgresql+asyncpg://{self.ALM_DB_USER}:{self.ALM_DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.ALM_DB_NAME}"
        )

    @property
    def REDIS_URL(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
