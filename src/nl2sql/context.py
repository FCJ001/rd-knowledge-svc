# ============================================================
# LangGraph Context — 运行时依赖注入（不参与 state 序列化）
# ============================================================

from typing import Any, Callable, TypedDict

from elasticsearch import AsyncElasticsearch
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.nl2sql.repositories import (
    PgMetaRepository,
    MilvusColumnRepository,
    MilvusMetricRepository,
    ESValueRepository,
)


class DataAgentContext(TypedDict, total=False):
    llm: BaseChatModel
    embedding_model: Embeddings
    milvus_client: MilvusClient
    milvus_column_repo: MilvusColumnRepository
    milvus_metric_repo: MilvusMetricRepository
    es_client: AsyncElasticsearch
    es_value_repo: ESValueRepository
    pg_meta_repo: PgMetaRepository
    dw_db_session: AsyncSession
    writer: Callable[[dict[str, Any]], None]  # 进度/结果推送回调
