# ============================================================
# API 层共享依赖：LLM / Embedding 模型实例（模块级缓存）
# ============================================================

from functools import lru_cache

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI

from src.core.config import get_settings


@lru_cache
def _get_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.BASE_URL_CHAT,
        temperature=0,
        # ★ 客户端层超时 + 有限重试：LLM 端挂起时请求能失败而非永久挂死
        request_timeout=settings.LLM_REQUEST_TIMEOUT,
        max_retries=2,
    )


@lru_cache
def _get_llm_deepseek() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.DEEPSEEK_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com/v1",
        temperature=0,
        request_timeout=settings.LLM_REQUEST_TIMEOUT,
        max_retries=2,
    )


@lru_cache
def _get_embedding_model() -> DashScopeEmbeddings:
    settings = get_settings()
    return DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )


def get_llm() -> ChatOpenAI:
    return _get_llm()


def get_embedding_model() -> DashScopeEmbeddings:
    return _get_embedding_model()
