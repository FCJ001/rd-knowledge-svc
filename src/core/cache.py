# ============================================================
# Redis 查询缓存
#
# 缓存对象：知识检索的最终答案（文档/图谱内容相对静态，缓存安全）。
# NL2SQL 数据查询结果不缓存 —— 运营数据时效性敏感，避免返回过期计数。
#
# ★ fail-open：Redis 不可用时读返回 None（当未命中），写忽略，
#   缓存故障绝不影响主链路。
# ============================================================

from __future__ import annotations

import json

from loguru import logger

from src.core.config import get_settings

settings = get_settings()


def _client():
    import redis.asyncio as aioredis

    from src.infra.redis_cache import redis_pool

    return aioredis.Redis(connection_pool=redis_pool)


async def get_json_cache(key: str) -> dict | None:
    """读缓存。未命中 / Redis 异常 → None。"""
    if not settings.QUERY_CACHE_ENABLED:
        return None
    try:
        raw = await _client().get(key)
        if raw:
            return json.loads(raw)
    except Exception as e:
        logger.warning(f"查询缓存读失败: {e}")
    return None


async def set_json_cache(key: str, value: dict, ttl: int | None = None) -> None:
    """写缓存（JSON 序列化）。ttl 默认取 settings.QUERY_CACHE_TTL。"""
    if not settings.QUERY_CACHE_ENABLED:
        return
    ttl = ttl if ttl is not None else settings.QUERY_CACHE_TTL
    try:
        await _client().set(key, json.dumps(value, ensure_ascii=False), ex=ttl)
    except Exception as e:
        logger.warning(f"查询缓存写失败: {e}")


def build_search_cache_key(
    question: str,
    channels: list[str],
    doc_type: str,
    model_code: str,
    use_hyde: bool,
    role: str,
) -> str:
    """知识检索缓存 key。role 参与 key（影响 NL2SQL 行过滤），user 不参与（跨用户复用）。"""
    import hashlib

    payload = f"{question}|{sorted(channels)}|{doc_type}|{model_code}|{use_hyde}|{role}"
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    # v2：image_urls 语义改为"答案引用的图片"，旧缓存全部失效
    return f"alm_cache:search:v2:{digest}"
