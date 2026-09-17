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
    owner_domain_id: int | None = None,
    business_line: str | None = None,
) -> str:
    """知识检索缓存 key。

    ★ role / owner_domain_id / business_line 必须参与 key：
      nl2sql 通道结果按这三个维度做行级过滤，缺了任何一个都会把
      A 域的查询结果缓存给 B 域用户（跨身份数据泄漏）。
    user_id 不参与（同域同角色下跨用户复用）。"""
    import hashlib

    payload = (
        f"{question}|{sorted(channels)}|{doc_type}|{model_code}|{use_hyde}|{role}"
        f"|{owner_domain_id}|{business_line}"
    )
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    # v3：key 加入行级过滤维度（v2 及以前不同域可能命中同一缓存，必须全部失效）
    return f"alm_cache:search:v3:{digest}"


async def invalidate_search_cache() -> int:
    """文档增删后失效全部检索缓存（SCAN 逐个删除，避免 KEYS 阻塞）。

    fail-open：Redis 异常只记日志，靠 TTL 兜底。返回删除数量。"""
    if not settings.QUERY_CACHE_ENABLED:
        return 0
    deleted = 0
    try:
        client = _client()
        async for key in client.scan_iter(match="alm_cache:search:v3:*", count=200):
            await client.delete(key)
            deleted += 1
    except Exception as e:
        logger.warning(f"检索缓存失效失败（等 TTL 过期兜底）: {e}")
    return deleted
