# ============================================================
# 按用户/天的 Token 配额
#
# 与 rate_limit 的分工：
#   rate_limit 管「单位时间请求数」；本模块管「单位时间成本」。
#   两者维度不同——低频的大上下文请求可以轻松绕过请求数限流把预算烧穿，
#   而一次知识检索本身就要 4~5 次 LLM 调用（改写/HyDE/精排/生成/幻觉检测）。
#
# ★ 事后记账、事前拦截：单次请求的花费只有跑完才知道，所以是第 N+1 次请求
#   被已累计的超额拦住。这不是"精确预扣费"，而是"超额即停"的止损机制。
# ★ fail-open：Redis 不可用时放行（与限流一致），配额不是安全边界，
#   不能让它的故障拖垮主链路。
# ============================================================

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import Depends, HTTPException
from loguru import logger

from src.core.config import get_settings
from src.core.deps import UserContext, get_current_user
from src.core.metrics import LLM_COST_USD, QUOTA_REJECTED

settings = get_settings()

# 当天配额键的 TTL：留 1 天冗余，跨零点的残留自然过期
_QUOTA_TTL_S = 2 * 24 * 3600


def _quota_key(user_id: str) -> str:
    day = datetime.now(UTC).strftime("%Y%m%d")
    return f"alm_quota:{day}:{user_id}"


async def get_used_tokens(user_id: str) -> int:
    """读取该用户今日已消耗的 token 数（Redis 异常返回 0，即不拦截）。"""
    try:
        from src.infra.redis_cache import get_redis_client
        client = await get_redis_client()
        raw = await client.get(_quota_key(user_id))
        return int(raw or 0)
    except Exception as e:
        logger.warning(f"配额读取失败，按未超额处理（fail-open）: {e}")
        return 0


async def add_tokens(user_id: str, tokens: int, cost_usd: float = 0.0, model: str = "") -> None:
    """请求结束后累计用量。失败的记账只告警，不影响已生成的答案。"""
    if tokens <= 0:
        return
    if cost_usd > 0:
        LLM_COST_USD.labels(model=model or "unknown").inc(cost_usd)
    try:
        from src.infra.redis_cache import get_redis_client
        client = await get_redis_client()
        key = _quota_key(user_id)
        pipe = client.pipeline()
        pipe.incrby(key, tokens)
        pipe.expire(key, _QUOTA_TTL_S)
        await pipe.execute()
    except Exception as e:
        logger.warning(f"配额写入失败（不影响本次结果）: {e}")


async def check_token_quota(
    user: UserContext = Depends(get_current_user),
) -> None:
    """FastAPI 依赖：今日已用 token 超过配额则 429。

    USER_DAILY_TOKEN_QUOTA <= 0 表示不限额（开发/内网默认）。
    """
    quota = settings.USER_DAILY_TOKEN_QUOTA
    if quota <= 0:
        return
    used = await get_used_tokens(user.user_id)
    if used >= quota:
        QUOTA_REJECTED.labels(scope="user").inc()
        logger.warning(f"Token 配额已用尽: user={user.user_id} used={used} quota={quota}")
        raise HTTPException(
            status_code=429,
            detail=f"今日调用额度已用尽（{quota} tokens），请明日再试或联系管理员",
        )
