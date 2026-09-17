# ============================================================
# FastAPI 依赖：身份注入 + 分页参数
#
# AUTH_MODE=jwt（生产）：验 Authorization: Bearer <token>，解 claims 构造身份。
# AUTH_MODE=header（开发期）：从 HTTP header 取身份（由项目一或网关透传）。
#   ★ header 模式下网关必须剥离/覆写外部传入的 X-User-* 头，否则身份可伪造。
#
# 调试（header 模式）：
#   curl -H "X-User-Id: 1" -H "X-User-Role: engineer" -H "X-Owner-Domain-Id: 5" ...
# ============================================================

from dataclasses import dataclass

import jwt as pyjwt
from fastapi import Header, HTTPException, Query

from src.core.config import get_settings
from src.core.logger import logger


@dataclass
class UserContext:
    """
    全链路身份上下文。
    同时喂给 NL2SQL 行过滤和 RAG model_code 过滤。
    """
    user_id: str
    session_id: str = ""
    role: str = "customer"              # engineer | business | aftersales | customer | admin
    business_line: str | None = None
    owner_domain_id: int | None = None


def _user_from_claims(claims: dict) -> UserContext:
    """JWT claims → 身份上下文。兼容 sub/user_id 两种主流 claim 命名。"""
    owner_domain_id = claims.get("owner_domain_id")
    try:
        owner_domain_id = int(owner_domain_id) if owner_domain_id is not None else None
    except (TypeError, ValueError):
        owner_domain_id = None
    return UserContext(
        user_id=str(claims.get("sub") or claims.get("user_id") or ""),
        session_id=str(claims.get("session_id") or ""),
        role=str(claims.get("role") or "customer"),
        business_line=claims.get("business_line"),
        owner_domain_id=owner_domain_id,
    )


async def get_current_user(
    authorization: str = Header("", alias="Authorization"),
    x_user_id: str = Header("", alias="X-User-Id"),
    x_session_id: str = Header("", alias="X-Session-Id"),
    x_user_role: str = Header("customer", alias="X-User-Role"),
    x_business_line: str | None = Header(None, alias="X-Business-Line"),
    x_owner_domain_id: int | None = Header(None, alias="X-Owner-Domain-Id"),
) -> UserContext:
    """身份注入。jwt 模式验签失败/缺凭证一律 401，不降级到 header 信任。"""
    settings = get_settings()

    if settings.AUTH_MODE == "jwt":
        token = authorization[7:].strip() if authorization.startswith("Bearer ") else authorization.strip()
        if not token:
            raise HTTPException(status_code=401, detail="缺少认证凭证")
        try:
            from src.utils.jwt_utils import verify_jwt
            claims = verify_jwt(token)
        except pyjwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="token 已过期")
        except pyjwt.InvalidTokenError as e:
            logger.warning(f"JWT 验签失败: {e}")
            raise HTTPException(status_code=401, detail="非法 token")
        return _user_from_claims(claims)

    # header 模式：开发期实现，直接信任网关透传的身份头
    return UserContext(
        user_id=x_user_id,
        session_id=x_session_id,
        role=x_user_role,
        business_line=x_business_line,
        owner_domain_id=x_owner_domain_id,
    )


class PageParams:
    """分页参数依赖"""
    def __init__(
        self,
        page: int = Query(1, ge=1, description="页码"),
        page_size: int = Query(20, ge=1, le=100, description="每页条数"),
        keyword: str | None = Query(None, description="搜索关键词"),
    ):
        self.page = page
        self.page_size = page_size
        self.keyword = keyword

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size
