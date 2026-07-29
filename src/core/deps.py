# ============================================================
# FastAPI 依赖：身份注入 + 分页参数
#
# 开发期：直接从 HTTP header 取身份信息（由项目一或网关透传）。
# 生产环境：验 JWT 后解 claims，替换 get_current_user 函数体即可。
#
# 调试：
#   curl -H "X-User-Id: 1" -H "X-User-Role: engineer" -H "X-Owner-Domain-Id: 5" ...
# ============================================================

from dataclasses import dataclass

from fastapi import Header, Query


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


async def get_current_user(
    x_user_id: str = Header("", alias="X-User-Id"),
    x_session_id: str = Header("", alias="X-Session-Id"),
    x_user_role: str = Header("customer", alias="X-User-Role"),
    x_business_line: str | None = Header(None, alias="X-Business-Line"),
    x_owner_domain_id: int | None = Header(None, alias="X-Owner-Domain-Id"),
) -> UserContext:
    """
    开发期实现：直接从 header 取身份，不查表不验签。
    真实环境替换为 verify_jwt(token) 解 claims。
    """
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
