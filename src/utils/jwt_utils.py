# ============================================================
# JWT 工具 — 只验签不签发
#
# token 由上游（项目一或 Java 网关）签发，本服务只负责验证：
#   - RS256（非对称）：配 JWT_PUBLIC_KEY（推荐，本服务不持有签发能力）
#   - HS256（对称）：配 JWT_SECRET
# ★ 密钥一律从环境注入（.env / KMS），绝不在代码里硬编码
# ============================================================

import jwt

from src.core.config import get_settings


def verify_jwt(token: str) -> dict:
    """验证 JWT token，返回 payload。失败抛 jwt.InvalidTokenError 系异常。"""
    settings = get_settings()
    key = settings.JWT_PUBLIC_KEY if settings.JWT_ALGORITHM.upper() == "RS256" else settings.JWT_SECRET
    if not key:
        # 拒绝用空密钥验签（等于不设防），由调用方转 401
        raise jwt.InvalidTokenError("JWT 验签密钥未配置（JWT_SECRET / JWT_PUBLIC_KEY）")
    return jwt.decode(
        token,
        key=key,
        algorithms=[settings.JWT_ALGORITHM],
        options={"require": ["exp"]},  # 必须带过期时间
    )
