# ============================================================
# JWT 工具 — 只验签不签发
#
# 项目二的 token 由项目一或 Java 网关签发（RS256），
# 本服务只负责验证。开发期走 header mock，不验签。
# ============================================================

import jwt

SECRET_KEY = "12af38e3ab85909849bfe0b89f89075d7677438a0f14c0304a46249ae513558d"
ALGORITHM = "HS256"


def verify_jwt(token: str) -> dict:
    """验证 JWT token，返回 payload"""
    try:
        payload = jwt.decode(token, key=SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise Exception("token已过期")
    except jwt.InvalidTokenError:
        raise Exception("非法token")
