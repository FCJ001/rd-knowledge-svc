# ============================================================
# 权限校验与生产配置防呆
#
# 此前 role 只是 UserContext 上的一个字符串（admin 是合法值但从未被校验），
# 删除文档这类不可逆操作只要求「已登录」。
# ============================================================

import pytest
from fastapi import HTTPException
from src.core.deps import UserContext, require_role, roles_from_csv


def test_roles_from_csv_parses_and_strips():
    assert roles_from_csv("admin, engineer") == ["admin", "engineer"]
    assert roles_from_csv("") == []
    assert roles_from_csv(None) == []
    assert roles_from_csv(" , ") == []


def test_require_role_rejects_empty_whitelist():
    """空白名单必须构造期报错。

    若放行，实现会退化成「拒绝所有人」（线上 403 而配置看着正常）
    或「放行所有人」（权限洞）——两种都不该由猜测决定。
    """
    with pytest.raises(ValueError, match="至少要指定一个角色"):
        require_role()


async def test_require_role_allows_listed_role():
    checker = require_role("admin", "engineer")
    user = UserContext(user_id="u1", role="engineer")
    assert await checker(user) is user


async def test_require_role_rejects_unlisted_role():
    checker = require_role("admin")
    with pytest.raises(HTTPException) as exc:
        await checker(UserContext(user_id="u1", role="customer"))
    assert exc.value.status_code == 403


def test_env_example_disables_minio_public_read():
    """示例配置不得默认开着公共读——示例是部署的复制源。"""
    from pathlib import Path
    text = Path(__file__).resolve().parent.parent / ".env.example"
    content = text.read_text(encoding="utf-8")
    assert "MINIO_PUBLIC_READ=false" in content
    assert "MINIO_PUBLIC_READ=true" not in content
