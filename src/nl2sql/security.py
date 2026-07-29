# ============================================================
# SQL 安全校验 — 四层防线
#
# 1. Prompt 层：SCHEMA_DESC 人工裁剪（vin/reporter_phone/customer_name 不列出）
# 2. 正则层：FORBIDDEN_PATTERNS
# 3. 执行层：SELECT-only + LIMIT 100 + timeout 10s
# 4. 数据库层：只读副本
#
# ★ _inject_where 用 sqlglot AST 注入（不用 str.replace）
# ★ apply_role_filter 支持 5 种角色
# ============================================================

import re

import sqlglot
import sqlglot.expressions as exp

FORBIDDEN_PATTERNS = [
    re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE)\b", re.IGNORECASE),
    re.compile(r"\balm_issues\b[^;]*\b(reporter_phone|vin|customer_name)\b", re.IGNORECASE),
]


def validate_sql(sql: str) -> tuple[bool, str]:
    """校验 SQL 安全性。返回 (is_valid, validated_sql_or_error)"""
    stripped = sql.strip().rstrip(";")

    if not stripped.upper().startswith("SELECT"):
        return False, "只允许 SELECT 查询"

    for pattern in FORBIDDEN_PATTERNS:
        if pattern.search(stripped):
            return False, "查询包含禁止的操作或字段"

    if "LIMIT" not in stripped.upper():
        stripped += " LIMIT 100"

    return True, stripped


def apply_role_filter(
    sql: str,
    role: str,
    business_line: str | None = None,
    owner_domain_id: int | None = None,
) -> tuple[bool, str]:
    """
    四类角色行级过滤。返回 (allowed, modified_sql)。
    customer 直接拒绝。
    """
    if role == "customer":
        return False, "当前角色无数据查询权限"

    if role == "admin":
        return True, sql

    condition = None
    if role == "engineer" and owner_domain_id is not None:
        condition = f"owner_domain_id = {owner_domain_id}"
    elif role == "business" and business_line:
        condition = f"business_line = '{business_line}'"
    elif role == "aftersales":
        condition = "status IN ('closed', 'verified')"

    if condition:
        sql = _inject_where_ast(sql, condition)
    return True, sql


def _inject_where_ast(sql: str, condition: str) -> str:
    """★ sqlglot AST 层注入 WHERE，不靠字符串替换。
    彻底解决医疗版 str.replace("WHERE", ...) 在子查询/CTE 中注入错误位置的问题。
    同时检测条件列名是否已存在，避免重复注入。"""
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
        condition_expr = sqlglot.parse_one(condition, dialect="postgres")

        # ★ 去重：如果 WHERE 里已经有同名列，跳过注入
        col_name = _extract_column_name(condition)
        where = tree.find(exp.Where)
        if where and col_name:
            existing_cols = {c.name for c in where.find_all(exp.Column) if hasattr(c, 'name')}
            if col_name in existing_cols:
                return _fix_sqlglot_output(tree.sql(dialect="postgres"))

        if where:
            where.set("this", exp.And(this=where.this, expression=condition_expr))
        else:
            tree.set("where", exp.Where(this=condition_expr))

        return _fix_sqlglot_output(tree.sql(dialect="postgres"))
    except Exception:
        # sqlglot 解析失败则降级为简单注入
        upper = sql.upper()
        if "WHERE" in upper:
            idx = upper.index("WHERE") + 5
            return sql[:idx] + f" {condition} AND" + sql[idx:]
        elif "LIMIT" in upper:
            idx = upper.index("LIMIT")
            return sql[:idx] + f" WHERE {condition} " + sql[idx:]
        elif "ORDER" in upper:
            idx = upper.index("ORDER")
            return sql[:idx] + f" WHERE {condition} " + sql[idx:]
        else:
            return f"SELECT * FROM ({sql}) AS _filtered WHERE {condition}"


def _fix_sqlglot_output(sql: str) -> str:
    """修复 sqlglot 输出中 PostgreSQL 不兼容的语法。
    例如 sqlglot 会把 INTERVAL '3 months' 改写为 INTERVAL '3' MONTHS（复数不合法）。"""
    return re.sub(
        r"INTERVAL\s+'(\d+)'\s+(DAYS|HOURS|MONTHS|YEARS|WEEKS|MINUTES|SECONDS)",
        r"INTERVAL '\1 \2'",
        sql,
        flags=re.IGNORECASE,
    )


def _extract_column_name(condition: str) -> str | None:
    """从注入条件中提取列名，用于去重检测。例如 'owner_domain_id = 5' → 'owner_domain_id'"""
    try:
        col = sqlglot.parse_one(condition, dialect="postgres").find(exp.Column)
        return col.name if col and hasattr(col, 'name') else None
    except Exception:
        return None
