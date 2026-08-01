# ============================================================
# Guardrails — 同步阻断层
#
# 在 LLM 调用前 / SQL 执行前检查，拦截危险行为：
#   1. SQL 注入检测
#   2. DDL 拦截（DROP/TRUNCATE/ALTER）
#   3. 无 WHERE 的 DELETE/UPDATE 拦截
#   4. 输出中的敏感信息检测
#
# 用法：
#   from src.rag.evaluation.guardrails import check_sql, check_output
#   ok, reason = check_sql(sql)    # → (True, "") 或 (False, "理由")
# ============================================================

from __future__ import annotations

import re

from src.core.config import get_settings


# ── SQL 危险操作模式 ─────────────────────────────────────────────────────

_DDL_PATTERNS = [
    (re.compile(r"\bDROP\s+(TABLE|DATABASE|SCHEMA|INDEX|VIEW)\b", re.IGNORECASE), "包含 DROP 操作"),
    (re.compile(r"\bTRUNCATE\s+(TABLE\s+)?\w+", re.IGNORECASE), "包含 TRUNCATE 操作"),
    (re.compile(r"\bALTER\s+(TABLE|DATABASE|SYSTEM)\b", re.IGNORECASE), "包含 ALTER 操作"),
]

_DML_WITHOUT_WHERE = [
    (re.compile(r"\bDELETE\s+FROM\s+\w+(?!.*\bWHERE\b)", re.IGNORECASE | re.DOTALL), "DELETE 缺少 WHERE 条件"),
    (re.compile(r"\bUPDATE\s+\w+\s+SET\s+.+?(?!.*\bWHERE\b)", re.IGNORECASE | re.DOTALL), "UPDATE 缺少 WHERE 条件"),
]

# 危险函数（可能造成信息泄露或执行系统命令）
_DANGEROUS_FUNCTIONS = [
    (re.compile(r"\bpg_read_file\b", re.IGNORECASE), "调用了文件读取函数 pg_read_file"),
    (re.compile(r"\bpg_read_binary_file\b", re.IGNORECASE), "调用了文件读取函数 pg_read_binary_file"),
    (re.compile(r"\blo_export\b", re.IGNORECASE), "调用了文件导出函数 lo_export"),
    (re.compile(r"\bdblink\b", re.IGNORECASE), "调用了外部连接函数 dblink"),
]

# 敏感信息检测（输出）
_SENSITIVE_PATTERNS = [
    (re.compile(r"\b\d{15,19}\b"), "疑似身份证/银行卡号"),
    (re.compile(r"\b1[3-9]\d{9}\b"), "疑似手机号"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "疑似邮箱地址"),
]


def check_sql(sql: str) -> tuple[bool, str]:
    """
    检查 SQL 语句是否安全。
    返回 (is_safe, reason)
    """
    settings = get_settings()
    if not settings.GUARDRAILS_ENABLED:
        return True, ""

    if not sql or not sql.strip():
        return True, ""

    # 1. DDL 检查
    if settings.GUARDRAILS_BLOCK_DDL:
        for pattern, desc in _DDL_PATTERNS:
            if pattern.search(sql):
                return False, f"[Guardrails] {desc}"

    # 2. 危险函数检查
    for pattern, desc in _DANGEROUS_FUNCTIONS:
        if pattern.search(sql):
            return False, f"[Guardrails] {desc}"

    # 3. 无 WHERE 的 DELETE/UPDATE
    if settings.GUARDRAILS_BLOCK_DML_WITHOUT_WHERE:
        for pattern, desc in _DML_WITHOUT_WHERE:
            if pattern.search(sql):
                return False, f"[Guardrails] {desc}"

    return True, ""


def check_output(text: str) -> tuple[bool, str]:
    """
    检查输出文本是否包含敏感信息。
    返回 (is_safe, reason)
    """
    settings = get_settings()
    if not settings.GUARDRAILS_ENABLED:
        return True, ""

    if not text:
        return True, ""

    for pattern, desc in _SENSITIVE_PATTERNS:
        matches = pattern.findall(text)
        if len(matches) > 5:  # 超过 5 个匹配才告警，避免误报
            return False, f"[Guardrails] 输出包含{desc}（{len(matches)}处匹配）"

    return True, ""
