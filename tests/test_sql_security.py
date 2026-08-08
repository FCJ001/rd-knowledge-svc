# ============================================================
# NL2SQL 安全防线 + Guardrails 单元测试
# 覆盖：validate_sql 四层防线 / apply_role_filter 角色过滤 /
#       _inject_where_ast AST 注入 / _remove_trailing_line_comment /
#       guardrails.check_sql / check_output
# ============================================================

from src.nl2sql.security import (
    _inject_where_ast,
    _remove_trailing_line_comment,
    apply_role_filter,
    validate_sql,
)
from src.rag.evaluation.guardrails import check_output, check_sql


# ── validate_sql ──────────────────────────────────────────────

def test_select_pass():
    ok, sql = validate_sql("SELECT * FROM alm_issues WHERE model_code='EV160'")
    assert ok
    assert "LIMIT 100" in sql


def test_non_select_rejected():
    ok, msg = validate_sql("UPDATE alm_issues SET severity='low'")
    assert not ok
    assert "SELECT" in msg


def test_forbidden_keyword_rejected():
    for bad in ["DROP TABLE alm_issues", "DELETE FROM alm_issues WHERE id=1", "TRUNCATE TABLE t"]:
        ok, _ = validate_sql(bad)
        assert not ok, f"{bad} 应被拦截"


def test_sensitive_field_rejected():
    # FORBIDDEN_PATTERNS 匹配 "引用了 alm_issues 又出现敏感列" 的语句
    ok, _ = validate_sql("SELECT * FROM alm_issues WHERE reporter_phone = '138'")
    assert not ok


def test_existing_limit_not_duplicated():
    ok, sql = validate_sql("SELECT id FROM alm_issues LIMIT 5")
    assert ok
    assert sql.count("LIMIT") == 1


def test_multi_statement_rejected():
    # LLM 面对复合提问可能用分号拼接多条 SELECT，asyncpg 不支持多命令 → 直接拒绝
    ok, msg = validate_sql(
        "SELECT * FROM alm_issues WHERE model_code='EV160'; "
        "SELECT COUNT(*) FROM alm_issues WHERE model_code='EV160'"
    )
    assert not ok
    assert "单条" in msg


def test_with_cte_allowed():
    # WITH ... SELECT 是合法的单条只读查询，sqlglot 中 WITH 挂在 Select 节点上
    ok, sql = validate_sql(
        "WITH recent AS (SELECT COUNT(*) AS c FROM alm_issues WHERE model_code='EV160') "
        "SELECT * FROM recent"
    )
    assert ok
    assert "LIMIT 100" in sql


def test_trailing_comment_stripped():
    ok, sql = validate_sql("SELECT * FROM alm_issues -- 备注")
    assert ok
    assert "--" not in sql
    assert "LIMIT 100" in sql


# ── _remove_trailing_line_comment ─────────────────────────────

def test_comment_inside_string_not_stripped():
    sql = "SELECT 'a--b' AS x FROM t"
    assert _remove_trailing_line_comment(sql) == sql


def test_inline_comment_with_newline_kept():
    sql = "SELECT * FROM t -- 注释\nWHERE id=1"
    # 注释后有换行，不是尾部注释 → 不截断
    assert "WHERE" in _remove_trailing_line_comment(sql)


# ── apply_role_filter ─────────────────────────────────────────

def test_customer_denied():
    ok, msg = apply_role_filter("SELECT 1", role="customer")
    assert not ok


def test_admin_passthrough():
    sql = "SELECT COUNT(*) FROM alm_issues"
    ok, out = apply_role_filter(sql, role="admin")
    assert ok and out == sql


def test_engineer_injects_owner_domain():
    ok, out = apply_role_filter("SELECT id FROM alm_issues LIMIT 5", role="engineer", owner_domain_id=7)
    assert ok
    assert "owner_domain_id = 7" in out
    assert "LIMIT" in out  # 原有 LIMIT 保留


def test_business_injects_business_line():
    ok, out = apply_role_filter("SELECT id FROM alm_issues", role="business", business_line="ev")
    assert ok
    assert "business_line = 'ev'" in out


def test_aftersales_injects_status():
    ok, out = apply_role_filter("SELECT id FROM alm_issues", role="aftersales")
    assert ok
    assert "status IN ('closed', 'verified')" in out


def test_no_duplicate_injection_same_column():
    sql = "SELECT id FROM alm_issues WHERE owner_domain_id = 3"
    ok, out = apply_role_filter(sql, role="engineer", owner_domain_id=7)
    assert ok
    # 已存在同名列，不重复注入
    assert out.count("owner_domain_id") == 1


def test_subquery_where_injected_at_top_level():
    sql = "SELECT * FROM (SELECT * FROM alm_issues WHERE status='open') AS sub LIMIT 5"
    ok, out = apply_role_filter(sql, role="engineer", owner_domain_id=1)
    assert ok
    # 注入发生在顶层 WHERE，而非子查询
    assert "owner_domain_id" in out


# ── guardrails.check_sql ──────────────────────────────────────

def test_guardrails_blocks_ddl():
    ok, reason = check_sql("DROP TABLE alm_issues")
    assert not ok
    assert "DROP" in reason


def test_guardrails_blocks_truncate():
    ok, reason = check_sql("TRUNCATE TABLE alm_issues")
    assert not ok


def test_guardrails_blocks_dangerous_function():
    ok, reason = check_sql("SELECT pg_read_file('/etc/passwd')")
    assert not ok
    assert "pg_read_file" in reason


def test_guardrails_blocks_delete_without_where():
    ok, reason = check_sql("DELETE FROM alm_issues")
    assert not ok
    assert "WHERE" in reason


def test_guardrails_allow_select():
    ok, _ = check_sql("SELECT COUNT(*) FROM alm_issues WHERE model_code='EV160'")
    assert ok


# ── guardrails.check_output ───────────────────────────────────

def test_check_output_few_phones_ok():
    ok, _ = check_output("联系方式已隐藏")  # 无敏感信息
    assert ok


def test_check_output_many_phones_blocked():
    text = " ".join("1380013800%d" % i for i in range(6))
    ok, reason = check_output(text)
    assert not ok
    assert "手机号" in reason


def test_check_output_many_ids_blocked():
    text = " ".join("1%017d" % i for i in range(6))
    ok, reason = check_output(text)
    assert not ok
    assert "身份证" in reason or "银行卡" in reason


def test_check_output_empty_ok():
    ok, _ = check_output("")
    assert ok
