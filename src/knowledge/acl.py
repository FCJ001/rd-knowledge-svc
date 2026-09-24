# ============================================================
# 文档级权限（ACL）—— 过滤必须在检索之前生效
#
# ★ 为什么不能"先检索再过滤"：
#   1) top_k 被无权内容吃掉。用户拿到的是"没找到"，而库里其实有他能看的内容——
#      表现为召回率无故下降，且无法从结果侧区分"确实没有"和"被权限挡了"。
#   2) 权限判定会散落到结果处理的每一处（上下文 / 图片 / 引用 / 缓存 / 审计），
#      漏一处就是泄露。过滤进查询本身，无权内容从不进入候选集，就没有"漏一处"。
#
# 所以本模块只产出一个东西：拼进 Milvus 布尔表达式的 ACL 谓词
# （array_contains_any(acl_roles, [<role>])），由检索层原样放进 AnnSearchRequest。
# 不做任何 Python 侧的结果过滤——那是本模块刻意不提供的接口。
#
# 角色词表与 UserContext.role 一致（engineer | business | aftersales | customer | admin）。
# admin 是超级角色，绕过 ACL（与 rd-chatBI 的 role_rules=all 口径一致）。
# ============================================================

from __future__ import annotations

from loguru import logger

from src.core.config import get_settings
from src.infra.milvus_client import escape_milvus_string

ADMIN_ROLE = "admin"

# 可被授予文档可见性的业务角色（admin 不在此列：它是绕过，不是被授予）
ASSIGNABLE_ROLES: frozenset[str] = frozenset(
    {"engineer", "business", "aftersales", "customer"}
)

# Milvus 侧 ACL 字段：ARRAY<VARCHAR>，每个 chunk 一行可见角色列表
ACL_FIELD = "acl_roles"


class AclConfigError(ValueError):
    """ACL 配置/取值非法（未知角色、空值等）—— 入库期直接拒绝，不静默放行。"""


def parse_acl_roles(raw: str | None) -> list[str]:
    """解析逗号分隔的可见角色列表。

    ★ 未知角色直接报错而不是丢弃：拼错的角色名（如 `enginer`）若被静默丢掉，
    结果是"这份文档谁（除 admin）都看不见"——一个只能靠用户投诉发现的问题。
    宁可让上传者在入库时就收到 400。

    admin 会被剔除：admin 恒可见，把它写进列表只会让"谁能看"变得难以推理。
    """
    roles: list[str] = []
    for item in (raw or "").split(","):
        role = item.strip()
        if not role or role == ADMIN_ROLE:
            continue
        if role not in ASSIGNABLE_ROLES:
            raise AclConfigError(
                f"未知角色 {role!r}；可用角色: {sorted(ASSIGNABLE_ROLES)}"
            )
        if role not in roles:
            roles.append(role)
    return sorted(roles)


def default_acl_roles() -> list[str]:
    """入库未显式指定时的默认可见角色（DOC_ACL_DEFAULT_ROLES）。

    默认空 = 仅 admin 可见（fail-closed）：新语料默认对所有人不可见，
    上传者必须显式声明可见范围。反过来的默认（默认所有人可见）在敏感语料
    上是静默泄露——不可见至少是能被发现、能被修的问题。
    """
    return parse_acl_roles(get_settings().DOC_ACL_DEFAULT_ROLES)


def effective_acl_roles(explicit: str | None) -> list[str]:
    """决定一份文档最终写入的可见角色。

    explicit=None → 取配置默认值；explicit="" → 显式声明"仅 admin"（空列表）。
    """
    if explicit is None:
        return default_acl_roles()
    return parse_acl_roles(explicit)


def doc_acl_expr(role: str) -> str | None:
    """构造文档检索的 ACL 谓词（并进 Milvus 布尔表达式）。

    返回 None 表示"本次查询无需 ACL 约束"，只有两种情形：
      - DOC_ACL_ENABLED=false（演练/排障开关，prod 下不允许）；
      - 调用方是 admin（超级角色）。
    其余角色一律返回谓词；未知角色返回的谓词匹配不到任何文档（fail-closed）。

    ★ 返回值必须是"已经定好的字符串"，不接受调用方再拼装：拼装点越多，
    漏转义/漏括号的面越大。角色取值已按词表校验，仍统一走转义做纵深防御。
    """
    if not get_settings().DOC_ACL_ENABLED:
        logger.warning(f"DOC_ACL_ENABLED=false：文档检索未做权限过滤（role={role}）")
        return None
    if role == ADMIN_ROLE:
        logger.info("admin 角色检索：跳过文档 ACL（超级角色）")
        return None
    if role not in ASSIGNABLE_ROLES:
        # 词表外的角色（token 被篡改/上游新增角色未同步）：谓词匹配不到任何文档。
        # 不报错、不降级为放行——检索侧 fail-closed，日志留痕供排查。
        logger.warning(f"角色 {role!r} 不在 ACL 词表内：按无可见文档处理（fail-closed）")
    escaped = escape_milvus_string(role)
    return f'array_contains_any({ACL_FIELD}, ["{escaped}"])'


def graph_read_allowed(role: str) -> bool:
    """图谱通道是否允许该角色检索。

    ★ 图谱节点级 ACL 未实现：Neo4j 是 Community 版（无细粒度权限），
    Cypher 又由 LLM 生成、没有结构化校验手段——任何"在生成的 Cypher 里
    补一句 WHERE business_line"的做法都只是正则把关，绕过方式比想象的多
    （prompt 注入改的就是这段生成结果）。
    做不到就别假装做到：非 admin 默认拒绝该通道，需要时由部署方显式承担风险。
    """
    if get_settings().GRAPH_ACL_ALLOW_NON_ADMIN:
        logger.warning(
            f"GRAPH_ACL_ALLOW_NON_ADMIN=true：图谱通道对非 admin 开放且无节点级 ACL（role={role}）"
        )
        return True
    return role == ADMIN_ROLE
