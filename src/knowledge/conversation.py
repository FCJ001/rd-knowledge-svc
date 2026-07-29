# ============================================================
# 多轮对话上下文管理（Redis）
# key: alm_ctx:{user_id}:{session_id}  TTL 1800
# ★ 新增 last_sql 字段支持下钻追问
# ============================================================

from __future__ import annotations

import json

from loguru import logger

CONTEXT_TTL = 1800
MAX_HISTORY = 10


async def load_conversation_context(
    redis_client,
    user_id: str,
    session_id: str,
) -> list[dict]:
    """从 Redis 加载对话上下文"""
    key = f"alm_ctx:{user_id}:{session_id}"
    try:
        raw = await redis_client.get(key)
        if raw:
            return json.loads(raw)
    except Exception as e:
        logger.warning(f"加载对话上下文失败: {e}")
    return []


async def save_conversation_context(
    redis_client,
    user_id: str,
    session_id: str,
    history: list[dict],
) -> None:
    """保存对话上下文到 Redis"""
    key = f"alm_ctx:{user_id}:{session_id}"
    trimmed = history[-MAX_HISTORY:]
    try:
        await redis_client.set(key, json.dumps(trimmed, ensure_ascii=False), ex=CONTEXT_TTL)
    except Exception as e:
        logger.warning(f"保存对话上下文失败: {e}")


async def append_turn(
    redis_client,
    user_id: str,
    session_id: str,
    question: str,
    answer: str,
    last_sql: str = "",  # ★ 新增：支持下钻
) -> list[dict]:
    """追加一轮对话到上下文"""
    history = await load_conversation_context(redis_client, user_id, session_id)
    history.append({"role": "user", "content": question})
    assistant_msg = {"role": "assistant", "content": answer[:500]}
    if last_sql:
        assistant_msg["sql"] = last_sql  # ★ 下钻时 LLM 参考上一条 SQL
    history.append(assistant_msg)
    await save_conversation_context(redis_client, user_id, session_id, history)
    return history


def format_conversation_context(history: list[dict]) -> str:
    """将对话历史格式化为 Prompt 字符串（最近 6 条）"""
    if not history:
        return ""
    parts = []
    for turn in history[-6:]:
        role = "用户" if turn["role"] == "user" else "助手"
        content = turn["content"]
        if turn.get("sql"):
            content += f"  [SQL: {turn['sql']}]"
        parts.append(f"{role}：{content}")
    return "以下是之前的对话上下文：\n" + "\n".join(parts) + "\n\n"
