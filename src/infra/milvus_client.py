# ============================================================
# Milvus 连接管理 — 统一 MilvusClient 风格
#
# 天宫医疗版两种客户端混用（legacy connections alias vs 新版 MilvusClient），
# 项目二统一到 MilvusClient + 进程级单例。
# ★ 客户端级 timeout：兜底所有 gRPC 调用，防慢查询无界挂起。
# ★ pymilvus 是同步客户端：async 代码里调用必须包 asyncio.to_thread，
#   否则一个慢查询冻结整个事件循环。
# ============================================================

import re
import threading

from pymilvus import MilvusClient

from src.core.config import get_settings

_client: MilvusClient | None = None
_lock = threading.Lock()

# doc_id 是 md5(doc_name)[:16]（hex），白名单校验防表达式注入
_DOC_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def get_milvus_client() -> MilvusClient:
    """返回进程级单例 MilvusClient"""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                settings = get_settings()
                _client = MilvusClient(
                    uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
                    timeout=settings.MILVUS_TIMEOUT,
                )
    return _client


def escape_milvus_string(value: str) -> str:
    """转义 Milvus 布尔表达式字符串字面量，防表达式注入。

    用户可控的 doc_type/model_code 等拼进 filter 前必须经过这里，
    否则值中含双引号即可 breakout（如 `x" or doc_id != "` 清空 collection）。"""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def assert_valid_doc_id(doc_id: str) -> str:
    """校验 doc_id 字符集（字母/数字/下划线/连字符），非法直接拒绝。"""
    if not _DOC_ID_RE.match(doc_id or ""):
        raise ValueError(f"非法 doc_id: {doc_id!r}")
    return doc_id


def check_milvus_health() -> bool:
    """检查 Milvus 连通性"""
    try:
        client = get_milvus_client()
        client.list_collections()
        return True
    except Exception:
        return False


def close_milvus_client() -> None:
    """关闭 Milvus 连接"""
    global _client
    if _client is not None:
        _client.close()
        _client = None
