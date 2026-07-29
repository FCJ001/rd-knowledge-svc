# ============================================================
# Milvus 连接管理 — 统一 MilvusClient 风格
#
# 天宫医疗版两种客户端混用（legacy connections alias vs 新版 MilvusClient），
# 项目二统一到 MilvusClient + 进程级单例。
# ============================================================

import threading

from pymilvus import MilvusClient

from src.core.config import get_settings

_client: MilvusClient | None = None
_lock = threading.Lock()


def get_milvus_client() -> MilvusClient:
    """返回进程级单例 MilvusClient"""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                settings = get_settings()
                _client = MilvusClient(
                    uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
                )
    return _client


def check_milvus_health() -> bool:
    """检查 Milvus 连通性"""
    try:
        client = get_milvus_client()
        collections = client.list_collections()
        return True
    except Exception:
        return False


def close_milvus_client() -> None:
    """关闭 Milvus 连接"""
    global _client
    if _client is not None:
        _client.close()
        _client = None
