# ============================================================
# TruLens 2.x 会话配置 + LLM Provider（LLM-as-Judge）
# ★ 评估结果存入 PG 数据库 trulens_eval
# ============================================================

from trulens.core import TruSession
from trulens.providers.litellm import LiteLLM

from src.core.config import get_settings

settings = get_settings()


def get_trulens_session() -> TruSession:
    """评估结果存入独立的 PostgreSQL 数据库"""
    return TruSession(
        database_url=(
            f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}"
            f"@{settings.DB_HOST}:{settings.DB_PORT}/trulens_eval"
        )
    )


def get_llm_provider() -> LiteLLM:
    """评估用 LLM Provider（LLM-as-Judge）。前缀 openai/ 走 OpenAI 兼容协议。"""
    return LiteLLM(
        model_engine=f"openai/{settings.CHAT_MODEL}",
        completion_kwargs={
            "api_key": settings.DASHSCOPE_API_KEY,
            "api_base": settings.BASE_URL_CHAT,
        },
    )


def launch_dashboard(session: TruSession | None = None, port: int = 8501):
    """启动 TruLens Streamlit Dashboard"""
    from trulens.dashboard import run_dashboard

    s = session or get_trulens_session()
    run_dashboard(session=s, port=port)


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    launch_dashboard(port=port)
