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


class DashScopeLiteLLM(LiteLLM):
    """DashScope 兼容的裁判 Provider。

    ★ DashScope 的 OpenAI 兼容端点要求：response_format 用 json_object/json_schema 时，
    messages 里必须出现字面量 "json"，否则 400（InvalidParameter）。
    TruLens 裁判（relevance_with_cot_reasons 等）走 Pydantic 结构化输出，
    litellm 对不支持 json_schema 的端点降级为 json_object，而其内置 prompt
    不带 "json" 字样 → 每次裁判调用必挂。
    这里在发送前给 prompt/messages 补上 JSON 字样，结构化输出照常生效。"""

    _JSON_HINT = "\n请以 JSON 格式输出结果。"

    def _create_chat_completion(self, prompt=None, messages=None,
                                response_format=None, **kwargs):
        if response_format is not None:
            if prompt is not None:
                prompt = f"{prompt}{self._JSON_HINT}"
            elif messages:
                messages = [dict(m) for m in messages]
                if messages[0].get("role") == "system":
                    messages[0]["content"] = (
                        messages[0].get("content", "") + self._JSON_HINT
                    )
                else:
                    messages.insert(
                        0, {"role": "system", "content": "请以 JSON 格式输出结果。"}
                    )
        return super()._create_chat_completion(
            prompt=prompt, messages=messages,
            response_format=response_format, **kwargs,
        )


def get_llm_provider() -> LiteLLM:
    """评估用 LLM Provider（LLM-as-Judge）。前缀 openai/ 走 OpenAI 兼容协议。"""
    return DashScopeLiteLLM(
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
