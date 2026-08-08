# ============================================================
# Token 用量 + 成本追踪
#
# 用法：
#   tracker = TokenTracker()
#   # 传给 LLM 的 callbacks
#   llm.invoke(messages, config={"callbacks": [tracker]})
#   # 查询结果
#   usage = tracker.usage  # → {"input": 1234, "output": 567, "cost": 0.0012}
# ============================================================

from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from src.core.config import get_settings
from src.core.metrics import LLM_CALLS, LLM_TOKENS


class TokenTracker(BaseCallbackHandler):
    """LangChain 回调，累计 Token 用量和费用"""

    def __init__(self, model: str | None = None):
        super().__init__()
        settings = get_settings()
        self._model = model or settings.CHAT_MODEL
        self._input_price = settings.MODEL_PRICING_INPUT / 1_000_000
        self._output_price = settings.MODEL_PRICING_OUTPUT / 1_000_000
        self._input_tokens = 0
        self._output_tokens = 0
        self._calls = 0

    # ── LangChain callback interface ─────────────────────────────────

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """LLM 调用结束，提取 token 用量并埋入 Prometheus 指标"""
        self._calls += 1
        input_tokens = output_tokens = 0
        if response.llm_output and "token_usage" in response.llm_output:
            model_name = response.llm_output.get("model_name")
            if model_name:
                self._model = model_name
            usage = response.llm_output["token_usage"]
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
        else:
            # LangChain 新版 usage_metadata（per-generation）
            for gen in response.generations:
                for g in gen:
                    msg = getattr(g, "message", None)
                    # ★ usage_metadata 默认是 None，hasattr 恒为 True → 需判空，
                    #    否则 um.get() 抛 AttributeError，回调被 LangChain 吞掉、指标静默丢失
                    um = getattr(msg, "usage_metadata", None) if msg else None
                    if um:
                        input_tokens += um.get("input_tokens", 0)
                        output_tokens += um.get("output_tokens", 0)

        self._input_tokens += input_tokens
        self._output_tokens += output_tokens

        # Prometheus 指标
        LLM_CALLS.labels(model=self._model).inc()
        if input_tokens:
            LLM_TOKENS.labels(model=self._model, kind="input").inc(input_tokens)
        if output_tokens:
            LLM_TOKENS.labels(model=self._model, kind="output").inc(output_tokens)

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def usage(self) -> dict:
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "total_tokens": self._input_tokens + self._output_tokens,
            "cost_usd": round(self.cost, 6),
            "calls": self._calls,
        }

    @property
    def cost(self) -> float:
        return self._input_tokens * self._input_price + self._output_tokens * self._output_price

    def reset(self):
        self._input_tokens = 0
        self._output_tokens = 0
        self._calls = 0
