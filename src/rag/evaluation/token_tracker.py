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


class TokenTracker(BaseCallbackHandler):
    """LangChain 回调，累计 Token 用量和费用"""

    def __init__(self):
        super().__init__()
        settings = get_settings()
        self._input_price = settings.MODEL_PRICING_INPUT / 1_000_000
        self._output_price = settings.MODEL_PRICING_OUTPUT / 1_000_000
        self._input_tokens = 0
        self._output_tokens = 0
        self._calls = 0

    # ── LangChain callback interface ─────────────────────────────────

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """LLM 调用结束，提取 token 用量"""
        self._calls += 1
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self._input_tokens += usage.get("prompt_tokens", 0)
            self._output_tokens += usage.get("completion_tokens", 0)
            return

        # LangChain 新版 usage_metadata（per-generation）
        for gen in response.generations:
            for g in gen:
                msg = getattr(g, "message", None)
                if msg and hasattr(msg, "usage_metadata"):
                    um = msg.usage_metadata
                    self._input_tokens += um.get("input_tokens", 0)
                    self._output_tokens += um.get("output_tokens", 0)

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
