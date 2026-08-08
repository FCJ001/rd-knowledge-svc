# ============================================================
# TokenTracker 单元测试
# 覆盖：usage_metadata=None 不崩溃（修复 AttributeError）/
#       llm_output token_usage 路径 / usage 聚合
# ============================================================

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from src.rag.evaluation.token_tracker import TokenTracker


def test_usage_metadata_none_no_crash():
    # langchain 的 AIMessage.usage_metadata 默认是 None（hasattr 恒为 True）
    # → 修复前 um.get() 抛 AttributeError
    msg = AIMessage(content="hi", usage_metadata=None)
    result = LLMResult(generations=[[ChatGeneration(message=msg)]], llm_output={})

    tracker = TokenTracker()
    tracker.on_llm_end(result)  # 不应抛异常

    assert tracker.usage["calls"] == 1
    assert tracker.usage["total_tokens"] == 0


def test_llm_output_token_usage_counted():
    result = LLMResult(generations=[[]], llm_output={
        "token_usage": {"prompt_tokens": 10, "completion_tokens": 20},
        "model_name": "qwen-max",
    })

    tracker = TokenTracker()
    tracker.on_llm_end(result)

    assert tracker.usage["input_tokens"] == 10
    assert tracker.usage["output_tokens"] == 20
    assert tracker.usage["calls"] == 1
