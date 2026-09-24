# ============================================================
# 配置防呆与精排输入口径
# ============================================================

import pytest
from src.core.config import Settings


def test_prod_rejects_public_minio_read():
    """prod 下开着 MinIO 公共读必须拒绝启动。

    公共读让原文/图片 URL 可被任意人访问，等于绕过文档级权限；
    只写进 README 的"生产清单"是不够的——必须由启动校验强制。
    """
    with pytest.raises(Exception) as exc:
        Settings(APP_ENV="prod")
    assert "MINIO_PUBLIC_READ" in str(exc.value)


def test_prod_rejects_dev_default_passwords():
    with pytest.raises(Exception) as exc:
        Settings(APP_ENV="prod")
    assert "DB_PASSWORD" in str(exc.value)


def test_dev_env_does_not_enforce_prod_checks():
    """非 prod 环境不应被生产校验拦住（本地开发要能直接起）。"""
    s = Settings(APP_ENV="dev")
    assert s.MINIO_PUBLIC_READ is True  # 开发默认开着便于前端直连


def test_rerank_reads_parent_text_not_child():
    """精排必须看生成时真正会用的那段文本（parent_text）。

    若精排读子块而生成用父块，排序依据与最终上下文不是同一段，
    名次与答案质量脱节——这是一处曾真实存在的口径不一致。
    """
    from src.knowledge.reranker import _rerank_text

    doc = {"text": "子块内容", "parent_text": "父块完整内容"}
    assert _rerank_text(doc) == "父块完整内容"


def test_rerank_falls_back_to_child_when_no_parent():
    """非 parent_child 策略下没有 parent_text，必须回落到 text。"""
    from src.knowledge.reranker import _rerank_text

    assert _rerank_text({"text": "仅子块"}) == "仅子块"
    assert _rerank_text({"text": "仅子块", "parent_text": ""}) == "仅子块"


def test_rag_tuning_config_reaches_online_path():
    """RAG_TOP_K / RAG_RERANK_TOP_K 必须对在线链路生效。

    此前这两个配置只有离线消融脚本在读，改 .env 对线上无影响——
    运维会以为调了参而实际没有。现在 search_docs_raw 的 None 默认值
    回落到 Settings，这里锁定该行为。
    """
    import inspect

    from src.knowledge.doc_rag import search_docs_raw, search_docs_with_stages

    # 参数解析在 search_docs_with_stages（两阶段实现），search_docs_raw 是它的薄封装
    src = inspect.getsource(search_docs_with_stages)
    assert "settings.RAG_TOP_K" in src
    assert "settings.RAG_RERANK_TOP_K" in src
    for fn in (search_docs_raw, search_docs_with_stages):
        sig = inspect.signature(fn)
        assert sig.parameters["top_k"].default is None
        assert sig.parameters["rerank_top_k"].default is None


def test_context_max_chars_config_exists():
    """contexts 截断长度必须是可配置项，而不是硬编码在检索逻辑里。"""
    s = Settings(APP_ENV="dev")
    assert s.RAG_CONTEXT_MAX_CHARS >= 500
    assert s.USER_DAILY_TOKEN_QUOTA == 0   # 默认不限额，避免误伤现有流量
    assert s.DOC_RETENTION_DAYS == 0       # 默认永久保留，不默认删数据


def test_rerank_candidate_cap_must_cover_top_k():
    """重排候选上限必须 ≥ RAG_TOP_K。

    否则超出上限的候选 LLM 从未见过、被打 0 分，且 len(parsed) != len(documents)
    会让动态截断静默跳过——实测表现为"改了 RAG_TOP_K 但检索行为莫名其妙"。
    """
    # 合法：两者配齐
    Settings(APP_ENV="dev", RAG_TOP_K=20, RERANK_MAX_CANDIDATES=20)
    Settings(APP_ENV="dev", RAG_TOP_K=10, RERANK_MAX_CANDIDATES=40)

    # 非法：候选上限小于初召条数
    with pytest.raises(Exception) as exc:
        Settings(APP_ENV="dev", RAG_TOP_K=30, RERANK_MAX_CANDIDATES=20)
    assert "RERANK_MAX_CANDIDATES" in str(exc.value)


def test_dynamic_topk_default_is_off():
    """动态断崖截断默认关闭，固定取 RAG_RERANK_TOP_K 条。

    实测（runbook 4.4.3）：开 → 证据中位 2 条、页级召回 89.1%、有据性 0.949；
    关 → 固定 5 条、页级召回 93.5%、有据性 0.965。证据条数与答案质量非单调、
    峰值在 5 条附近，而 RAG_RERANK_TOP_K 本来就是 5。
    """
    s = Settings(APP_ENV="dev")
    assert s.RAG_DYNAMIC_TOPK is False
    assert s.RAG_RERANK_TOP_K == 5


def test_rerank_provider_default_is_dashscope():
    """精排默认后端是 qwen3-rerank 专用模型（2026-09-24 起）。

    专用模型 pointwise 分数确定性、可复现，拒答阈值才可标定；
    LLM 清单式重排（deepseek）分数为自评、有漂移，退为可选后端。
    直接断言字段声明默认值，不依赖本地 .env 是否覆盖。
    """
    assert Settings.model_fields["RERANK_PROVIDER"].default == "dashscope"


def test_prod_dashscope_rerank_requires_api_key():
    """prod 下 RERANK_PROVIDER=dashscope 必须配置 DASHSCOPE_API_KEY。

    精排在检索关键路径上：无密钥时每个查询都会先打一次注定失败的
    API 再回退 RRF 序（多付一次超时等待 + 每条告警日志），启动即拒
    比运行期静默降级好。仅 prod 强制——dev/CI 允许无密钥启动，
    由 rerank_docs 的运行期降级链兜底（回退 RRF 融合序）。
    """
    with pytest.raises(Exception) as exc:
        Settings(APP_ENV="prod", RERANK_PROVIDER="dashscope", DASHSCOPE_API_KEY="")
    assert "DASHSCOPE_API_KEY" in str(exc.value)


def test_dev_dashscope_without_key_still_starts():
    """dev 下无密钥不拦启动（CI 环境没有 .env 与密钥，要能跑测试）。"""
    s = Settings(APP_ENV="dev", RERANK_PROVIDER="dashscope", DASHSCOPE_API_KEY="")
    assert s.RERANK_PROVIDER == "dashscope"
