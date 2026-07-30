# ============================================================
# Gradio 界面 — 研发知识库 (rd-knowledge-svc)
#
# 运行: python test/test_rag_gradio.py
# 浏览器打开: http://localhost:7861
#
# 布局: 3 Tab — 智能问答 | 数据分析 | 文档中心
# 色调: 深蓝科技风 (#0b1120 底 + #38bdf8 主色)
# ============================================================

from __future__ import annotations

import asyncio
import html
import json
import os
import re
import sys
import tempfile
import uuid
from datetime import datetime
from io import BytesIO
from typing import Any

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import warnings
from dotenv import load_dotenv
from loguru import logger
from PIL import Image

# DashScopeEmbeddings 在 langchain-community 中（deprecated 但仍可用）
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_openai import ChatOpenAI

from src.core.config import get_settings

load_dotenv()
settings = get_settings()

# ══════════════════════════════════════════════════════════════════════
# 模型 & 基础设施 初始化
# ══════════════════════════════════════════════════════════════════════


def _get_llm():
    return ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.BASE_URL_CHAT,
        temperature=0,
    )


def _get_embedding_model():
    return DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )


# ══════════════════════════════════════════════════════════════════════
# 意图路由 — 根据问题自动选择通道
# ══════════════════════════════════════════════════════════════════════

ROUTE_MAP: dict[str, list[str]] = {
    "repair_guide":  ["doc_rag"],
    "spec_query":    ["doc_rag"],
    "tsb_lookup":    ["doc_rag", "graph_rag"],
    "issue_trace":   ["graph_rag"],
    "metric_query":  ["nl2sql"],
    "change_review": ["doc_rag", "graph_rag"],
    "knowledge_qa":  ["doc_rag", "graph_rag"],
}


async def _auto_route(question: str) -> tuple[list[str], str]:
    """LLM 分析意图 → 自动选择通道。返回 (channels, intent_label)"""
    try:
        from src.knowledge.query_rewriter import rewrite_query
        llm = _get_llm()
        result = await rewrite_query(question, llm)
        intent = result.get("intent", "knowledge_qa")
        channels = ROUTE_MAP.get(intent, ["doc_rag", "graph_rag"])
    except Exception:
        intent = "knowledge_qa"
        channels = ["doc_rag", "graph_rag"]

    labels = {
        "repair_guide": "查维修手册",
        "spec_query": "查技术规范",
        "tsb_lookup": "查TSB",
        "issue_trace": "查图谱追溯",
        "metric_query": "查运营数据",
        "change_review": "变更影响分析",
        "knowledge_qa": "综合检索",
    }
    return channels, labels.get(intent, "综合检索")


# ══════════════════════════════════════════════════════════════════════
# 聊天标记生成 — AI 回答渲染为带来源标签的 HTML
# ══════════════════════════════════════════════════════════════════════

def _render_answer(answer: str) -> str:
    """把纯文本回答转成格式化的 HTML，高亮来源引用和警告"""
    text = html.escape(answer)

    # 给 [文档名, 第X页] 加上样式
    text = re.sub(
        r'\[([^\]]+?),\s*第(\d+)页\]',
        r'<span class="citation">[<strong>\1</strong>, 第\2页]</span>',
        text,
    )
    # 给 Markdown 风格的三级标题加粗
    text = re.sub(r'### (.+)', r'<h4>\1</h4>', text)
    # 给 ⚠️ 警告加样式
    text = re.sub(
        r'(⚠️.*)',
        r'<div class="hallucination-warning">\1</div>',
        text,
    )
    # 代码块（SQL / Cypher）
    text = re.sub(
        r'```(.*?)```',
        r'<pre><code>\1</code></pre>',
        text,
        flags=re.DOTALL,
    )

    return f'<div class="answer-content">{text}</div>'


def _build_channel_badges(channels: list[str]) -> str:
    """生成通道来源标签 HTML"""
    icons = {"doc_rag": "📄", "graph_rag": "🔗", "nl2sql": "📊"}
    classes = {"doc_rag": "tag-doc", "graph_rag": "tag-graph", "nl2sql": "tag-sql"}
    names = {"doc_rag": "文档检索", "graph_rag": "知识图谱", "nl2sql": "运营数据"}

    badges = []
    for ch in channels:
        icon = icons.get(ch, "")
        name = names.get(ch, ch)
        cls = classes.get(ch, "tag-doc")
        badges.append(f'<span class="channel-tag {cls}">{icon} {name}</span>')
    return " ".join(badges)


# ══════════════════════════════════════════════════════════════════════
# Tab 1: 智能问答 (Chat 风格 + 自动路由)
# ══════════════════════════════════════════════════════════════════════

async def _chat_handler(
    message: str,
    history: list[dict],
    role: str,
    auto_route: bool,
    manual_channels: list[str],
    model_code: str,
    doc_type: str,
    use_hyde: bool = False,
):
    """核心聊天处理：自动路由 → 多通道检索 → 渲染回答"""
    if not message.strip():
        return history, history

    # ── 决定通道 ──
    if auto_route:
        channels, intent_label = await _auto_route(message)
        channel_note = f"🤖 自动识别意图: {intent_label}"
    else:
        channels = [c for c in ["doc_rag", "graph_rag", "nl2sql"] if c in manual_channels]
        if not channels:
            channels = ["doc_rag", "graph_rag"]
        channel_note = None

    role_map = {
        "研发工程师": "engineer",
        "业务人员": "business",
        "售后人员": "aftersales",
        "管理员": "admin",
    }
    internal_role = role_map.get(role, "engineer")

    # ── 执行检索 ──
    from src.knowledge.fusion import multi_channel_search
    from src.knowledge.doc_rag import extract_image_urls, search_docs_raw
    from src.infra.milvus_client import get_milvus_client
    from src.infra.neo4j_client import get_neo4j_driver
    from src.infra.db import AsyncSessionLocal

    llm = _get_llm()
    embedding_model = _get_embedding_model()
    milvus_client = get_milvus_client()
    neo4j_driver = get_neo4j_driver()

    try:
        async with AsyncSessionLocal() as db:
            answer = await multi_channel_search(
                question=message,
                llm=llm,
                embedding_model=embedding_model,
                milvus_client=milvus_client,
                neo4j_driver=neo4j_driver,
                db_session=db if "nl2sql" in channels else None,
                channels=channels,
                role=internal_role,
                use_hyde=use_hyde,
            )
    except Exception as e:
        answer = f"❌ 检索失败: {e}"

    # ── 提取图片 ──
    image_html = ""
    if "doc_rag" in channels:
        try:
            doc_hits = await search_docs_raw(
                message, embedding_model, milvus_client,
                top_k=10, rerank_top_k=5,
                model_code=model_code or None, doc_type=doc_type or None,
            )
            image_urls = extract_image_urls(doc_hits)
            if image_urls:
                from src.infra.minio_client import download_file
                img_tags = []
                for url in image_urls[:6]:
                    try:
                        parts = url.split(f"{settings.MINIO_BUCKET}/", 1)
                        if len(parts) == 2:
                            img_bytes = download_file(parts[1])
                            b64 = _img_to_base64(img_bytes)
                            img_tags.append(
                                f'<img src="data:image/png;base64,{b64}" '
                                f'class="inline-image" />'
                            )
                    except Exception:
                        pass
                if img_tags:
                    image_html = (
                        '<div class="image-gallery">'
                        + "".join(img_tags)
                        + "</div>"
                    )
        except Exception:
            pass

    # ── 构建 chatbot 消息 ──
    answer_body = _render_answer(answer) + image_html

    badges = _build_channel_badges(channels)
    footer = f'<div class="message-footer">{badges}'
    if channel_note:
        footer += f' <span class="intent-note">{channel_note}</span>'
    footer += "</div>"

    answer_html = f"{answer_body}{footer}"

    new_history = list(history) if history else []
    new_history.append({"role": "user", "content": message})
    new_history.append({"role": "assistant", "content": answer_html})

    return new_history, new_history


def _clear_history():
    return [], []


# ══════════════════════════════════════════════════════════════════════
# Tab 2: 数据分析 (NL2SQL)
# ══════════════════════════════════════════════════════════════════════

async def _bi_query_handler(
    question: str,
    role: str,
    owner_domain_id,
    business_line,
):
    if not question.strip():
        return "", "", "", "请输入查询问题", "", None

    role_map = {
        "研发工程师": "engineer",
        "业务人员": "business",
        "售后人员": "aftersales",
        "管理员": "admin",
    }
    internal_role = role_map.get(role, "engineer")
    owner_domain_id = int(owner_domain_id) if owner_domain_id else None
    business_line = business_line.strip() if business_line.strip() else None

    from src.nl2sql.engine import run_query
    from src.nl2sql.chart_advisor import recommend_chart, render_chart
    from src.infra.db_readonly import ReadOnlySessionLocal

    llm = _get_llm()

    async with ReadOnlySessionLocal() as db:
        result = await run_query(
            question=question, llm=llm, db=db,
            role=internal_role,
            owner_domain_id=owner_domain_id,
            business_line=business_line,
        )

    sql = result.sql or ""
    data_str = (
        json.dumps(
            result.data, ensure_ascii=False, indent=2, default=str,
        )[:8000]
        if result.data
        else ""
    )
    summary = result.summary or ""
    error = result.error or ""
    chart_fig = None
    chart_json = ""
    insights = ""

    if result.success and result.data:
        try:
            config = await recommend_chart(question, result.data, result.columns, llm)
            chart_fig = render_chart(result.data, config)
            chart_json = json.dumps(config, ensure_ascii=False, indent=2, default=str)
            # 生成洞察
            insights = config.get("description", "")
        except Exception as e:
            chart_json = f"图表生成失败: {e}"

    return sql, data_str, summary, error, chart_json, chart_fig, insights


async def _nl2sql_load_history(history_selector: str):
    """加载历史查询 (占位，后续可接 Redis / session)"""
    return "", "", "历史查询功能开发中，请手动输入查询"


# ══════════════════════════════════════════════════════════════════════
# Tab 3: 文档中心 (上传 + 管理)
# ══════════════════════════════════════════════════════════════════════

async def _upload_doc(
    file_path: str,
    doc_type: str,
    category: str,
    model_code: str,
    chunk_strategy: str,
):
    if not file_path:
        return "⚠️ 请先选择文件"

    from src.knowledge.doc_ingestion import ingest_and_index
    from src.infra.db import AsyncSessionLocal

    doc_name = os.path.basename(file_path)
    progress_msgs = []

    async with AsyncSessionLocal() as db:
        try:
            progress_msgs.append(f"🔄 开始解析: {doc_name}")
            doc_id = await ingest_and_index(
                file_path=file_path, doc_name=doc_name, doc_type=doc_type,
                category=category or "通用", business_line="",
                model_code=model_code or "通用", chunk_strategy=chunk_strategy,
                parser="mineru", db=db,
            )
            await db.commit()
            progress_msgs.append("✅ 解析完成")
            # 刷新文档列表
            count_msg, rows = await _list_docs("")
            return (
                f"## ✅ 上传成功\n\n"
                f"| 项目 | 详情 |\n|------|------|\n"
                f"| 文件名 | {doc_name} |\n"
                f"| doc_id | `{doc_id}` |\n"
                f"| 文档类型 | {doc_type} |\n"
                f"| 切片策略 | {chunk_strategy} |\n"
                f"| 车型 | {model_code or '通用'} |\n",
                count_msg, rows,
            )
        except Exception as e:
            return f"## ❌ 上传失败\n\n```\n{e}\n```", "上传失败", []


async def _list_docs(keyword: str):
    from sqlalchemy import select
    from src.knowledge.model import KnowledgeDoc
    from src.infra.db import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as db:
            if keyword.strip():
                stmt = (
                    select(KnowledgeDoc)
                    .where(KnowledgeDoc.doc_name.ilike(f"%{keyword}%"))
                    .order_by(KnowledgeDoc.id.desc())
                    .limit(50)
                )
            else:
                stmt = select(KnowledgeDoc).order_by(KnowledgeDoc.id.desc()).limit(50)

            result = await db.execute(stmt)
            docs = result.scalars().all()

            if not docs:
                return "没有找到文档", []

            rows = [
                [
                    d.doc_id,
                    d.doc_name,
                    d.doc_type,
                    d.category or "-",
                    d.model_code or "-",
                    d.chunk_count or 0,
                    d.chunk_strategy or "-",
                    "✅" if d.status == "active" else d.status,
                ]
                for d in docs
            ]
            return f"共 {len(rows)} 条记录", rows
    except Exception as e:
        return f"查询失败: {e}", []


async def _delete_doc(doc_id: str):
    if not doc_id.strip():
        return "⚠️ 请输入 doc_id", "", []

    from src.knowledge.doc_ingestion import delete_doc as delete_doc_service
    from src.infra.db import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as db:
            await delete_doc_service(doc_id.strip(), db)
            await db.commit()
        count_msg, rows = await _list_docs("")
        return f"✅ 已删除: `{doc_id}`", count_msg, rows
    except Exception as e:
        return f"❌ 删除失败: {e}", "", []


def _refresh_docs(keyword: str):
    return asyncio.run(_list_docs(keyword))


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def _img_to_base64(img_bytes: bytes) -> str:
    import base64
    return base64.b64encode(img_bytes).decode("utf-8")


def _run_async(coro):
    return asyncio.run(coro)


# ══════════════════════════════════════════════════════════════════════
# 主题 & 样式
# ══════════════════════════════════════════════════════════════════════

THEME_JSON = {
    "primary_hue": "sky",
    "secondary_hue": "slate",
    "neutral_hue": "slate",
}


theme = gr.themes.Soft(**THEME_JSON).set(
    # ── 背景 ──
    body_background_fill="#0b1120",
    body_background_fill_dark="#0b1120",
    block_background_fill="#1a2332",
    block_background_fill_dark="#1a2332",
    block_label_background_fill="#1a2332",
    input_background_fill="#1a2332",
    input_background_fill_dark="#1a2332",
    background_fill_primary="#1a2332",
    background_fill_secondary="#1a2332",

    # ── 边框 ──
    border_color_primary="#2d3a4f",
    block_border_color="#2d3a4f",
    input_border_color="#2d3a4f",
    block_border_width="1px",

    # ── 文字 ──
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",
    body_text_color_subdued="#94a3b8",
    block_title_text_color="#e2e8f0",
    input_placeholder_color="#64748b",

    # ── 按钮 ──
    button_primary_background_fill="#38bdf8",
    button_primary_background_fill_hover="#7dd3fc",
    button_primary_text_color="#0b1120",
    button_primary_text_color_hover="#0b1120",
    button_primary_border_color="transparent",
    button_secondary_background_fill="#243044",
    button_secondary_border_color="#2d3a4f",
    button_cancel_background_fill="#7f1d1d",
    button_cancel_background_fill_hover="#991b1b",
    button_cancel_text_color="#fca5a5",

    # ── 圆角 ──
    block_radius="10px",
)

CSS = """
/* ═══════════════════════════════════════════════════
   全局
   ═══════════════════════════════════════════════════ */
body {
    font-size: 14px;
    font-family: "Inter", "Noto Sans SC", system-ui, sans-serif;
}
h1, h2, h3 { color: #e2e8f0 !important; }
code, pre, textarea[class*="mono"] {
    font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", monospace;
}

/* ── 圆角 ── */
.gr-button { border-radius: 8px !important; }
input, textarea, select { border-radius: 8px !important; }
.gr-box, .gr-form { border-radius: 10px !important; }

/* ── Tab ── */
.tabs > .tab-nav > button.selected {
    background: #1a2332 !important;
    color: #38bdf8 !important;
}
.tabs > .tab-nav > button {
    background: transparent !important;
    color: #94a3b8 !important;
}

/* ── Chatbot 代码块 ── */
.chatbot code, .chatbot pre {
    background: #0b1120 !important;
}

/* ── 滚动条 ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0b1120; }
::-webkit-scrollbar-thumb { background: #2d3a4f; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3d4f63; }

/* ═══════════════════════════════════════════════════
   Chatbot 消息
   ═══════════════════════════════════════════════════ */
.bubble-wrap {
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}
.bubble.user {
    background: #1a2332 !important;
    border-radius: 14px 14px 4px 14px !important;
    padding: 14px 18px !important;
    margin: 8px 0 16px 0 !important;
    color: #e2e8f0 !important;
    max-width: 85% !important;
    margin-left: auto !important;
    line-height: 1.6 !important;
}
.bubble.bot {
    background: #0f1d2e !important;
    border-left: 3px solid #38bdf8 !important;
    border-radius: 14px 14px 14px 4px !important;
    padding: 14px 18px !important;
    margin: 8px 0 16px 0 !important;
    color: #e2e8f0 !important;
    max-width: 95% !important;
    margin-right: auto !important;
    line-height: 1.6 !important;
}

/* ── 回答内容 ── */
.answer-content h4 {
    margin: 12px 0 6px 0;
    font-size: 15px;
    color: #e2e8f0;
    border-bottom: 1px solid #2d3a4f;
    padding-bottom: 4px;
}
.answer-content pre {
    background: #0b1120;
    border: 1px solid #2d3a4f;
    border-left: 3px solid #a78bfa;
    border-radius: 6px;
    padding: 12px 16px;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 13px;
    line-height: 1.5;
    overflow-x: auto;
    margin: 8px 0;
}
.answer-content code {
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 12px;
    background: rgba(56,189,248,0.08);
    padding: 2px 6px;
    border-radius: 3px;
}
.answer-content pre code {
    background: transparent;
    padding: 0;
}

/* ── 来源引用 ── */
.citation {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    color: #7dd3fc;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
    margin: 0 2px;
    border: 1px solid rgba(56,189,248,0.25);
}

/* ── 幻觉警告 ── */
.hallucination-warning {
    background: rgba(251,191,36,0.08);
    border-left: 3px solid #fbbf24;
    border-radius: 0 6px 6px 0;
    padding: 8px 14px;
    margin: 10px 0;
    font-size: 13px;
    color: #fcd34d;
}

/* ═══════════════════════════════════════════════════
   通道标签
   ═══════════════════════════════════════════════════ */
.channel-tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 500;
    margin-right: 6px;
}
.tag-doc {
    background: rgba(56,189,248,0.12);
    color: #7dd3fc;
    border: 1px solid rgba(56,189,248,0.3);
}
.tag-graph {
    background: rgba(167,139,250,0.12);
    color: #c4b5fd;
    border: 1px solid rgba(167,139,250,0.3);
}
.tag-sql {
    background: rgba(251,146,60,0.12);
    color: #fdba74;
    border: 1px solid rgba(251,146,60,0.3);
}
.intent-note {
    color: #94a3b8;
    font-size: 11px;
    margin-left: 4px;
}

/* ── 消息脚注 ── */
.message-footer {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid #1e293b;
}

/* ── 图片画廊 ── */
.image-gallery {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 10px;
}
.inline-image {
    max-width: 160px;
    max-height: 120px;
    border-radius: 6px;
    border: 1px solid #2d3a4f;
    object-fit: cover;
    cursor: pointer;
    transition: border-color 0.2s;
}
.inline-image:hover { border-color: #38bdf8; }

/* ═══════════════════════════════════════════════════
   数据分析 Tab
   ═══════════════════════════════════════════════════ */
.sql-display {
    background: #0b1120 !important;
    border-left: 3px solid #a78bfa !important;
    border-radius: 6px !important;
    font-family: "JetBrains Mono", "Fira Code", monospace !important;
    font-size: 13px !important;
}
.sql-display textarea {
    background: #0b1120 !important;
    color: #e2e8f0 !important;
    font-family: "JetBrains Mono", "Fira Code", monospace !important;
}

.data-display {
    background: #1a2332 !important;
    border-radius: 6px !important;
}
.data-display textarea {
    background: #1a2332 !important;
    color: #e2e8f0 !important;
}

.insight-card {
    background: #1a2332;
    border: 1px solid #2d3a4f;
    border-radius: 8px;
    padding: 14px 18px;
    margin-top: 8px;
    color: #e2e8f0;
    font-size: 13px;
    line-height: 1.6;
}

/* ═══════════════════════════════════════════════════
   文档中心 Tab
   ═══════════════════════════════════════════════════ */
.upload-zone {
    border: 2px dashed #2d3a4f !important;
    border-radius: 12px !important;
    padding: 24px !important;
    text-align: center !important;
    transition: border-color 0.25s, background 0.25s !important;
    background: rgba(56,189,248,0.02) !important;
}
.upload-zone:hover {
    border-color: #38bdf8 !important;
    background: rgba(56,189,248,0.04) !important;
}

/* ── 表格 ── */
table {
    border-collapse: collapse !important;
    width: 100% !important;
    font-size: 13px !important;
}
table th {
    background: #243044 !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    padding: 8px 12px !important;
    text-align: left !important;
    border-bottom: 2px solid #2d3a4f !important;
}
table td {
    padding: 7px 12px !important;
    color: #e2e8f0 !important;
    border-bottom: 1px solid #1e293b !important;
}
table tbody tr:hover {
    background: rgba(56,189,248,0.04) !important;
}

/* ═══════════════════════════════════════════════════
   通用
   ═══════════════════════════════════════════════════ */
.status-ok    { color: #34d399 !important; }
.status-warn  { color: #fbbf24 !important; }
.status-error { color: #f87171 !important; }

/* ── Accordion ── */
.accordion {
    border: 1px solid #2d3a4f !important;
    border-radius: 8px !important;
    background: #1a2332 !important;
}

/* ── 按钮增强 ── */
.gr-button-primary:hover {
    box-shadow: 0 0 12px rgba(56,189,248,0.3) !important;
}
.btn-danger {
    background: #7f1d1d !important;
    color: #fca5a5 !important;
    border: 1px solid #991b1b !important;
}
.btn-danger:hover {
    background: #991b1b !important;
    box-shadow: 0 0 8px rgba(248,113,113,0.3) !important;
}

/* ── 过渡动画 ── */
input, textarea, button, select {
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
input:focus, textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
}
"""

# ══════════════════════════════════════════════════════════════════════
# UI 常量
# ══════════════════════════════════════════════════════════════════════

DOC_TYPES = ["repair_manual", "spec_doc", "tsb", "issue_case"]
ROLES = ["研发工程师", "业务人员", "售后人员", "管理员"]
CHUNK_STRATEGIES = ["fixed", "semantic", "parent_child"]
ROLE_MAP_FRIENDLY = {
    "研发工程师": "engineer",
    "业务人员": "business",
    "售后人员": "aftersales",
    "管理员": "admin",
}

# ══════════════════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════════════════

with gr.Blocks(
    title="研发知识库",
    analytics_enabled=False,
) as demo:

    # ── Header ──
    gr.HTML("""
    <div style="text-align:center; padding: 8px 0 20px 0;">
      <h1 style="margin:0; font-size:24px; letter-spacing:-0.5px;">研发知识库 &amp; 运营问答服务</h1>
      <p style="color:#64748b; margin:4px 0 0 0; font-size:13px;">
        文档 RAG · 知识图谱 · NL2SQL/ChatBI &nbsp;|&nbsp; rd-knowledge-svc
      </p>
    </div>
    """)

    # ═══════════════════════════════════════════════════════════════
    # Tab 1: 智能问答
    # ═══════════════════════════════════════════════════════════════

    with gr.Tab("💬 智能问答"):
        # 聊天历史状态
        chat_state = gr.State([])

        with gr.Row():
            # ── 左侧：聊天区 ──
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(
                    value=[],
                    label="对话",
                    height=560,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入你的问题，例如：EV160电池过热怎么排查？系统会自动选择最佳检索通道...",
                        scale=9,
                        show_label=False,
                        container=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

            # ── 右侧：高级设置 ──
            with gr.Column(scale=2):
                with gr.Accordion("⚙️ 高级设置", open=False):
                    role_dropdown = gr.Dropdown(
                        choices=ROLES, value="研发工程师",
                        label="👤 角色",
                    )
                    auto_route_toggle = gr.Checkbox(
                        value=True,
                        label="🤖 自动选择检索通道（推荐）",
                        info="关闭后可手动勾选下方通道",
                    )
                    manual_channels = gr.CheckboxGroup(
                        choices=["doc_rag", "graph_rag", "nl2sql"],
                        value=["doc_rag", "graph_rag"],
                        label="检索通道",
                        info="仅在「自动选择」关闭时生效",
                        interactive=True,
                    )
                    model_code_input = gr.Textbox(
                        label="车型代码",
                        placeholder="如: EV160",
                        info="留空则不限制车型",
                    )
                    doc_type_dropdown = gr.Dropdown(
                        choices=[""] + DOC_TYPES,
                        value="",
                        label="文档类型",
                        info="留空则不限制类型",
                    )
                    use_hyde_toggle = gr.Checkbox(
                        value=False,
                        label="📝 假设文档嵌入 (HyDE)",
                        info="LLM 先将问题扩写成假设文档再检索，提升召回精度（会增加 1-2 秒延迟）",
                    )
                    clear_btn = gr.Button("🗑 清除对话", variant="secondary", size="sm")

                # 精简帮助说明
                gr.Markdown("""
                <div style="font-size:12px; color:#64748b; margin-top:12px;">
                <strong>💡 使用提示</strong><br>
                • 直接提问，系统自动判断该查文档还是查数据<br>
                • 追问时会记住上一轮的上下文<br>
                • 支持车型过滤，避免跨车型串味
                </div>
                """)

        # ── 事件绑定 ──
        def _handle_chat(
            msg, hist, role, auto_route, manual_ch, model_code, doc_type, use_hyde,
        ):
            return asyncio.run(
                _chat_handler(
                    msg, hist, role, auto_route, manual_ch, model_code, doc_type, use_hyde,
                )
            )

        chat_inputs = [
            msg_input, chat_state, role_dropdown,
            auto_route_toggle, manual_channels,
            model_code_input, doc_type_dropdown, use_hyde_toggle,
        ]

        send_event = send_btn.click(
            _handle_chat,
            chat_inputs,
            [chat_state, chatbot],
        ).then(lambda: "", None, [msg_input])

        msg_input.submit(
            _handle_chat,
            chat_inputs,
            [chat_state, chatbot],
        ).then(lambda: "", None, [msg_input])

        clear_btn.click(
            _clear_history, None, [chat_state, chatbot],
        )

    # ═══════════════════════════════════════════════════════════════
    # Tab 2: 数据分析
    # ═══════════════════════════════════════════════════════════════

    with gr.Tab("📊 数据分析"):
        with gr.Row():
            # ── 左侧：查询 ──
            with gr.Column(scale=2):
                gr.Markdown("### 自然语言数据查询")

                bi_question = gr.Textbox(
                    label="查询问题",
                    placeholder="例：最近一周各责任域的 issue 闭环率是多少？",
                    lines=2,
                )
                with gr.Row():
                    bi_role = gr.Dropdown(
                        choices=ROLES, value="研发工程师",
                        label="角色",
                    )
                    bi_domain = gr.Textbox(
                        label="责任域 ID",
                        value="1",
                        info="工程师角色用",
                    )
                    bi_bl = gr.Textbox(
                        label="业务线",
                        value="",
                        info="业务人员角色用",
                    )
                bi_btn = gr.Button("执行查询", variant="primary")

                try:
                    from src.nl2sql.echarts_builder import to_echarts_option
                    _HAS_ECHARTS = True
                except ImportError:
                    _HAS_ECHARTS = False

                if _HAS_ECHARTS:
                    gr.HTML("""
                    <div id="echarts-container"
                         style="width:100%;height:300px;margin-top:12px;
                                background:#1a2332;border-radius:8px;
                                border:1px solid #2d3a4f;display:none;">
                    </div>
                    """)

            # ── 右侧：结果 ──
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("SQL"):
                        bi_sql = gr.Textbox(
                            label="生成的 SQL",
                            lines=6,
                            elem_classes="sql-display",
                        )
                    with gr.TabItem("数据"):
                        bi_data = gr.Textbox(
                            label="查询结果 (JSON)",
                            lines=8,
                            elem_classes="data-display",
                        )
                    with gr.TabItem("图表"):
                        bi_chart_fig = gr.Plot(label="可视化图表")
                        bi_chart_json = gr.Textbox(
                            label="图表配置 (JSON)",
                            lines=4,
                            visible=False,
                        )

                bi_summary = gr.Textbox(
                    label="数据摘要",
                    lines=2,
                    interactive=False,
                )
                bi_error = gr.Textbox(
                    label="错误信息",
                    lines=2,
                    visible=False,
                )
                bi_insights = gr.Textbox(
                    label="💡 数据洞察",
                    lines=2,
                    interactive=False,
                    elem_classes="insight-card",
                )

        # ── 事件绑定 ──
        def _execute_bi(question, role, domain, bl):
            return asyncio.run(
                _bi_query_handler(question, role, domain, bl)
            )

        bi_btn.click(
            _execute_bi,
            [bi_question, bi_role, bi_domain, bi_bl],
            [bi_sql, bi_data, bi_summary, bi_error, bi_chart_json, bi_chart_fig, bi_insights],
        )

    # ═══════════════════════════════════════════════════════════════
    # Tab 3: 文档中心
    # ═══════════════════════════════════════════════════════════════

    with gr.Tab("📄 文档中心"):
        # ── 上半部分：上传 ──
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("### 上传新文档")
                with gr.Group(elem_classes="upload-zone"):
                    up_file = gr.File(
                        label="选择或拖拽文档到此处",
                        file_types=[".pdf", ".docx", ".doc", ".txt", ".md"],
                        file_count="single",
                    )
                with gr.Row():
                    up_type = gr.Dropdown(
                        choices=DOC_TYPES, value="spec_doc",
                        label="文档类型",
                    )
                    up_strategy = gr.Dropdown(
                        choices=CHUNK_STRATEGIES, value="fixed",
                        label="切片策略",
                    )
                with gr.Row():
                    up_model = gr.Textbox(
                        label="车型代码", value="",
                        placeholder="如: EV160",
                    )
                    up_category = gr.Textbox(
                        label="分类标签", value="通用",
                    )
                up_btn = gr.Button("上传并入库", variant="primary")

            with gr.Column(scale=5):
                up_result = gr.Markdown(
                    value="等待上传...\n\n上传文档后将自动完成：**解析 → 切片 → 嵌入 → 索引**",
                    label="上传状态",
                )

        gr.Markdown("---")

        # ── 下半部分：管理 ──
        gr.Markdown("### 已入库文档")
        with gr.Row():
            dm_keyword = gr.Textbox(
                label="搜索",
                placeholder="输入文档名关键词...",
                scale=4,
            )
            dm_search_btn = gr.Button("查询", variant="secondary", scale=1)
            dm_refresh_btn = gr.Button("刷新", variant="secondary", scale=1)

        dm_count = gr.Textbox(
            label="",
            interactive=False,
            value="点击「查询」或「刷新」加载文档列表",
            show_label=False,
            container=False,
        )
        dm_table = gr.Dataframe(
            headers=[
                "doc_id",
                "文档名",
                "类型",
                "分类",
                "车型",
                "切片数",
                "切片策略",
                "状态",
            ],
            label="",
            wrap=True,
            interactive=False,
        )

        gr.Markdown("#### 删除文档")
        with gr.Row():
            dd_id = gr.Textbox(
                label="doc_id",
                placeholder="输入要删除的 doc_id",
                scale=8,
            )
            dd_btn = gr.Button(
                "确认删除",
                variant="stop",
                scale=1,
                size="sm",
            )

        # ── 事件绑定 ──
        def _upload_wrapper(file, doc_type, category, model_code, strategy):
            if file is None:
                return "## ⚠️ 请先选择文件", "", []
            return asyncio.run(
                _upload_doc(file.name, doc_type, category, model_code, strategy)
            )

        up_btn.click(
            _upload_wrapper,
            [up_file, up_type, up_category, up_model, up_strategy],
            [up_result, dm_count, dm_table],
        )

        dm_search_btn.click(
            _refresh_docs, [dm_keyword], [dm_count, dm_table],
        )
        dm_refresh_btn.click(
            _refresh_docs, [dm_keyword], [dm_count, dm_table],
        )

        def _delete_wrapper(doc_id):
            return asyncio.run(_delete_doc(doc_id))

        dd_btn.click(
            _delete_wrapper, [dd_id], [up_result, dm_count, dm_table],
        )

    # ── Footer ──
    gr.HTML("""
    <div style="text-align:center; padding:16px 0 4px 0; color:#475569; font-size:11px;">
      rd-knowledge-svc &nbsp;|&nbsp; powered by DashScope qwen-max &middot; Milvus &middot; Neo4j
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        theme=theme,
        css=CSS,
    )
