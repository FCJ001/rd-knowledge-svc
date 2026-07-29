# ============================================================
# Gradio 界面 — 知识库 + NL2SQL/ChatBI 功能测试
#
# 运行: python test/test_rag_gradio.py
# 浏览器打开: http://localhost:7861
# ============================================================

import asyncio
import os
import sys
import json
import tempfile
from io import BytesIO

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from loguru import logger
from PIL import Image

from src.core.config import get_settings
from src.infra.db import AsyncSessionLocal
from src.infra.db_readonly import ReadOnlySessionLocal
from src.infra.milvus_client import get_milvus_client
from src.infra.neo4j_client import get_neo4j_driver
from src.infra.minio_client import download_file

load_dotenv()
settings = get_settings()


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
# Tab 1: 知识检索
# ══════════════════════════════════════════════════════════════════════

async def _knowledge_search(question: str, channels: list, role: str):
    if not question.strip():
        return "请输入问题", "", []

    from src.knowledge.fusion import multi_channel_search
    from src.knowledge.doc_rag import extract_image_urls, search_docs_raw

    llm = _get_llm()
    embedding_model = _get_embedding_model()
    milvus_client = get_milvus_client()
    neo4j_driver = get_neo4j_driver()

    selected_channels = [c for c in ["doc_rag", "graph_rag", "nl2sql"] if c in channels]

    async with AsyncSessionLocal() as db:
        answer = await multi_channel_search(
            question=question,
            llm=llm,
            embedding_model=embedding_model,
            milvus_client=milvus_client,
            neo4j_driver=neo4j_driver,
            db_session=db if "nl2sql" in selected_channels else None,
            channels=selected_channels,
            role=role,
        )

    # ★ 单独查文档获取图片 URL，下载到本地转为 PIL Image
    image_list = []
    if "doc_rag" in selected_channels:
        try:
            doc_hits = await search_docs_raw(
                question, embedding_model, milvus_client, top_k=10, rerank_top_k=5,
            )
            image_urls = extract_image_urls(doc_hits)
            for url in image_urls:
                try:
                    # URL 格式: http://localhost:9000/knowledge-docs/images/{doc_id}/{filename}
                    # 从 URL 提取 object_name
                    parts = url.split(f"{settings.MINIO_BUCKET}/", 1)
                    if len(parts) == 2:
                        object_name = parts[1]
                        img_bytes = download_file(object_name)
                        pil_img = Image.open(BytesIO(img_bytes))
                        image_list.append((pil_img, os.path.basename(object_name)))
                except Exception as e:
                    logger.warning(f"图片加载失败 {url}: {e}")
        except Exception as e:
            logger.warning(f"图片提取失败: {e}")

    debug = f"通道: {', '.join(selected_channels)}\n角色: {role}"
    return answer, debug, image_list


def knowledge_search_handler(question, doc_rag, graph_rag, nl2sql, role):
    channels = []
    if doc_rag:
        channels.append("doc_rag")
    if graph_rag:
        channels.append("graph_rag")
    if nl2sql:
        channels.append("nl2sql")
    if not channels:
        return "请至少选择一个检索通道", "", []
    return asyncio.run(_knowledge_search(question, channels, role))


# ══════════════════════════════════════════════════════════════════════
# Tab 2: NL2SQL / ChatBI
# ══════════════════════════════════════════════════════════════════════

async def _bi_query(question: str, role: str, owner_domain_id, business_line):
    if not question.strip():
        return "", "", "", "请输入查询问题", "", None

    from src.nl2sql.engine import run_query
    from src.nl2sql.chart_advisor import recommend_chart, render_chart

    llm = _get_llm()
    owner_domain_id = int(owner_domain_id) if owner_domain_id else None
    business_line = business_line.strip() if business_line.strip() else None

    async with ReadOnlySessionLocal() as db:
        result = await run_query(
            question=question, llm=llm, db=db,
            role=role,
            owner_domain_id=owner_domain_id,
            business_line=business_line,
        )

    sql = result.sql or ""
    data_str = json.dumps(result.data, ensure_ascii=False, indent=2, default=str) if result.data else ""
    summary = result.summary or ""
    error = result.error or ""
    chart_str = ""
    chart_fig = None

    if result.success and result.data:
        try:
            config = await recommend_chart(question, result.data, result.columns, llm)
            chart_fig = render_chart(result.data, config)
            chart_str = json.dumps({
                "chart_type": config.get("chart_type", "table"),
                "title": config.get("title", ""),
                "description": config.get("description", ""),
            }, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            chart_str = f"图表生成失败: {e}"

    return sql, data_str[:5000], summary, error, chart_str, chart_fig


def bi_query_handler(question, role, owner_domain_id, business_line):
    return asyncio.run(_bi_query(question, role, owner_domain_id, business_line))


# ══════════════════════════════════════════════════════════════════════
# Tab 3: 文档上传
# ══════════════════════════════════════════════════════════════════════

async def _upload_doc(file_path, doc_type, category, business_line, model_code, chunk_strategy):
    if not file_path:
        return "请上传文件"

    from src.knowledge.doc_ingestion import ingest_and_index

    doc_name = os.path.basename(file_path)

    async with AsyncSessionLocal() as db:
        try:
            doc_id = await ingest_and_index(
                file_path=file_path, doc_name=doc_name, doc_type=doc_type,
                category=category, business_line=business_line,
                model_code=model_code, chunk_strategy=chunk_strategy,
                parser="mineru", db=db,
            )
            await db.commit()
            return (
                f"上传成功\n"
                f"文件名: {doc_name}\n"
                f"doc_id: {doc_id}\n"
                f"类型: {doc_type}\n"
                f"切片策略: {chunk_strategy}\n"
                f"车型: {model_code or '通用'}"
            )
        except Exception as e:
            return f"上传失败: {e}"


def upload_handler(file, doc_type, category, business_line, model_code, chunk_strategy):
    if file is None:
        return "请上传文件"
    return asyncio.run(_upload_doc(
        file.name, doc_type, category, business_line, model_code, chunk_strategy,
    ))


# ══════════════════════════════════════════════════════════════════════
# Tab 4: 文档管理
# ══════════════════════════════════════════════════════════════════════

async def _list_docs(keyword: str):
    """返回 (count_msg, rows) — rows 是 list[list] 用于更新 Dataframe"""
    from sqlalchemy import select
    from src.knowledge.model import KnowledgeDoc

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
                    d.doc_id, d.doc_name, d.doc_type, d.category or "-",
                    d.model_code or "-", d.chunk_count or 0,
                    d.chunk_strategy or "-", d.status,
                ]
                for d in docs
            ]
            return f"共 {len(rows)} 条记录", rows
    except Exception as e:
        return f"查询失败: {e}", []


def list_docs_handler(keyword):
    return asyncio.run(_list_docs(keyword))


async def _delete_doc(doc_id: str):
    if not doc_id.strip():
        return "请输入 doc_id"

    from src.knowledge.doc_ingestion import delete_doc as delete_doc_service

    try:
        async with AsyncSessionLocal() as db:
            await delete_doc_service(doc_id.strip(), db)
            await db.commit()
        return f"已删除: {doc_id}"
    except Exception as e:
        return f"删除失败: {e}"


def delete_doc_handler(doc_id):
    return asyncio.run(_delete_doc(doc_id))


# ══════════════════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════════════════

DOC_TYPES = ["repair_manual", "spec_doc", "tsb", "issue_case"]
ROLES = ["engineer", "business", "aftersales", "admin"]
CHUNK_STRATEGIES = ["fixed", "semantic", "parent_child"]

with gr.Blocks(title="研发知识库 — 功能测试") as demo:
    gr.Markdown("""
    # 研发知识库 & 运营问答服务
    **rd-knowledge-svc** — 文档 RAG + 知识图谱 + NL2SQL/ChatBI
    """)

    # ── Tab 1: 知识检索 ──────────────────────────────────────────────

    with gr.Tab("知识检索"):
        gr.Markdown("### 多通道融合检索")
        with gr.Row():
            with gr.Column(scale=2):
                ks_question = gr.Textbox(
                    label="问题",
                    placeholder="例：汉EV 2024款的电池管理系统有哪些安全保护机制？",
                    lines=2,
                )
                with gr.Row():
                    ks_doc = gr.Checkbox(label="文档检索 (Doc RAG)", value=True)
                    ks_graph = gr.Checkbox(label="知识图谱 (Graph RAG)", value=True)
                    ks_nl2sql = gr.Checkbox(label="运营数据 (NL2SQL)", value=False)
                ks_role = gr.Dropdown(choices=ROLES, value="engineer", label="角色")
                ks_btn = gr.Button("检索", variant="primary")
            with gr.Column(scale=3):
                ks_answer = gr.Textbox(label="回答", lines=12)
                ks_debug = gr.Textbox(label="调试信息", lines=3)
                ks_images = gr.Gallery(label="相关图片", columns=3, height=300, show_label=True)
        ks_btn.click(
            knowledge_search_handler,
            [ks_question, ks_doc, ks_graph, ks_nl2sql, ks_role],
            [ks_answer, ks_debug, ks_images],
        )

    # ── Tab 2: NL2SQL / ChatBI ───────────────────────────────────────

    with gr.Tab("NL2SQL / ChatBI"):
        gr.Markdown("### 自然语言数据查询 + 图表推荐")
        with gr.Row():
            with gr.Column(scale=2):
                bi_question = gr.Textbox(
                    label="自然语言查询",
                    placeholder="例：电池系统域有哪些critical级别的问题？",
                    lines=2,
                )
                bi_role = gr.Dropdown(choices=ROLES, value="engineer", label="角色")
                with gr.Row():
                    bi_domain = gr.Textbox(label="owner_domain_id（工程师角色用）", value="1")
                    bi_bl = gr.Textbox(label="business_line（业务角色用）", value="")
                bi_btn = gr.Button("执行查询", variant="primary")
            with gr.Column(scale=3):
                bi_sql = gr.Textbox(label="生成的 SQL", lines=4)
                bi_data = gr.Textbox(label="查询结果", lines=6)
                bi_summary = gr.Textbox(label="摘要", lines=3)
                bi_error = gr.Textbox(label="错误", lines=2)
                bi_chart = gr.Textbox(label="图表配置 (JSON)", lines=4)
                bi_chart_fig = gr.Plot(label="可视化图表")
        bi_btn.click(
            bi_query_handler,
            [bi_question, bi_role, bi_domain, bi_bl],
            [bi_sql, bi_data, bi_summary, bi_error, bi_chart, bi_chart_fig],
        )

    # ── Tab 3: 文档上传 ──────────────────────────────────────────────

    with gr.Tab("文档上传"):
        gr.Markdown("### 上传文档并自动入库（解析-切片-嵌入-索引）")
        with gr.Row():
            with gr.Column(scale=2):
                up_file = gr.File(
                    label="选择文档（PDF/Word/TXT/MD）",
                    file_types=[".pdf", ".docx", ".doc", ".txt", ".md"],
                )
                up_type = gr.Dropdown(choices=DOC_TYPES, value="spec_doc", label="文档类型")
                up_category = gr.Textbox(label="分类标签", value="通用")
                up_bl = gr.Textbox(label="业务线", value="")
                up_model = gr.Textbox(label="车型代码（如 汉EV_2024）", value="")
                up_strategy = gr.Dropdown(choices=CHUNK_STRATEGIES, value="fixed", label="切片策略")
                up_btn = gr.Button("上传并入库", variant="primary")
            with gr.Column(scale=3):
                up_result = gr.Textbox(label="入库结果", lines=6)
        up_btn.click(
            upload_handler,
            [up_file, up_type, up_category, up_bl, up_model, up_strategy],
            [up_result],
        )

    # ── Tab 4: 文档管理 ──────────────────────────────────────────────

    with gr.Tab("文档管理"):
        gr.Markdown("### 已入库文档管理")
        with gr.Row():
            dm_keyword = gr.Textbox(label="搜索关键词（留空查全部）", value="")
            dm_btn = gr.Button("查询", variant="primary")
        dm_count = gr.Textbox(label="查询结果", interactive=False, value="点击查询按钮加载文档列表")
        dm_table = gr.Dataframe(
            headers=["doc_id", "文档名", "类型", "分类", "车型", "切片数", "切片策略", "状态"],
            label="文档列表",
            wrap=True,
        )
        dm_btn.click(list_docs_handler, [dm_keyword], [dm_count, dm_table])

        gr.Markdown("---\n### 删除文档")
        with gr.Row():
            dd_id = gr.Textbox(label="doc_id", placeholder="输入要删除的 doc_id")
            dd_btn = gr.Button("删除", variant="stop")
        dd_result = gr.Textbox(label="删除结果", interactive=False)
        dd_btn.click(delete_doc_handler, [dd_id], [dd_result])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
