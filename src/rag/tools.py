# ============================================================
# RAG LangChain 工具封装
# ============================================================

from langchain_core.tools import tool

from src.rag.engine import RAGEngine

KNOWLEDGE_BASES = {
    "repair_manual": {"collection": "alm_docs", "description": "维修手册、诊断流程、拆装规范"},
    "spec_doc": {"collection": "alm_docs", "description": "设计规范、供应商规格书、接口文档"},
    "tsb": {"collection": "alm_docs", "description": "技术通报 TSB、服务活动公告"},
    "issue_cases": {"collection": "alm_cases", "description": "历史问题闭环案例（关单问题向量化）"},
}


def create_rag_tools(engine: RAGEngine) -> list:
    """创建 RAG 工具（供项目一 Agent 调用）"""

    @tool
    async def search_knowledge(question: str, knowledge_base: str = "repair_manual") -> str:
        """检索知识库。knowledge_base: repair_manual/spec_doc/tsb/issue_cases"""
        kb = KNOWLEDGE_BASES.get(knowledge_base)
        if not kb:
            return f"未知知识库: {knowledge_base}。可选: {list(KNOWLEDGE_BASES.keys())}"
        filters = {"doc_type": knowledge_base}
        return await engine.query(question, filters)

    @tool
    async def search_knowledge_with_sources(question: str) -> str:
        """检索所有知识库并返回带来源的答案"""
        return await engine.query(question)

    @tool
    async def list_knowledge_bases() -> str:
        """列出所有可用的知识库"""
        lines = [f"- {k}: {v['description']}" for k, v in KNOWLEDGE_BASES.items()]
        return "可用知识库：\n" + "\n".join(lines)

    return [search_knowledge, search_knowledge_with_sources, list_knowledge_bases]
