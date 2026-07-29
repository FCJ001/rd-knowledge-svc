# ============================================================
# 知识库数据模型 — 5 张自有表
#
# knowledge_docs         文档元数据
# doc_ingest_jobs        入库任务进度
# knowledge_feedback     用户反馈
# knowledge_notifications 知识更新通知
# rag_experiments        RAG 消融实验结果
# ============================================================

from datetime import datetime

from sqlalchemy import Float, Integer, String, Text, DateTime, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column

from src.core.base_model import Base, BaseModel


class KnowledgeDoc(BaseModel):
    """文档元数据"""
    __tablename__ = "knowledge_docs"

    doc_id: Mapped[str] = mapped_column(String(16), unique=True, index=True, comment="md5(doc_name)[:16]")
    doc_name: Mapped[str] = mapped_column(String(255), comment="文档名称")
    doc_type: Mapped[str] = mapped_column(String(50), comment="文档类型: repair_manual/spec_doc/tsb/issue_case")
    category: Mapped[str | None] = mapped_column(String(100), comment="分类标签")
    business_line: Mapped[str | None] = mapped_column(String(50), comment="业务线")
    model_code: Mapped[str | None] = mapped_column(String(50), comment="车型代码")
    minio_key: Mapped[str | None] = mapped_column(String(500), comment="MinIO 原始文件 key")
    page_count: Mapped[int | None] = mapped_column(Integer, comment="总页数")
    chunk_count: Mapped[int | None] = mapped_column(Integer, comment="切片数量")
    chunk_strategy: Mapped[str | None] = mapped_column(String(20), comment="切片策略: fixed/semantic/parent_child")
    version: Mapped[int] = mapped_column(Integer, default=1, comment="版本号")
    status: Mapped[str] = mapped_column(String(20), default="indexed", comment="状态: ingesting/indexed/deleted")


class DocIngestJob(BaseModel):
    """入库任务进度"""
    __tablename__ = "doc_ingest_jobs"

    doc_id: Mapped[str] = mapped_column(String(16), index=True, comment="关联 knowledge_docs.doc_id")
    stage: Mapped[str] = mapped_column(String(20), comment="阶段: parse/chunk/embed/index")
    progress: Mapped[int] = mapped_column(Integer, default=0, comment="进度百分比 0-100")
    parser: Mapped[str] = mapped_column(String(20), default="mineru", comment="解析器: mineru/llamaindex")
    retry_count: Mapped[int] = mapped_column(Integer, default=0, comment="重试次数")
    error_msg: Mapped[str | None] = mapped_column(Text, comment="错误信息")


class KnowledgeFeedback(BaseModel):
    """用户反馈"""
    __tablename__ = "knowledge_feedback"

    user_id: Mapped[str] = mapped_column(String(50), index=True, comment="用户标识")
    question: Mapped[str] = mapped_column(Text, comment="用户问题")
    answer_preview: Mapped[str | None] = mapped_column(Text, comment="答案摘要")
    rating: Mapped[int] = mapped_column(Integer, comment="评分: 1 赞 / -1 踩 / 0 中性")
    comment: Mapped[str | None] = mapped_column(Text, comment="评语")
    intent: Mapped[str | None] = mapped_column(String(50), comment="意图分类")
    channels: Mapped[str | None] = mapped_column(String(100), comment="使用的检索通道")
    trace_id: Mapped[str | None] = mapped_column(String(20), comment="关联请求")


class KnowledgeNotification(Base):
    """知识更新通知"""
    __tablename__ = "knowledge_notifications"

    __abstract__ = False
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_name: Mapped[str] = mapped_column(String(255), comment="文档名称")
    doc_type: Mapped[str] = mapped_column(String(50), comment="文档类型")
    category: Mapped[str | None] = mapped_column(String(100), comment="分类")
    action: Mapped[str] = mapped_column(String(20), comment="操作: upload/update/delete")
    is_read: Mapped[bool] = mapped_column(Boolean, default=False, comment="是否已读")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class RagExperiment(BaseModel):
    """RAG 消融实验结果"""
    __tablename__ = "rag_experiments"

    exp_name: Mapped[str] = mapped_column(String(100), comment="实验名称")
    chunk_strategy: Mapped[str] = mapped_column(String(20), comment="切片策略")
    chunk_size: Mapped[int] = mapped_column(Integer, comment="切片大小")
    top_k: Mapped[int] = mapped_column(Integer, comment="检索 Top-K")
    rerank_top_k: Mapped[int] = mapped_column(Integer, comment="重排 Top-K")
    use_hyde: Mapped[bool] = mapped_column(Boolean, default=False, comment="启用 HyDE")
    use_hybrid: Mapped[bool] = mapped_column(Boolean, default=False, comment="启用混合检索")
    context_relevance: Mapped[float | None] = mapped_column(Float, comment="上下文相关性")
    groundedness: Mapped[float | None] = mapped_column(Float, comment="有据性")
    answer_relevance: Mapped[float | None] = mapped_column(Float, comment="答案相关性")
    safety_score: Mapped[float | None] = mapped_column(Float, comment="安全性评分")
    p95_latency_ms: Mapped[float | None] = mapped_column(Float, comment="P95 延迟(ms)")
