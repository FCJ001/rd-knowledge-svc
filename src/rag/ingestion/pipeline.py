# ============================================================
# 入库管线：parse → chunk → embed → index
#
# ★ 补齐医疗版空文件
# 幂等：doc_id = md5(doc_name)[:16]，重复上传先删后插
# 进度：落 doc_ingest_jobs 表
# ============================================================

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from loguru import logger
from pymilvus import DataType, MilvusClient

from src.core.config import get_settings
from src.rag.config import ChunkingConfig
from src.rag.ingestion.chunkers import get_chunker
from src.rag.ingestion.embedders import EmbeddingGenerator
from src.rag.ingestion.parsers import DocumentParser

settings = get_settings()

COLLECTION_NAME = "alm_docs"
EMBEDDING_DIM = 1024


@dataclass
class DocMetadata:
    doc_name: str
    doc_type: str          # repair_manual / spec_doc / tsb / issue_case
    category: str = ""
    business_line: str = ""
    model_code: str = ""   # ★ 汽车域刚需：按车型过滤


class IngestionPipeline:
    def __init__(
        self,
        milvus_client: MilvusClient,
        chunking_config: ChunkingConfig | None = None,
        parser: str = "mineru",
    ):
        self.milvus = milvus_client
        self.chunking_config = chunking_config or ChunkingConfig()
        self.parser = DocumentParser(parser=parser)
        self.embedder = EmbeddingGenerator()
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """确保 alm_docs collection 存在（含 dense + sparse 字段）"""
        if self.milvus.has_collection(COLLECTION_NAME):
            return

        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("id", DataType.VARCHAR, max_length=256, is_primary=True)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
        schema.add_field("doc_name", DataType.VARCHAR, max_length=256)
        schema.add_field("doc_type", DataType.VARCHAR, max_length=50)
        schema.add_field("category", DataType.VARCHAR, max_length=100)
        schema.add_field("business_line", DataType.VARCHAR, max_length=50)
        schema.add_field("model_code", DataType.VARCHAR, max_length=50)  # ★
        schema.add_field("page_number", DataType.INT64)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("parent_text", DataType.VARCHAR, max_length=65535)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)  # ★ Hybrid

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            metric_type="COSINE",
            index_type="IVF_FLAT",
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name="sparse_embedding",
            metric_type="IP",  # ★ 手动生成 sparse 向量用 IP，BM25 仅限 Milvus 内置 Function
            index_type="SPARSE_INVERTED_INDEX",
        )

        self.milvus.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"Milvus collection '{COLLECTION_NAME}' 创建成功（含 sparse 字段）")

    async def ingest(self, file_path: str, meta: DocMetadata) -> str:
        """返回 doc_id"""
        doc_id = hashlib.md5(meta.doc_name.encode()).hexdigest()[:16]

        # 幂等：先删后插
        self.milvus.delete(
            collection_name=COLLECTION_NAME,
            filter=f'doc_id == "{doc_id}"',
        )

        # 1. parse
        md_text, pages = await self.parser.parse(file_path)
        if not md_text or len(md_text.strip()) < 10:
            logger.error(f"文档解析后无内容: {meta.doc_name}")
            return doc_id

        # 2. chunk
        chunker = get_chunker(self.chunking_config)
        chunks = chunker.chunk(md_text, metadata={
            "doc_name": meta.doc_name,
            "doc_type": meta.doc_type,
        })

        if not chunks:
            logger.warning(f"切片后无内容: {meta.doc_name}")
            return doc_id

        texts = [c.text for c in chunks]

        # 3. embed (dense + sparse)
        dense_vecs, sparse_vecs = await self.embedder.generate(texts)

        # 4. index (batch=50)
        batch_size = 50
        all_data = []

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_dense = dense_vecs[i:i + batch_size]
            batch_sparse = sparse_vecs[i:i + batch_size]

            for j, chunk in enumerate(batch_chunks):
                chunk_idx = i + j
                record = {
                    "id": f"{doc_id}_{chunk_idx}",
                    "doc_id": doc_id,
                    "doc_name": meta.doc_name,
                    "doc_type": meta.doc_type,
                    "category": meta.category,
                    "business_line": meta.business_line,
                    "model_code": meta.model_code,
                    "page_number": pages[chunk_idx] if chunk_idx < len(pages) else 0,
                    "chunk_index": chunk_idx,
                    "parent_text": chunk.metadata.get("parent_text", "")[:65000],
                    "text": chunk.text[:65000],
                    "embedding": batch_dense[j] if j < len(batch_dense) else [],
                    "sparse_embedding": batch_sparse[j] if j < len(batch_sparse) else {},
                }
                all_data.append(record)

        if all_data:
            self.milvus.insert(collection_name=COLLECTION_NAME, data=all_data)
            logger.info(f"入库完成: {meta.doc_name} doc_id={doc_id} chunks={len(all_data)}")

        return doc_id


def get_ingestion_pipeline(milvus: MilvusClient) -> IngestionPipeline:
    """工厂函数"""
    return IngestionPipeline(milvus_client=milvus)
