# ============================================================
# 入库管线：parse → chunk → embed → index
#
# ★ sparse_embedding 由 Milvus 2.6 内置 BM25 Function 自动生成
#   与天宫医疗方案一致：无需手动计算 BM25 向量
# 幂等：doc_id = md5(doc_name)[:16]，重复上传先删后插
# 进度：落 doc_ingest_jobs 表
# ★ 图片：MinerU 提取 → MinIO → markdown 引用替换为 MinIO URL
# ============================================================

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from loguru import logger
from pymilvus import DataType, Function, FunctionType, MilvusClient

from src.core.config import get_settings
from src.infra.minio_client import ensure_bucket_exists, upload_file
from src.rag.config import ChunkingConfig
from src.rag.ingestion.chunkers import get_chunker, merge_short_chunks
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
        self._ensure_collection()

    def _collection_has_bm25(self) -> bool:
        """检测已有 collection 是否含 BM25 Function。

        幂等安全：describe_collection 失败或无法确认时，保守返回 True（不删除），
        避免把已有文档数据误删。"""
        try:
            desc = self.milvus.describe_collection(COLLECTION_NAME)
        except Exception as e:
            logger.warning(f"describe_collection 失败，保守跳过重建: {e}")
            return True

        if not isinstance(desc, dict):
            return True

        # 兼容 describe_collection 返回结构：functions 可能在顶层或嵌套在 schema 下
        candidates = [desc]
        schema = desc.get("schema")
        if isinstance(schema, dict):
            candidates.append(schema)

        for cfg in candidates:
            for f in cfg.get("functions") or []:
                if not isinstance(f, dict):
                    continue
                name = str(f.get("name", "")).lower()
                ftype = f.get("type")
                if name == "bm25" or ftype in (FunctionType.BM25.value, "BM25"):
                    return True
        return False

    def _ensure_collection(self) -> None:
        """确保 alm_docs collection 存在，含 BM25 Function（与天宫医疗一致）。
        幂等：已存在且含 BM25 Function 则跳过；
        仅旧版 collection（无 BM25 Function）才删除重建，避免每次实例化都清空索引。"""
        if self.milvus.has_collection(COLLECTION_NAME):
            if self._collection_has_bm25():
                logger.info(
                    f"collection '{COLLECTION_NAME}' 已存在且含 BM25 Function，跳过重建"
                )
                return
            self.milvus.drop_collection(COLLECTION_NAME)
            logger.info(
                f"旧版 collection '{COLLECTION_NAME}'（无 BM25 Function），删除重建"
            )

        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("id", DataType.VARCHAR, max_length=256, is_primary=True)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
        schema.add_field("doc_name", DataType.VARCHAR, max_length=256)
        schema.add_field("doc_type", DataType.VARCHAR, max_length=50)
        schema.add_field("category", DataType.VARCHAR, max_length=100)
        schema.add_field("business_line", DataType.VARCHAR, max_length=50)
        schema.add_field("model_code", DataType.VARCHAR, max_length=50)
        schema.add_field("page_number", DataType.INT64)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("parent_text", DataType.VARCHAR, max_length=65535)
        schema.add_field("text", DataType.VARCHAR, max_length=65535, enable_analyzer=True)
        schema.add_field("image_urls", DataType.VARCHAR, max_length=65535)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)

        # ★ Milvus 2.6 内置 BM25 Function：自动从 text 生成 sparse_embedding
        bm25_fn = Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_embedding"],
        )
        schema.add_function(bm25_fn)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            metric_type="COSINE",
            index_type="IVF_FLAT",
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name="sparse_embedding",
            metric_type="BM25",  # ★ 内置 Function 输出字段用 BM25 度量
            index_type="SPARSE_INVERTED_INDEX",
        )

        self.milvus.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"Milvus collection '{COLLECTION_NAME}' 创建成功（BM25 Function）")

    async def ingest(self, file_path: str, meta: DocMetadata) -> str:
        """返回 doc_id"""
        doc_id = hashlib.md5(meta.doc_name.encode()).hexdigest()[:16]

        # 1. parse — MinerU 返回 (md_text, pages, images_dict)
        md_text, pages, images = await self.parser.parse(file_path)
        if not md_text or len(md_text.strip()) < 10:
            logger.error(f"文档解析后无内容: {meta.doc_name}")
            return doc_id

        # ★ 上传图片到 MinIO，替换 markdown 引用为 MinIO URL
        if images:
            md_text = await self._upload_images_and_replace_refs(
                md_text, images, doc_id, meta.doc_name,
            )

        # 2. chunk — semantic 策略需要 embedding model
        from src.rag.ingestion.embedders import DenseEmbedder
        dense_embedder = DenseEmbedder()
        chunker = get_chunker(self.chunking_config, embedding_model=dense_embedder.model)
        chunks = chunker.chunk(md_text, metadata={
            "doc_name": meta.doc_name,
            "doc_type": meta.doc_type,
        })

        # ★ 短 chunk 合并：碎片并入相邻块，减少检索稀释
        chunks = merge_short_chunks(
            chunks,
            min_chars=self.chunking_config.merge_min_chars,
            max_chars=self.chunking_config.merge_max_chars,
        )

        if not chunks:
            logger.warning(f"切片后无内容: {meta.doc_name}")
            return doc_id

        texts = [c.text for c in chunks]

        # 3. embed (only dense — sparse 由 Milvus BM25 Function 自动生成)
        dense_vecs = await dense_embedder.embed(texts)

        # 幂等：解析/切片/嵌入全部成功后才删旧数据，失败时保留旧版本文档
        self.milvus.delete(
            collection_name=COLLECTION_NAME,
            filter=f'doc_id == "{doc_id}"',
        )

        # 4. index (batch=50)
        batch_size = 50
        all_data = []

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_dense = dense_vecs[i:i + batch_size]

            for j, chunk in enumerate(batch_chunks):
                chunk_idx = i + j
                image_urls = self._extract_image_urls(chunk.text)
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
                    "image_urls": json.dumps(image_urls, ensure_ascii=False)[:65000],
                    "embedding": batch_dense[j] if j < len(batch_dense) else [],
                }
                all_data.append(record)

        if all_data:
            self.milvus.insert(collection_name=COLLECTION_NAME, data=all_data)
            logger.info(f"入库完成: {meta.doc_name} doc_id={doc_id} chunks={len(all_data)}")

        return doc_id

    async def _upload_images_and_replace_refs(
        self, md_text: str, images: dict[str, bytes], doc_id: str, doc_name: str,
    ) -> str:
        """上传图片到 MinIO；开启 VL 时用 qwen-vl-max 生成图片描述，
        替换 markdown 中 `![](images/xxx.jpg)` 为 `![描述](minio_url)`。"""
        ensure_bucket_exists()
        for img_name, img_bytes in images.items():
            # 确定 content_type
            ext = img_name.rsplit(".", 1)[-1].lower() if "." in img_name else "jpg"
            content_type_map = {
                "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "gif": "image/gif",
                "webp": "image/webp", "bmp": "image/bmp",
                "svg": "image/svg+xml",
            }
            content_type = content_type_map.get(ext, "image/jpeg")

            object_name = f"images/{doc_id}/{img_name}"
            try:
                upload_file(object_name, img_bytes, content_type=content_type)
            except Exception as e:
                logger.error(f"图片上传 MinIO 失败 {object_name}: {e}")
                continue

            # 构造 MinIO URL（![]() 引用需要直接访问的 URL）
            scheme = "https" if settings.MINIO_SECURE else "http"
            minio_url = f"{scheme}://{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET}/{object_name}"

            # ★ VL 摘要：描述写入 alt，检索可命中图片内容（fail-open：失败则 alt 为空）
            #   附带图片所在段落上下文，让描述带上车型/部件名等可检索专业词
            summary = ""
            if settings.IMAGE_SUMMARIZE_ENABLED:
                from src.rag.ingestion.image_summarizer import summarize_image
                context = self._extract_image_context(md_text, f"images/{img_name}")
                summary = await summarize_image(
                    img_bytes, content_type, settings.VL_MODEL, context=context,
                )
            alt = f"[{summary}]" if summary else "[]"

            # 替换 markdown 中的图片引用：![]() → ![summary](minio_url)
            for ref in (f"images/{img_name}", img_name):
                md_text = md_text.replace(f"![]({ref})", f"!{alt}({ref})")
                md_text = md_text.replace(f"]({ref})", f"]({minio_url})")

        logger.info(f"图片上传 MinIO 完成: {doc_name} ({len(images)} 张)")
        return md_text

    @staticmethod
    def _extract_image_context(md_text: str, img_ref: str, window: int = 200) -> str:
        """取图片引用前的一段 markdown 作为 VL 摘要的辅助上下文。

        窗口 window 字符内，向前截到距离图片最近的分隔点：优先上一个空行（段落边界），
        其次上一个图片引用的右括号之后——避免把邻居图的引用字符串混进上下文。"""
        idx = md_text.find(f"![]({img_ref})")
        if idx == -1:
            return ""
        prefix = md_text[max(0, idx - window):idx]

        candidates: list[int] = []
        blank = prefix.rfind("\n\n")
        if blank >= 0:
            candidates.append(blank + 2)  # 空行之后
        prev_img = prefix.rfind("![](")
        if prev_img >= 0:
            close = prefix.find(")", prev_img)
            candidates.append(close + 1 if close >= 0 else prev_img)  # 上一个图片引用之后
        if candidates:
            prefix = prefix[max(candidates):]
        return prefix.strip()

    @staticmethod
    def _extract_image_urls(text: str) -> list[str]:
        """从 markdown 文本中提取所有图片 URL"""
        pattern = r"!\[.*?\]\((https?://[^\s)]+)\)"
        return re.findall(pattern, text)


def get_ingestion_pipeline(milvus: MilvusClient) -> IngestionPipeline:
    """工厂函数"""
    return IngestionPipeline(milvus_client=milvus)
