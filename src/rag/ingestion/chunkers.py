# ============================================================
# 文档切片策略：fixed / semantic / parent_child
# 原文照搬 tiangong-agent
# ============================================================

from dataclasses import dataclass
from typing import ClassVar

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings

from src.core.config import get_settings
from src.rag.config import ChunkingConfig


class TruncatingEmbeddings(Embeddings):
    """包装 embedding model，自动截断超长文本，避免 DashScope 8192 token 限制"""

    MAX_CHARS: ClassVar[int] = 6000

    def __init__(self, base: Embeddings):
        self._base = base

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        truncated = [t[:self.MAX_CHARS] if len(t) > self.MAX_CHARS else t for t in texts]
        return self._base.embed_documents(truncated)

    def embed_query(self, text: str) -> list[float]:
        return self._base.embed_query(text[:self.MAX_CHARS] if len(text) > self.MAX_CHARS else text)


@dataclass
class Chunk:
    text: str
    metadata: dict


class FixedChunker:
    def __init__(self, config: ChunkingConfig):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " "],
        )

    def chunk(self, text: str, metadata: dict = None) -> list[Chunk]:
        docs = self.splitter.create_documents([text])
        return [
            Chunk(text=doc.page_content, metadata={**(metadata or {}), "chunk_index": i})
            for i, doc in enumerate(docs)
        ]


class SemanticChunkerWrapper:
    def __init__(self, embedding_model: DashScopeEmbeddings, breakpoint_threshold: float = 0.3):
        self.chunker = SemanticChunker(
            embeddings=TruncatingEmbeddings(embedding_model),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=breakpoint_threshold,
        )

    def chunk(self, text: str, metadata: dict = None) -> list[Chunk]:
        docs = self.chunker.create_documents([text])
        return [
            Chunk(text=doc.page_content, metadata={**(metadata or {}), "chunk_index": i})
            for i, doc in enumerate(docs)
        ]


class ParentChildChunker:
    def __init__(self, config: ChunkingConfig):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.parent_chunk_size, chunk_overlap=128,
            separators=["\n\n", "\n"],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " "],
        )

    def chunk(self, text: str, metadata: dict = None) -> list[Chunk]:
        parent_docs = self.parent_splitter.create_documents([text])
        chunks = []
        for pi, parent in enumerate(parent_docs):
            child_docs = self.child_splitter.create_documents([parent.page_content])
            for ci, child in enumerate(child_docs):
                chunks.append(Chunk(
                    text=child.page_content,
                    metadata={
                        **(metadata or {}),
                        "parent_index": pi,
                        "parent_text": parent.page_content,
                        "chunk_index": len(chunks),
                    },
                ))
        return chunks


def merge_short_chunks(
    chunks: list[Chunk],
    min_chars: int = 200,
    max_chars: int = 800,
) -> list[Chunk]:
    """合并相邻的短 chunk，减少检索碎片。

    规则：
    - 当前 chunk 或上一个 chunk 长度 < min_chars 视为碎片，尝试并入前一块；
    - 合并后总长 <= max_chars 才合并，否则保留独立；
    - parent_child 策略下两者 parent_index 必须相同（跨父块不合并）；
    - 合并后重新编号 chunk_index，元数据保留首块（含 parent_text）。
    """
    if not chunks:
        return []

    result: list[Chunk] = []
    for cur in chunks:
        if not result:
            result.append(Chunk(text=cur.text, metadata=dict(cur.metadata)))
            continue
        last = result[-1]
        pa = last.metadata.get("parent_index")
        pb = cur.metadata.get("parent_index")
        same_parent = pa is None or pb is None or pa == pb
        if (
            (len(cur.text) < min_chars or len(last.text) < min_chars)
            and len(last.text) + len(cur.text) <= max_chars
            and same_parent
        ):
            result[-1] = Chunk(
                text=last.text + "\n\n" + cur.text,
                metadata=dict(last.metadata),
            )
        else:
            result.append(Chunk(text=cur.text, metadata=dict(cur.metadata)))

    for i, c in enumerate(result):
        c.metadata["chunk_index"] = i
    return result


def get_chunker(config: ChunkingConfig, embedding_model: DashScopeEmbeddings = None):
    if config.strategy == "semantic":
        if embedding_model is None:
            embedding_model = DashScopeEmbeddings(
                model=get_settings().EMBEDDING_MODEL,
                dashscope_api_key=get_settings().DASHSCOPE_API_KEY,
            )
        return SemanticChunkerWrapper(embedding_model)
    elif config.strategy == "parent_child":
        return ParentChildChunker(config)
    else:
        return FixedChunker(config)
