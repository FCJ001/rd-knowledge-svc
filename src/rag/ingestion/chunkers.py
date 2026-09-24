# ============================================================
# 文档切片策略：fixed / semantic / parent_child
# 原文照搬 tiangong-agent
# ============================================================

import re
from dataclasses import dataclass
from typing import ClassVar

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

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
            for child in child_docs:
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
    - **受保护块（atomic）不参与合并**——表格/公式整体成块是它的存在意义，
      合并进去等于又破坏了边界；
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
        if cur.metadata.get("atomic") or last.metadata.get("atomic"):
            result.append(Chunk(text=cur.text, metadata=dict(cur.metadata)))
            continue
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


# ── 受保护块：表格 / 公式整体成块，不参与递归切分 ────────────────────────
#
# 为什么需要：通用切分器不认识表格边界。一张复杂表格在 markdown 里是
# HTML + `> 表格摘要` + `> VL 校读` + `![表格原图](...)` 一整段，很容易
# 超过 512 字，会被从中间腰斩——列含义、表头与行的对应关系全部丢失。
#
# 判定为「表格/公式块 + 紧随其后的注解行」：注解行指紧跟着的引用块（> 开头）、
# 图片引用（![ 开头）与不一致告警（⚠️ 开头），它们语义上属于同一块，
# 必须跟着一起走。
_TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
_FORMULA_RE = re.compile(r"\$\$[\s\S]*?\$\$")
_ANNOTATION_PREFIXES = (">", "![", "⚠️", "**⚠️")


def _is_annotation(line: str) -> bool:
    """注解行：语义上属于前一个表格/公式块，必须跟着它走。"""
    return line.lstrip().startswith(_ANNOTATION_PREFIXES)


def split_protected_blocks(md_text: str) -> list[tuple[str, bool]]:
    """把 markdown 切分为 [(片段, 是否受保护块)]，保持原顺序。

    受保护块 = 表格 HTML 或 $$公式$$，连同其后紧跟的注解行（摘要/校读/原图）。
    """
    segments: list[tuple[str, bool]] = []
    lines = md_text.split("\n")
    i = 0
    buf: list[str] = []

    def flush():
        if buf:
            text = "\n".join(buf).strip()
            if text:
                segments.append((text, False))
            buf.clear()

    while i < len(lines):
        line = lines[i]
        m_table = _TABLE_RE.search(line)
        m_formula = None if m_table else _FORMULA_RE.search(line)
        if m_table or m_formula:
            flush()
            block_lines = [line]
            # 表格/公式可能跨多行：向后拼到闭合标记
            closer = "</table>" if m_table else "$$"
            while closer not in block_lines[-1].lower() and i + 1 < len(lines):
                i += 1
                block_lines.append(lines[i])
            # ★ 注解行要收进块里，而不是跳过——`> 表格摘要`、`> VL 校读`、
            #   `![表格原图]` 语义上属于这张表，丢掉它们等于丢了表格的解释
            j = i + 1
            while j < len(lines) and _is_annotation(lines[j]):
                block_lines.append(lines[j])
                j += 1
            i = j
            segments.append(("\n".join(block_lines).strip(), True))
            continue
        buf.append(line)
        i += 1
    flush()
    return segments


class ProtectedBlockChunker:
    """包装任意基础切分器：受保护块整体成块，其余走基础切分。

    受保护块的 metadata 打 `atomic=True`，供 merge_short_chunks 跳过合并；
    parent_text 置为块自身内容（parent_child 策略下 format_doc_context 会读它）。
    """

    # 超过这个长度说明表格极大，embedding 侧会截断（TruncatingEmbeddings 6000 字）
    OVERSIZE_CHARS = 6000

    def __init__(self, base):
        self.base = base

    def chunk(self, text: str, metadata: dict = None) -> list[Chunk]:
        metadata = metadata or {}
        out: list[Chunk] = []
        for seg, is_atomic in split_protected_blocks(text):
            if is_atomic:
                if len(seg) > self.OVERSIZE_CHARS:
                    logger.warning(
                        f"受保护块超过 {self.OVERSIZE_CHARS} 字（{len(seg)} 字），"
                        "嵌入时会被截断，检索可能只能命中前半部分"
                    )
                out.append(Chunk(text=seg, metadata={
                    **metadata, "atomic": True,
                    "parent_text": seg, "chunk_index": len(out),
                }))
            else:
                for c in self.base.chunk(seg, metadata):
                    c.metadata["chunk_index"] = len(out)
                    c.metadata.pop("atomic", None)
                    out.append(c)
        return out


def get_chunker(config: ChunkingConfig, embedding_model: DashScopeEmbeddings = None):
    if config.strategy == "semantic":
        if embedding_model is None:
            embedding_model = DashScopeEmbeddings(
                model=get_settings().EMBEDDING_MODEL,
                dashscope_api_key=get_settings().DASHSCOPE_API_KEY,
            )
        base = SemanticChunkerWrapper(embedding_model)
    elif config.strategy == "parent_child":
        base = ParentChildChunker(config)
    else:
        base = FixedChunker(config)

    if config.protect_blocks:
        return ProtectedBlockChunker(base)
    return base
