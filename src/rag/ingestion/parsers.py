# ============================================================
# 文档解析器：MinerU 优先 → LlamaIndex 降级
#
# ★ 补齐医疗版空文件
# ============================================================

from pathlib import Path

from loguru import logger

from src.knowledge.mineru_client import parse_document


class DocumentParser:
    """PDF/DOCX → markdown + page_number"""

    def __init__(self, parser: str = "mineru"):
        self.parser = parser

    async def parse(self, file_path: str) -> tuple[str, list[int]]:
        """
        Returns: (markdown_content, page_numbers_per_chunk)

        MinerU 优先（高精度表格识别），失败降级到 LlamaIndex。
        ★ MinerU 返回的 page_number 取真实值，不是硬编码 0。
        """
        file_name = Path(file_path).name

        if self.parser == "mineru":
            try:
                md_text, pages = await parse_document(file_path, file_name)
                if md_text and len(md_text.strip()) > 10:
                    logger.info(f"MinerU 解析成功: {file_name} ({len(md_text)} chars)")
                    return md_text, pages
            except Exception as e:
                logger.warning(f"MinerU 解析失败，降级到 LlamaIndex: {e}")

        logger.info(f"使用 LlamaIndex 解析: {file_name}")
        return await self._parse_with_llamaindex(file_path)

    async def _parse_with_llamaindex(self, file_path: str) -> tuple[str, list[int]]:
        """LlamaIndex 兜底解析"""
        from llama_index.core import SimpleDirectoryReader
        from llama_index.core.node_parser import SentenceSplitter

        try:
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()
        except Exception as e:
            logger.error(f"LlamaIndex 解析失败: {e}")
            return "", []

        if not documents:
            return "", []

        # 合并所有文档的文本
        full_text = "\n\n".join(doc.get_content() for doc in documents)

        # 提取页码
        pages = []
        for doc in documents:
            page = doc.metadata.get("page_label", 0)
            try:
                pages.append(int(page))
            except (ValueError, TypeError):
                pages.append(0)

        return full_text, pages
