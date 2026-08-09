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

    async def parse(
        self, file_path: str, formula_enable: bool = True,
        return_content_list: bool = False,
    ) -> tuple[str, list[int], dict[str, bytes], list[dict], list[dict]]:
        """
        Returns: (markdown_content, page_numbers_per_chunk, images_dict,
                  table_blocks, equation_blocks)
            images_dict: key=文件名, value=图片 bytes
            table_blocks: return_content_list=True 时，MinerU content_list 的表格块
                [{page_idx(0-based), bbox[x0,y0,x1,y1](归一化), table_body(HTML|None)}]
            equation_blocks: 公式块 [{page_idx, bbox(归一化), text(LaTeX $$..$$)}]

        MinerU 优先（高精度表格识别），失败降级到 LlamaIndex。
        ★ MinerU 返回的 page_number 取真实值，不是硬编码 0。
        formula_enable: True=公式输出 LaTeX 文本。
        return_content_list: True 时请求 content_list，供表格/公式原图定位。
        """
        file_name = Path(file_path).name

        if self.parser == "mineru":
            try:
                (md_text, pages, images, table_blocks, equation_blocks) = (
                    await parse_document(
                        file_path, file_name, formula_enable=formula_enable,
                        return_content_list=return_content_list,
                    )
                )
                if md_text and len(md_text.strip()) > 10:
                    logger.info(
                        f"MinerU 解析成功: {file_name} ({len(md_text)} chars, "
                        f"{len(images)} images, {len(table_blocks)} tables, "
                        f"{len(equation_blocks)} equations)"
                    )
                    return md_text, pages, images, table_blocks, equation_blocks
            except Exception as e:
                logger.warning(f"MinerU 解析失败，降级到 LlamaIndex: {e}")

        logger.info(f"使用 LlamaIndex 解析: {file_name}")
        return await self._parse_with_llamaindex(file_path)

    async def _parse_with_llamaindex(
        self, file_path: str,
    ) -> tuple[str, list[int], dict[str, bytes], list[dict], list[dict]]:
        """LlamaIndex 兜底解析"""
        from llama_index.core import SimpleDirectoryReader

        try:
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()
        except Exception as e:
            logger.error(f"LlamaIndex 解析失败: {e}")
            return "", [], [], [], []

        if not documents:
            return "", [], [], [], []

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

        return full_text, pages, {}, [], []
