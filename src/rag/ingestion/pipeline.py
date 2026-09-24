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

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from pymilvus import DataType, Function, FunctionType, MilvusClient

from src.core.config import get_settings
from src.infra.milvus_client import escape_milvus_string
from src.infra.minio_client import ensure_bucket_exists, upload_file
from src.knowledge.acl import ACL_FIELD
from src.rag.config import ChunkingConfig
from src.rag.ingestion.chunkers import get_chunker, merge_short_chunks
from src.rag.ingestion.image_summarizer import summarize_table
from src.rag.ingestion.parsers import DocumentParser

if TYPE_CHECKING:
    import pymupdf

settings = get_settings()

COLLECTION_NAME = "alm_docs"
EMBEDDING_DIM = 1024

# 匹配 markdown 中的 HTML 表格（MinerU 复杂表格输出为 <table>）
TABLE_HTML_RE = re.compile(r"<table\b[\s\S]*?</table>", re.IGNORECASE)

# 行间公式 $$..$$（公式原图只嵌入行间公式）
FORMULA_BLOCK_RE = re.compile(r"\$\$[\s\S]+?\$\$")


def _normalize_anchor(text: str) -> str:
    """去空白用于锚点匹配（与表格/公式匹配同一套思路：忽略排版差异）"""
    return re.sub(r"[\s\u00a0]+", "", text or "")


# 章节归属时前向搜索的窗口（归一化字符数）
_SECTION_SEARCH_WINDOW = 20000

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)


def _normalize_with_pos(text: str) -> tuple[str, list[int]]:
    """去空白并保留「归一化下标 → 原文下标」映射，用于把位置还原回原文。"""
    out: list[str] = []
    pos: list[int] = []
    for i, ch in enumerate(text):
        if not ch.isspace():
            out.append(ch)
            pos.append(i)
    return "".join(out), pos


def _heading_paths(md_text: str) -> list[tuple[int, str]]:
    """提取 markdown 标题及其在「归一化文本」中的下标，返回 [(norm_idx, 章节路径)]。

    章节路径按层级拼成 `一级 > 二级 > 三级`，用于给 chunk 补上下文锚。
    技术文档的标题本身就是强语义信号，把它写进被索引的文本，
    同时进入稠密与 BM25 两路。
    """
    norm_md, _ = _normalize_with_pos(md_text)
    # 逐行扫原始文本，按标题层级维护栈
    stack: list[tuple[int, str]] = []   # (level, title)
    result: list[tuple[int, str]] = []
    cursor = 0
    for m in _HEADING_RE.finditer(md_text):
        level = len(m.group(1))
        title = m.group(2).strip()
        if not title:
            continue
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        # H1 通常是文档标题，与前缀里的「文档：xxx」重复，不进章节路径
        path = " > ".join(t for lv, t in stack if lv > 1)
        key = _normalize_anchor(title)[:20]
        if not key:
            continue
        idx = norm_md.find(key, cursor)
        if idx < 0:
            continue
        cursor = idx
        result.append((idx, path))
    return result


def _assign_chunk_sections(chunks, md_text: str) -> list[str]:
    """给每个 chunk 归属它所属的最后一个标题路径；无法定位则返回空串。

    位置用「chunk 开头若干字符在归一化全文里的首次出现」定位，游标只前进
    （chunk 与标题都是文档序）。定位不到就继承上一个 chunk 的章节——
    章节归属在文档序上是单调的，继承比留空更接近真相，且不会张冠李戴。
    """
    if not chunks:
        return []
    headings = _heading_paths(md_text)
    if not headings:
        return [""] * len(chunks)

    norm_md, _ = _normalize_with_pos(md_text)
    out: list[str] = []
    cursor = 0
    last = ""
    for chunk in chunks:
        key = _normalize_anchor(getattr(chunk, "text", "") or "")[:40]
        if key:
            # ★ 限定前向窗口：无界 find 会因一次误配把游标甩到文末，
            #   之后所有 chunk 都继承同一个章节（页码链路踩过同样的坑）。
            #   窗口取 2 万字符 ≈ 40 个块，足够宽裕又不会跨全文乱跳。
            idx = norm_md.find(key, cursor, cursor + _SECTION_SEARCH_WINDOW)
            if idx >= 0:
                cursor = idx
                # 取最后一个起点不晚于该 chunk 的标题
                for h_idx, path in headings:
                    if h_idx <= idx:
                        last = path
                    else:
                        break
        out.append(last)
    return out


def _build_chunk_prefix(meta, section: str) -> str:
    """构造写入被索引文本的前缀（检索用上下文锚）。

    ★ 只拼进 `text`（embedding 与 BM25 的实际输入），不改 `parent_text`：
    parent_text 是给 LLM 生成与前端展示用的，doc_name 已经在引用里出现，
    再拼一遍是噪音。
    """
    parts = []
    if meta.doc_name:
        parts.append(f"文档：{Path(meta.doc_name).stem}")
    if meta.doc_type:
        parts.append(f"类型：{meta.doc_type}")
    if meta.model_code:
        parts.append(f"车型：{meta.model_code}")
    if section:
        parts.append(f"章节：{section}")
    return f"【{'｜'.join(parts)}】\n" if parts else ""


def _char_ngrams(text: str, n: int = 4) -> set[str]:
    """字符级 n-gram 集合，用于容忍插入/重排的模糊匹配。

    纯前缀匹配不够用：chunk 是块序列切出来的，开头可能与块的起点错位
    （表格/公式原图注解会插进正文），前缀对不上却实际同源。
    实测反例：chunk「这是第五章的正文内容 制动系统的补充说明」与块
    「第五章 制动系统 这是第五章的正文内容」——前缀互不包含，但 n-gram
    重叠 8/16，能正确判为同页。
    """
    if len(text) <= n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _assign_chunk_pages(
    chunks,
    page_anchors: list[tuple[int, str]],
    key_chars: int = 80,
    min_overlap: float = 0.3,
    window: int = 150,
    max_page_jump: int = 30,
    confident_overlap: float = 0.8,
) -> list[int]:
    """把每个 chunk 映射回页码，返回与 chunks 等长的页码列表（0 = 未知）。

    ★ 为什么不能用 `pages[chunk_idx]`：那两者的下标空间不同——content_list
    的下标是「块」，chunk_idx 是「切片」，一个块可能被切成多个 chunk，
    一个 chunk 也可能横跨多个块。用块下标取切片页码会得到看似合理却错误的
    页码，而错误的引用比没有引用更糟，所以宁可留 0。

    做法：chunk 与锚点都是文档序，用一个只前进的游标做 n-gram 重叠匹配。
    - 取 chunk 开头若干字符的 n-gram 集合，在游标之后的**窗口内**找最优锚点；
    - **窗口内取最佳重叠**（不是全局取最佳，也不是窗口内取首个）：
      全局取最佳会让一次错误匹配把游标带到文末——实测 289 页手册上 727 块里
      有 604 块因此塌到最后一页；而窗口内取首个会错配到相似锚点
      （相邻页的标题往往只差一个编号，「章节7…」会命中「章节1…」）；
    - **单块页码跳变守卫**：一个 chunk 不该横跨 30 页以上，超出即视为误配并
      不推进游标。这是最后一道防线——前两道被绕过时它仍能挡住塌陷；
    - 未命中但此前已有命中 → 沿用上一个已知页码（文档序下的合理推断）；
    - 一个有意义的命中都没有 → 全部 0，绝不猜。
    """
    if not chunks:
        return []
    if not page_anchors:
        return [0] * len(chunks)

    anchor_grams = [_char_ngrams(_normalize_anchor(t)) for _, t in page_anchors]
    pages: list[int] = []
    cursor = 0
    last_known = 0
    matched_any = False
    for chunk in chunks:
        key = _normalize_anchor(getattr(chunk, "text", "") or "")[:key_chars]
        if key:
            key_grams = _char_ngrams(key)
            denom = max(1, len(key_grams))
            best_i, best_score = -1, 0.0
            for i in range(cursor, min(cursor + window, len(page_anchors))):
                score = len(key_grams & anchor_grams[i]) / denom
                if score > best_score:
                    best_score, best_i = score, i
                if score >= 0.9:
                    break
            if best_i >= 0 and best_score >= min_overlap:
                page = page_anchors[best_i][0]
                # 软跳变守卫：低重叠加远跳 = 误配（真实远跳会近乎精确匹配）
                wild = (matched_any and page - last_known > max_page_jump
                        and best_score < confident_overlap)
                if not wild:
                    cursor = best_i
                    last_known = page
                    matched_any = True
        # 一个都没匹配过时 last_known 仍是 0（未知），不制造虚假页码
        pages.append(last_known if matched_any else 0)
    return pages


def _normalize_table_html(html: str) -> str:
    """去 HTML 标签与空白，用于 content_list table_body 与 md <table> 的匹配"""
    text = re.sub(r"<[^>]+>", "", html or "")
    return re.sub(r"[\s\u00a0]+", "", text)



def _normalize_formula(text: str) -> str:
    """去所有空白，用于 content_list equation.text 与 md $$..$$ token 的匹配"""
    return re.sub(r"[\s\u00a0]+", "", text or "")


def _format_vl_note(summary: str, vl_md: str) -> str:
    """VL 表格增强结果格式化为引用块：摘要（可检索文本）+ 校读 Markdown 表格。
    两者都可能为空（fail-open 下单侧缺失）。"""
    lines: list[str] = []
    if summary:
        lines.append(f"> 表格摘要：{summary}")
    if vl_md:
        if lines:
            lines.append(">")
        lines.append("> VL 校读：")
        lines.extend(f"> {ln}".rstrip() for ln in vl_md.strip().splitlines())
    return "\n".join(lines)


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _extract_numbers(text: str) -> set[float]:
    """提取文本中的数值集合（去千分位逗号、剥 HTML 标签、归一化 1.50==1.5）"""
    text = re.sub(r"(?<=\d),(?=\d)", "", text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return {round(float(m), 4) for m in _NUM_RE.findall(text)}


def _table_numbers_mismatch(html: str, vl_md: str) -> tuple[set[float], set[float]] | None:
    """双通道数值互查：比较 MinerU HTML 与 VL 校读的数值集合。

    返回 (仅 HTML 有, 仅 VL 有)；任一侧为空或无数值时不可比，返回 None。
    判定策略：只有当两侧各自都存在对方没有的数值时才算实质分歧——
    单侧多出的条款号/行号（VL 常省略）不算错误。"""
    if not vl_md or not html:
        return None
    html_nums = _extract_numbers(html)
    vl_nums = _extract_numbers(vl_md)
    if not html_nums or not vl_nums:
        return None
    return html_nums - vl_nums, vl_nums - html_nums


def _fmt_nums(nums: set[float], limit: int = 6) -> str:
    return ", ".join(f"{x:g}" for x in sorted(nums)[:limit])


def _locate_formula_rect(page, bbox: pymupdf.Rect, md_before: str) -> pymupdf.Rect | None:
    """定位公式在页面上的真实渲染区域（pt 坐标）。

    实测 MinerU content_list 的 equation 块 bbox 整体偏高 ~90-110pt、且高度不可靠，
    直接用会裁到公式上方正文（如“（1）（2）”列表行）。因此：
      1) 优先用 md 中公式前文本的尾行做锚点（page.search_for），锚点下方即公式顶；
         公式底/宽用渲染密度扫描定：连续 >=3 行密集=正文段（如“式中：”），其顶即底；
         公式内部分数条等单行密集不误判。
      2) 锚点找不到（扫描版 PDF 无文本层）退化为在 bbox 下方扫描“稀疏公式带”。
      3) 全部失败返回 None，调用方退回 bbox 留白裁剪。
    """
    import pymupdf

    def _scan_band(top: float, bot: float):
        clip_top = max(0.0, top)
        clip_bot = min(page.rect.height, bot)
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(2.0, 2.0),
            clip=pymupdf.Rect(0, clip_top, page.rect.width, clip_bot),
            alpha=False,
        )
        w, h = pix.width, pix.height
        cnt = [0] * h
        xmin: list = [None] * h
        xmax: list = [None] * h
        for yy in range(h):
            c = 0
            mn = None
            mx = None
            for xx in range(w):
                try:
                    v = pix.pixel(xx, yy)[0]
                except TypeError:
                    v = pix.pixel(xx, yy)
                if v < 128:
                    c += 1
                    if mn is None:
                        mn = xx
                    mx = xx
            cnt[yy] = c
            if c:
                xmin[yy] = mn
                xmax[yy] = mx
        thresh = max(50, int(0.05 * w))
        return clip_top, cnt, xmin, xmax, thresh

    def _formula_bottom(ftop, scan_top, cnt, thresh, band_bot):
        """自公式顶向下：连续 >=3 行密集=正文段，其顶即公式底；内容空隙 >=6pt 则停。"""
        dense_run = 0
        prev = None
        for yy in range(len(cnt)):
            y = yy / 2.0 + scan_top
            if y <= ftop:
                continue
            if cnt[yy] >= thresh:
                dense_run += 1
                if dense_run >= 3:
                    return y - (dense_run - 1) / 2.0
            else:
                if dense_run:
                    dense_run = 0
                elif cnt[yy] > 0:
                    if prev is not None and y - prev >= 6:
                        return prev + 1.0
                    prev = y
        last = None
        for yy in range(len(cnt) - 1, -1, -1):
            if cnt[yy] > 0:
                last = yy / 2.0 + scan_top
                break
        return min(last + 1.0 if last else ftop + 40.0, band_bot)

    def _crop(ftop, fbot, scan_top, cnt, xmin, xmax):
        xs = []
        for yy in range(len(cnt)):
            if xmin[yy] is not None and ftop - 3 <= yy / 2.0 + scan_top <= fbot + 3:
                xs.append(xmin[yy])
                xs.append(xmax[yy])
        if not xs:
            return None
        return pymupdf.Rect(min(xs) / 2.0 - 8.0, ftop - 4.0, max(xs) / 2.0 + 10.0, fbot + 4.0)

    # 1) 锚点定位：md 公式前文本尾行，逐步缩短后缀 search_for（多命中取离 bbox 近的）
    anchor_bot = None
    btop = bbox.y0
    tail = (md_before or "").strip().split("\n")[-1]
    norm = re.sub(r"[\s\u00a0]+", "", tail)
    for ln in range(min(len(norm), 24), 7, -1):
        hits = page.search_for(norm[-ln:])
        if hits:
            hits = sorted(hits, key=lambda r: abs((r.y0 + r.y1) / 2 - (btop + 110.0)))
            anchor_bot = hits[0].y1
            break
    if anchor_bot is not None:
        scan_top = max(0.0, anchor_bot - 4.0)
        scan_bot = min(page.rect.height, anchor_bot + 70.0)
        scan_top, cnt, xmin, xmax, thresh = _scan_band(scan_top, scan_bot)
        ftop = None
        for yy in range(len(cnt)):
            if cnt[yy] > 0 and yy / 2.0 + scan_top > anchor_bot + 0.5:
                ftop = yy / 2.0 + scan_top
                break
        if ftop is not None:
            fbot = _formula_bottom(ftop, scan_top, cnt, thresh, scan_bot)
            crop = _crop(ftop, fbot, scan_top, cnt, xmin, xmax)
            if crop is not None:
                return crop

    # 2) 兜底：bbox 下方扫稀疏公式带（公式实测位于 bbox 下 ~110pt）
    scan_top = max(0.0, btop - 20.0)
    scan_bot = min(page.rect.height, btop + 170.0)
    scan_top, cnt, xmin, xmax, thresh = _scan_band(scan_top, scan_bot)
    runs = []
    cur = None
    for yy in range(len(cnt)):
        if cnt[yy] > 0:
            if cur is None:
                cur = [yy, yy]
            elif yy - cur[1] > 8:
                runs.append(cur)
                cur = [yy, yy]
            else:
                cur[1] = yy
        elif cur is not None:
            runs.append(cur)
            cur = None
    if cur:
        runs.append(cur)
    best = None
    for s, e in runs:
        y0p = s / 2.0 + scan_top
        y1p = e / 2.0 + scan_top
        if y1p - y0p < 8:
            continue
        score = abs((y0p - btop) - 110.0)
        if best is None or score < best[0]:
            best = (score, y0p, y1p)
    if best:
        crop = _crop(best[1], best[2], scan_top, cnt, xmin, xmax)
        if crop is not None:
            return crop
    return None


@dataclass
class DocMetadata:
    doc_name: str
    doc_type: str          # repair_manual / spec_doc / tsb / issue_case
    category: str = ""
    business_line: str = ""
    model_code: str = ""   # ★ 汽车域刚需：按车型过滤
    # 可见角色（检索前 ACL 过滤依据，见 knowledge/acl.py）。
    # 默认空 = 仅 admin 可见（检索谓词匹配不到任何角色）——fail-closed：
    # 宁可让上传者显式声明，也不要"忘了填"变成所有人可见。
    acl_roles: list[str] = field(default_factory=list)


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

    def _ensure_acl_field(self) -> None:
        """已存在的 collection 补 ACL 字段（add_collection_field，非破坏性）。

        旧 collection 没有该字段时直接插入会报错；删库重建则是灾难。
        Milvus 2.6 支持给存量 collection 增补标量字段，这里幂等补上。
        ★ 存量行的 acl_roles 为 null，ACL 谓词匹配不到 → 对非 admin 不可见
        （fail-closed）。必须跑 scripts/backfill_acl.py 显式赋权后存量才可见。
        """
        try:
            desc = self.milvus.describe_collection(COLLECTION_NAME)
        except Exception as e:
            logger.warning(f"describe_collection 失败，跳过 ACL 字段检查: {e}")
            return
        fields = desc.get("fields") or (desc.get("schema") or {}).get("fields") or []
        if any(f.get("name") == ACL_FIELD for f in fields):
            return
        try:
            self.milvus.add_collection_field(
                COLLECTION_NAME,
                field_name=ACL_FIELD,
                data_type=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=16,
                max_length=32,
                nullable=True,
            )
            logger.warning(
                f"collection '{COLLECTION_NAME}' 已补 ACL 字段 '{ACL_FIELD}'；"
                "存量 chunk 的可见角色为空（对非 admin 不可见，fail-closed），"
                "请运行 scripts/backfill_acl.py 为存量文档赋权"
            )
        except Exception as e:
            # 字段已存在（并发实例化）不算失败；其他失败要让入库早暴露
            if "field already exist" not in str(e).lower():
                raise

    def _ensure_collection(self) -> None:
        """确保 alm_docs collection 存在，含 BM25 Function（与天宫医疗一致）。
        幂等：已存在且含 BM25 Function 则跳过；
        仅旧版 collection（无 BM25 Function）才删除重建，避免每次实例化都清空索引。"""
        if self.milvus.has_collection(COLLECTION_NAME):
            if self._collection_has_bm25():
                logger.info(
                    f"collection '{COLLECTION_NAME}' 已存在且含 BM25 Function，跳过重建"
                )
                self._ensure_acl_field()
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
        # 可见角色数组：检索前 ACL 过滤的依据（array_contains_any 谓词，见 knowledge/acl.py）
        schema.add_field(ACL_FIELD, DataType.ARRAY,
                         element_type=DataType.VARCHAR, max_capacity=16, max_length=32)
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

        # 1. parse — MinerU 返回
        # (md_text, pages, images_dict, table_blocks, equation_blocks, page_anchors)
        (md_text, pages, images, table_blocks, equation_blocks, page_anchors) = (
            await self.parser.parse(
                file_path,
                return_content_list=(
                    settings.TABLE_ORIGINALS_ENABLED or settings.FORMULA_IMAGE_ENABLED
                ),
            )
        )
        if not md_text or len(md_text.strip()) < 10:
            # 抛错而非静默返回：静默会让空文档标成 indexed 上线，检索永远召回不到
            raise ValueError(f"文档解析后无内容: {meta.doc_name}")

        # ★ 公式原图对照（bbox 裁剪）：content_list 的 equation 块带 bbox，
        #   渲染页面裁剪出公式原图嵌入对应公式下方，防 LaTeX 识别不准确
        if settings.FORMULA_IMAGE_ENABLED and equation_blocks:
            md_text = await self._embed_formula_originals(
                md_text, equation_blocks, file_path, doc_id, meta.doc_name,
            )

        # ★ 表格原图对照（bbox 裁剪）：渲染页面裁剪出表格原图嵌入对应表格下方，
        #   防复杂表格（colspan/rowspan）OCR 串行/丢列；跨页表格续页块归并到主块
        if settings.TABLE_ORIGINALS_ENABLED and table_blocks:
            md_text = await self._embed_table_originals(
                md_text, table_blocks, file_path, doc_id, meta.doc_name,
            )

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
            raise ValueError(f"切片后无内容: {meta.doc_name}")

        # ★ 页码映射：用解析期留下的页码锚点把 chunk 定位回页，
        #   不能用 pages[chunk_idx]（块下标 ≠ 切片下标，会得到错误页码）。
        chunk_pages = _assign_chunk_pages(chunks, page_anchors)
        located = sum(1 for p in chunk_pages if p)
        if page_anchors and located < len(chunks):
            logger.warning(
                f"页码定位部分失败: {meta.doc_name} "
                f"{len(chunks) - located}/{len(chunks)} 个 chunk 只能沿用前一个已知页码"
            )
        if not any(chunk_pages):
            logger.warning(
                f"全部 chunk 页码未知（page_number=0）: {meta.doc_name}；"
                "引用将不显示页码，页级评测标签也无法使用"
            )

        # ★ 章节归属 + 检索用前缀：把「哪本手册/哪个车型/哪一章」写进被索引的
        #   text，同时进入稠密与 BM25 两路索引（contextual retrieval 的零 LLM 版）。
        chunk_sections = _assign_chunk_sections(chunks, md_text)
        prefixed = 0
        for c, section in zip(chunks, chunk_sections, strict=False):
            prefix = _build_chunk_prefix(meta, section)
            if prefix:
                c.text = prefix + c.text
                prefixed += 1
        logger.info(
            f"上下文前缀: {prefixed}/{len(chunks)} 个 chunk 已加前缀"
            + (f"（{sum(1 for s in chunk_sections if s)} 个定位到章节）" if prefixed else "")
        )

        texts = [c.text for c in chunks]

        # 3. embed (only dense — sparse 由 Milvus BM25 Function 自动生成)
        dense_vecs = await dense_embedder.embed(texts)

        # ★ 数量校验在删旧数据之前：嵌入部分失败（数量不符）时保留旧版本文档，
        #   让本次入库失败可重试，而不是"旧的删了、新的进不去"
        if len(dense_vecs) != len(texts):
            raise RuntimeError(
                f"嵌入数量不符: 预期 {len(texts)} 实际 {len(dense_vecs)}，"
                f"保留旧版本数据: {meta.doc_name}"
            )

        # 幂等：解析/切片/嵌入全部成功后才删旧数据，失败时保留旧版本文档
        await asyncio.to_thread(
            self.milvus.delete,
            collection_name=COLLECTION_NAME,
            filter=f'doc_id == "{escape_milvus_string(doc_id)}"',
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
                    "acl_roles": meta.acl_roles[:16],
                    "page_number": chunk_pages[chunk_idx] if chunk_idx < len(chunk_pages) else 0,
                    "chunk_index": chunk_idx,
                    "parent_text": chunk.metadata.get("parent_text", "")[:65000],
                    "text": chunk.text[:65000],
                    "image_urls": json.dumps(image_urls, ensure_ascii=False)[:65000],
                    "embedding": batch_dense[j] if j < len(batch_dense) else [],
                }
                all_data.append(record)

        if all_data:
            await asyncio.to_thread(
                self.milvus.insert, collection_name=COLLECTION_NAME, data=all_data,
            )
            logger.info(
                f"入库完成: {meta.doc_name} doc_id={doc_id} chunks={len(all_data)} "
                f"acl_roles={meta.acl_roles or '（仅 admin 可见）'}"
            )

        return doc_id

    # ============================================================
    # 公式原图对照（bbox 裁剪）
    # ============================================================

    async def _embed_formula_originals(
        self, md_text: str, equation_blocks: list[dict],
        file_path: str, doc_id: str, doc_name: str,
    ) -> str:
        """公式原图嵌入 markdown：渲染页面 + bbox 裁剪公式原图，嵌到对应公式下方。

        MinerU content_list 的 equation 块携带完整 LaTeX（$$..$$）+ bbox + page_idx。
        匹配规则：归一化 LaTeX（去空白）与 md 中行间公式 token 按文档顺序一一对应，
        找到后在该公式后插入原图引用；匹配不上的块走文末『公式原图（识别对照）』附录。
        fail-open：裁剪/上传失败只跳过单块，不阻塞入库。"""
        if not equation_blocks:
            return md_text

        ordered = sorted(
            equation_blocks,
            key=lambda b: (
                int(b.get("page_idx", 0) or 0),
                (b.get("bbox") or [0, 0, 0, 0])[1],
            ),
        )
        md_formulas = list(FORMULA_BLOCK_RE.finditer(md_text))

        refs: list[tuple[int, int, str]] = []  # (start, end, ref)
        appendix: list[str] = []
        fi = 0
        for block in ordered:
            eq_norm = _normalize_formula(block.get("text"))
            # 先按文档顺序匹配 md 行间公式 token：取公式前文本作锚点，
            # 传给 _render_block_original 用“锚点+密度扫描”定位真实渲染区域
            # （MinerU equation bbox 整体偏高 ~90-110pt，直接裁会裁到上方正文）
            if eq_norm:
                while fi < len(md_formulas) and _normalize_formula(
                    md_formulas[fi].group(0)
                ) != eq_norm:
                    fi += 1
            matched = fi < len(md_formulas)
            md_before = md_text[: md_formulas[fi].start()] if matched else ""
            ref = await self._render_block_original(
                block, file_path, doc_id, doc_name,
                kind="formula", md_before=md_before,
            )
            if not ref:
                continue
            if matched:
                refs.append((md_formulas[fi].start(), md_formulas[fi].end(), ref))
                fi += 1
            else:
                appendix.append(ref)

        if not refs and not appendix:
            return md_text

        # 在对应公式后插入原图引用
        out: list[str] = []
        pos = 0
        for _start, end, ref in sorted(refs):
            out.append(md_text[pos:end])
            out.append(f"\n\n{ref}\n")
            pos = end
        out.append(md_text[pos:])
        md_text = "".join(out)

        if appendix:
            md_text += "\n\n## 公式原图（识别对照）\n" + "\n\n".join(appendix)
            logger.info(f"公式原图 {len(appendix)} 张无法定位，转文末附录: {doc_name}")

        logger.info(f"公式原图嵌入完成 {len(refs)} 张（逐公式对照）: {doc_name}")
        return md_text

    # ============================================================
    # 表格原图对照（bbox 裁剪）
    # ============================================================

    async def _embed_table_originals(
        self, md_text: str, table_blocks: list[dict],
        file_path: str, doc_id: str, doc_name: str,
    ) -> str:
        """表格原图嵌入 markdown：渲染页面 + bbox 裁剪表格原图，嵌到对应 </table> 后。

        跨页表格结构（实测 MinerU 3.4.4）：
          - 主块携带合并后的完整 table_body HTML + 首页 bbox
          - 每个续页块 table_body=None，仅带该页 bbox
        匹配规则：主块按文档顺序与 md 中 <table> 一一对应（归一化 body 子串校验）；
          续页块裁剪图归并到最近的已匹配主块，多张图一起嵌到那一个 </table> 后。
        fail-open：裁剪/上传失败只跳过单块；无法匹配到 md 表格的块走文末
          『表格原图（识别对照）』附录，保证原图不丢。"""
        if not table_blocks:
            return md_text

        md_tables = list(TABLE_HTML_RE.finditer(md_text))
        vl_enabled = settings.TABLE_VL_ENABLED

        # 1) 按文档顺序排所有表格块，逐块匹配
        ordered = sorted(
            table_blocks,
            key=lambda b: (
                int(b.get("page_idx", 0) or 0),
                (b.get("bbox") or [0, 0, 0, 0])[1],
            ),
        )
        table_images: dict[int, list[str]] = {}  # md table 索引 -> 图片引用
        vl_notes: dict[int, list[str]] = {}      # md table 索引 -> VL 摘要+校读引用块
        appendix: list[str] = []
        t_idx = 0
        last_matched: int | None = None

        for block in ordered:
            bbox = block.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            capture: dict = {} if vl_enabled else None
            img_ref = await self._render_block_original(
                block, file_path, doc_id, doc_name, kind="table", capture=capture,
            )
            if not img_ref:
                continue

            body = block.get("table_body")
            if body:
                body_norm = _normalize_table_html(body)
                while (
                    t_idx < len(md_tables)
                    and body_norm
                    not in _normalize_table_html(md_tables[t_idx].group(0))
                ):
                    t_idx += 1
                if t_idx < len(md_tables):
                    last_matched = t_idx
                    table_images.setdefault(t_idx, []).append(img_ref)
                    # ★ 表格 VL 增强：对主块原图做语义摘要 + Markdown 校读（fail-open）
                    if vl_enabled and capture.get("png"):
                        try:
                            summary, vl_md = await summarize_table(capture["png"], "image/png")
                            note = _format_vl_note(summary, vl_md)
                            if note:
                                # ★ 数值互查：两份独立转写的数字集合交叉比对
                                diff = _table_numbers_mismatch(body, vl_md)
                                if diff and diff[0] and diff[1]:
                                    note += (
                                        "\n> ⚠️ 双通道转写数值不一致，以原图为准"
                                        f"（HTML 有：{_fmt_nums(diff[0])}；"
                                        f"校读有：{_fmt_nums(diff[1])}）"
                                    )
                                    logger.warning(
                                        f"表格双通道转写数值不一致 {doc_name} "
                                        f"page={block.get('page_idx')}: "
                                        f"HTML有[{_fmt_nums(diff[0])}] 校读有[{_fmt_nums(diff[1])}]"
                                    )
                                vl_notes.setdefault(t_idx, []).append(note)
                        except Exception as e:
                            logger.warning(f"表格 VL 增强失败（fail-open）: {e}")
                    t_idx += 1
                else:
                    last_matched = None
                    appendix.append(img_ref)
            else:
                # 续页块（table_body=None）：归并到最近匹配的主块
                if last_matched is not None:
                    table_images.setdefault(last_matched, []).append(img_ref)
                else:
                    appendix.append(img_ref)

        if not table_images and not appendix:
            return md_text

        # 2) 在 </table> 后插入对应图片引用与 VL 增强块
        out: list[str] = []
        pos = 0
        for i, m in enumerate(md_tables):
            out.append(md_text[pos:m.end()])
            for ref in table_images.get(i, []):
                out.append(f"\n\n{ref}\n")
            for note in vl_notes.get(i, []):
                out.append(f"\n\n{note}\n")
            pos = m.end()
        out.append(md_text[pos:])
        md_text = "".join(out)

        # 3) 无法定位的块走文末附录
        if appendix:
            md_text += "\n\n## 表格原图（识别对照）\n" + "\n\n".join(appendix)
            logger.info(f"表格原图 {len(appendix)} 张无法定位，转文末附录: {doc_name}")

        logger.info(
            f"表格原图嵌入完成 {sum(len(v) for v in table_images.values())} 张，"
            f"VL 校读 {sum(len(v) for v in vl_notes.values())} 张: {doc_name}"
        )
        return md_text

    async def _render_block_original(
        self, block: dict, file_path: str, doc_id: str, doc_name: str,
        kind: str = "table", md_before: str = "", capture: dict | None = None,
    ) -> str | None:
        """渲染 page_idx 页面，按归一化 bbox 裁剪表格/公式区域 → 上传 MinIO → 返回引用。

        bbox 坐标系：MinerU 归一化（页面渲染宽度=842），换算到 PDF pt：
          pt = norm * 页宽 / 842。
        kind=table  （表格原图）：bbox 底部常裁掉最后一行（实测偏短 40~55pt），
            用该区域文本块向下吸附，无文本层（扫描版）则用固定 60pt 下边距兜底；
        kind=formula（公式原图）：MinerU equation bbox 整体偏高 ~90-110pt 且高度不可靠，
            优先用 _locate_formula_rect（公式前文本锚点 + 渲染密度扫描）定位真实区域，
            md_before 传公式前 markdown；定位失败退回 bbox 上下留白裁剪。
        capture：传 dict 时把裁剪出的 png 字节带出（表格 VL 校读复用同一张图，不重复渲染）。
        fail-open：pymupdf 缺失/裁剪异常返回 None，不阻塞入库。"""
        import pymupdf  # 延迟导入，pymupdf 缺失时跳过原图

        bbox = block.get("bbox")
        page_idx = int(block.get("page_idx", 0) or 0)
        try:
            doc = pymupdf.open(file_path)
            if page_idx < 0 or page_idx >= doc.page_count:
                return None
            page = doc[page_idx]

            u2pt = page.rect.width / 842.0
            x0, y0, x1, y1 = bbox
            rect = pymupdf.Rect(
                x0 * u2pt, y0 * u2pt, x1 * u2pt, y1 * u2pt,
            )
            if kind == "formula":
                # 公式：优先用锚点+密度扫描定位真实渲染区域（bbox 偏高需修正）
                crop = _locate_formula_rect(page, rect, md_before or "")
                if crop is None:
                    # 兜底：bbox 上下各留白（上标/下标），右侧兜底
                    crop = pymupdf.Rect(
                        rect.x0 - 6.0, rect.y0 - 14.0,
                        rect.x1 + 20.0, rect.y1 + 18.0,
                    )
                padded = crop
            else:
                # 表格：底部固定 60pt 下边距（bbox 常裁掉最后一行）；右侧 +30pt 兜底
                padded = pymupdf.Rect(
                    rect.x0, rect.y0, rect.x1 + 30.0, rect.y1 + 60.0,
                )
                # 文本块向下吸附：扩展到该区域内最底部文本块，捕捉被裁掉的行
                blocks = [
                    b for b in page.get_text("blocks")
                    if b[0] < padded.x1 and b[2] > padded.x0
                    and b[1] < padded.y1 and b[3] > padded.y0
                ]
                if blocks:
                    padded.y1 = max(padded.y1, max(b[3] for b in blocks))
                    padded.x1 = max(padded.x1, max(b[2] for b in blocks))
            padded &= page.rect
            if padded.width < 20 or padded.height < 20:
                return None

            # 渲染裁剪区域（2x 缩放，保证原图清晰度）
            scale = 2.0
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(scale, scale), clip=padded, alpha=False,
            )
            png_bytes = pix.tobytes("png")
            if capture is not None:
                capture["png"] = png_bytes

            object_name = (
                f"images/{doc_id}/{kind}/"
                f"p{page_idx}_{int(rect.y0)}_{int(rect.x0)}.png"
            )
            await asyncio.to_thread(ensure_bucket_exists)
            await asyncio.to_thread(upload_file, object_name, png_bytes, "image/png")

            scheme = "https" if settings.MINIO_SECURE else "http"
            minio_url = (
                f"{scheme}://{settings.MINIO_ENDPOINT}/"
                f"{settings.MINIO_BUCKET}/{object_name}"
            )
            alt = "表格原图" if kind == "table" else "公式原图"
            return f"![{alt}]({minio_url})"
        except Exception as e:
            logger.error(
                f"{'表格' if kind == 'table' else '公式'}原图裁剪/上传失败 "
                f"{doc_name} page={page_idx}: {e}"
            )
            return None

    async def _upload_images_and_replace_refs(
        self, md_text: str, images: dict[str, bytes], doc_id: str, doc_name: str,
    ) -> str:
        """上传图片到 MinIO；开启 VL 时用 qwen-vl-max 生成图片描述，
        替换 markdown 中 `![](images/xxx.jpg)` 为 `![描述](minio_url)`；
        未被 markdown 引用的"孤儿图"追加到文档末尾，保证图片内容也可被检索。"""
        await asyncio.to_thread(ensure_bucket_exists)
        orphaned: list[str] = []  # 追加用的 ![描述](url) 片段
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
                await asyncio.to_thread(upload_file, object_name, img_bytes, content_type)
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
            referenced = any(
                f"![]({ref})" in md_text or f"]({ref})" in md_text
                for ref in (f"images/{img_name}", img_name)
            )
            for ref in (f"images/{img_name}", img_name):
                md_text = md_text.replace(f"![]({ref})", f"!{alt}({ref})")
                md_text = md_text.replace(f"]({ref})", f"]({minio_url})")

            # ★ 孤儿图：markdown 中无引用，追加到文档末尾使其可被检索
            if not referenced:
                orphaned.append(f"![{summary}]({minio_url})")

        if orphaned:
            md_text = f"{md_text}\n\n## 附：文档配图\n" + "\n\n".join(orphaned)
            logger.info(f"追加 {len(orphaned)} 张未引用图片到文档末尾")

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
