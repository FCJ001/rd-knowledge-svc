#!/usr/bin/env python3
# ============================================================
# 为评测集填充页级标签（relevant_pages）
#
#   python scripts/enrich_page_labels.py --list     # 只看每题的命中页，不写文件
#   python scripts/enrich_page_labels.py            # 写入 eval_dataset_draft.csv
#
# 做法：不猜。每题给一组「必须同时出现的检索词」（主题词 + 答案值），
# 在标注文档的 chunk 里做全量匹配，命中 chunk 的 page_number 就是该题的
# 相关页。答案值本身是最强的定位锚——它出现在哪一页，答案就在哪一页。
#
# ★ 匹配不到就留空，绝不填一个"看起来合理"的页码。
#   错误的页级标签比没有标签更糟：它会让页级指标失去意义。
# ★ 覆盖不了的题（描述型、跨章节型）保持 doc 级，由人工按需补。
# ============================================================

import argparse
import asyncio
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = REPO_ROOT / "scripts" / "eval_dataset_draft.csv"

STD = "纯电动汽车出厂安全技术规范_第1部分_高压部件.pdf"
NOTE = "纯电动汽车出厂安全技术规范_编制说明.pdf"
DESIGN = "电动乘用车安全设计规范.pdf"
CATARC = "CATARC_GB18384.3修改单宣贯讲义.pdf"
MANUAL = "北汽新能源EV160_200维修手册.pdf"

# (问题前缀, 文档, [必须同时出现的词])
SPECS: list[tuple[str, str, list[str]]] = [
    # ── T/ZJSAE 出厂高压部件标准正文 ──
    ("驱动电机定子绕组对机壳的冷态绝缘电阻", STD, ["驱动电机", "20MΩ"]),
    ("驱动电机绕组耐电压试验中漏电流", STD, ["驱动电机", "15mA"]),
    ("驱动电机绕组对温度传感器", STD, ["温度传感器", "1500V"]),
    ("驱动电机控制器动力端子与外壳之间的绝缘电阻", STD, ["控制器", "1MΩ"]),
    ("驱动电机控制器信号端子与外壳之间需要耐受", STD, ["信号端子", "500V"]),
    ("驱动电机控制器耐压试验的持续时间", STD, ["控制器", "10mA"]),
    ("动力电池耐电压试验的漏电流", STD, ["动力电池", "20mA"]),
    ("车载充电机各独立带电端口回路与地", STD, ["车载充电机", "10MΩ"]),
    ("DC/DC 转换器各带电电路与地", STD, ["DC/DC", "10MΩ"]),
    ("电动压缩机清空冷冻机油后", STD, ["压缩机", "50MΩ"]),
    ("电动压缩机充入制冷剂", STD, ["压缩机", "10MΩ"]),
    ("电动压缩机驱动控制器各出线端", STD, ["出线端", "50MΩ"]),
    ("PTC 加热器导线端子与金属外壳", STD, ["PTC", "50MΩ"]),
    ("PTC 加热器输入功率不高于 5kW", STD, ["PTC", "10mA"]),
    ("PTC 加热器输入功率高于 5kW", STD, ["PTC", "20mA"]),
    ("带控制的 PTC 加热器泄漏电流", STD, ["PTC", "5mA"]),
    ("本文件的试验条件中，海拔和相对湿度", STD, ["海拔", "1400"]),
    ("驱动电机绝缘电阻测试时，测试电压", STD, ["250Vdc"]),
    ("驱动电机耐压试验中，电压从半值升至全值", STD, ["10s"]),
    ("动力电池绝缘电阻测量时，测量电压", STD, ["1.5"]),
    ("动力电池耐电压测试施加的交流电频率", STD, ["60Hz"]),
    ("PTC 加热器绝缘电阻测试时，测试电压", STD, ["220Vdc"]),
    ("如果整车厂坚持对电动压缩机绕组重复进行耐电压测试", STD, ["压缩机", "80%"]),
    ("该标准的验证试验中，从成品库任意抽取", NOTE, ["抽取", "台车辆"]),
    ("该标准的试验条件中，环境温度", STD, ["18℃"]),
    # ── 编制说明 ──
    ("该标准的技术指标数据主要参照哪份企业标准", NOTE, ["Q/NGA48-028-2020"]),
    ("专家意见中，关于适用海拔高度", NOTE, ["1400m", "1000m"]),
    ("专家评审意见中，试验电压应不大于表1", NOTE, ["80%"]),
    ("该标准编制说明的成文日期", NOTE, ["2023年5月8日"]),
    ("专家对「Ma 修改为 mA」", NOTE, ["Ma", "修改为"]),
    # ── 电动乘用车安全设计规范 ──
    ("B 级电压电路的电压范围", DESIGN, ["1500Vd.c"]),
    ("A 级电压电路的最大工作电压", DESIGN, ["60Vd.c"]),
    ("整车各直流电路与交流电路的绝缘电阻", DESIGN, ["500Ω/V"]),
    ("高压连接器在装配完好时的防护等级", DESIGN, ["IPXXD"]),
    ("高压连接器被分开后", DESIGN, ["0.2J"]),
    ("交流充电插座在未耦合状态下", DESIGN, ["IPXXB"]),
    ("直流充电座在充电插头被拔下后", DESIGN, ["1s"]),
    ("动力电池包的防护等级", DESIGN, ["IP67"]),
    ("乘员舱和行李舱外部的其它 B 级电压部件", DESIGN, ["IP67"]),
    ("乘员舱和行李舱内部的其它 B 级电压部件", DESIGN, ["IPX4"]),
    ("整车防水试验后，绝缘电阻测试", DESIGN, ["24 小时"]),
    ("主动放电需要在多长时间内完成", DESIGN, ["3s"]),
    ("被动放电的时间要求", DESIGN, ["3min"]),
    ("B 级电压系统 Y 电容的总能量限值", DESIGN, ["0.2J"]),
    ("低速提示音功能的工作车速范围", DESIGN, ["20km/h"]),
    ("救援信息卡至少应包含哪些信息", DESIGN, ["救援信息卡"]),
    ("高压维修开关应具备什么装置", DESIGN, ["高压互锁"]),
    # ── GB/T 18384.3 修改单宣贯 ──
    ("GB/T 18384.3-2015 修改单对 48V 车型豁免", CATARC, ["48V", "豁免"]),
    ("48V 系统为什么无法通过绝缘电阻试验", CATARC, ["负极接地"]),
    ("国际法规中把 48V 系统所属的电压级别", CATARC, ["B1 级"]),
    ("在最大工作电压下，直流电路与交流电路的绝缘电阻最小值", CATARC, ["500Ω/V"]),
    ("GB/T 18384.3-2015 修改单自什么日期", CATARC, ["2017", "7月1日"]),
    # ── 维修手册 ──
    ("2016 款 EV160/200 相比 E150EV", MANUAL, ["分散式", "集成式"]),
    ("EV160/200 的中控屏尺寸", MANUAL, ["10.4"]),
    ("空调空气滤清器安装在什么位置", MANUAL, ["空调空气滤清器"]),
    ("拆卸散热器下端出水管和补水管", MANUAL, ["鲤鱼钳"]),
    ("「高压系统电压校验错误」的故障代码", MANUAL, ["P103064"]),
    ("保险丝 FB133", MANUAL, ["FB133"]),
    ("EV160/200 是在哪个车型平台上开发", MANUAL, ["绅宝D20"]),
]


def _norm(s: str) -> str:
    return re.sub(r"[\s\u00a0]+", "", s or "")


def _read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


async def main(write: bool) -> int:
    from src.infra.milvus_client import get_milvus_client

    client = get_milvus_client()
    rows = await asyncio.to_thread(
        client.query, collection_name="alm_docs", filter="",
        output_fields=["doc_name", "page_number", "text"], limit=16384,
        consistency_level="Strong",
    )
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
        by_doc.setdefault(r["doc_name"], []).append(r)
    print(f"Milvus 共 {len(rows)} 块，{len(by_doc)} 份文档\n")

    path = DATASET
    items = await asyncio.to_thread(_read_rows, path)
    enriched, missed, skipped = 0, [], 0
    for item in items:
        spec = next((s for s in SPECS if item["question"].startswith(s[0])), None)
        if not spec:
            skipped += 1
            continue
        _, doc, terms = spec
        chunks = by_doc.get(doc) or []
        pages: list[int] = []
        for c in chunks:
            text = _norm(c["text"])
            if all(_norm(t) in text for t in terms):
                pg = c.get("page_number")
                if isinstance(pg, int) and pg > 0 and pg not in pages:
                    pages.append(pg)
        if not pages:
            missed.append((item["question"], doc, terms))
            continue
        pages.sort()
        if write:
            item["relevant_pages"] = ";".join(f"{doc}:p{p}" for p in pages)
            item["note"] = (item["note"].replace(" | 已填充页级标签", "")
                            + " | 已填充页级标签（答案值定位，待人工复核）").strip(" |")
        enriched += 1
        print(f"  ✓ p{pages} ← {item['question'][:40]}")

    print(f"\n填充 {enriched} 题；跳过 {skipped} 题（描述型/跨章节型，无精确答案值）")
    if missed:
        print(f"\n★ {len(missed)} 题未匹配到任何页（保持 doc 级，需人工补）：")
        for q, doc, terms in missed:
            print(f"  ✗ {q[:40]}  词={terms}  文档={doc}")

    if write:
        await asyncio.to_thread(_write_rows, path, items)
        print(f"\n已写回 {path}")
    else:
        print("\n（--list 模式，未写文件）")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="填充页级标签")
    ap.add_argument("--list", action="store_true", help="只看结果，不写文件")
    raise SystemExit(asyncio.run(main(not ap.parse_args().list)))
