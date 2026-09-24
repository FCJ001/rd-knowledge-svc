#!/usr/bin/env python3
# ============================================================
# 存量数据 ACL 回填（非破坏性）
#
#   python scripts/backfill_acl.py --list                          # 看现状，不动数据
#   python scripts/backfill_acl.py --dry-run --roles engineer,...  # 预演
#   python scripts/backfill_acl.py --roles engineer,business,aftersales,customer
#   python scripts/backfill_acl.py --roles engineer --doc 维修手册   # 只回填匹配文档
#
# 背景：
#   ACL 字段（acl_roles，ARRAY<VARCHAR>）是后加的。pipeline 实例化时会给
#   存量 collection 自动补字段（add_collection_field，非破坏性），但存量行的
#   acl_roles 为 null —— ACL 谓词（array_contains_any）匹配不到 null，
#   即存量文档对非 admin 不可见（fail-closed，宁可不可见不可泄露）。
#   本脚本把存量行的可见角色显式赋上，是"存量可见性"的唯一迁移通道。
#
# ★ upsert 是"整行删了重写"：必须把原行的全部标量字段 + embedding 向量
#   带回去；sparse_embedding 由 BM25 Function 从 text 重新生成，不能也不需要传。
# ============================================================

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.knowledge.acl import ACL_FIELD, parse_acl_roles

COLLECTION = "alm_docs"

# upsert 需要带回的全部字段（sparse_embedding 除外：BM25 Function 生成）
COPY_FIELDS = [
    "id", "doc_id", "doc_name", "doc_type", "category",
    "business_line", "model_code", ACL_FIELD,
    "page_number", "chunk_index", "parent_text", "text", "image_urls",
    "embedding",
]


def load_rows(client) -> list[dict]:
    rows = client.query(COLLECTION, filter="", output_fields=COPY_FIELDS, limit=16384)
    if not rows:
        print("collection 为空，无需回填")
        return []
    return rows


def ensure_acl_field(client) -> None:
    """补 ACL 字段（与 pipeline._ensure_acl_field 同逻辑；脚本独立跑时 pipeline
    不一定实例化过）。"""
    from pymilvus import DataType

    desc = client.describe_collection(COLLECTION)
    fields = desc.get("fields") or (desc.get("schema") or {}).get("fields") or []
    if any(f.get("name") == ACL_FIELD for f in fields):
        return
    client.add_collection_field(
        COLLECTION,
        field_name=ACL_FIELD,
        data_type=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=16,
        max_length=32,
        nullable=True,
    )
    print(f"[字段] 已为 '{COLLECTION}' 补上 {ACL_FIELD} 字段")


def main() -> int:
    ap = argparse.ArgumentParser(description="存量数据 ACL 回填（非破坏性）")
    ap.add_argument("--roles", default=None,
                    help="回填的可见角色，逗号分隔（如 engineer,business）；admin 不可指定（恒可见）")
    ap.add_argument("--doc", default=None, help="只回填 doc_name 含该子串的文档")
    ap.add_argument("--limit", type=int, default=0, help="最多回填 N 行（0=不限），用于小步验证")
    ap.add_argument("--list", action="store_true", help="只看各文档回填状态，不动数据")
    ap.add_argument("--dry-run", action="store_true", help="预演：不执行 upsert")
    args = ap.parse_args()

    if not args.list:
        if not args.roles:
            print("[失败] 必须显式指定 --roles（回填是授权决策，不允许默认值）")
            return 1
        try:
            roles = parse_acl_roles(args.roles)
        except ValueError as e:
            print(f"[失败] {e}")
            return 1
        if not roles:
            print("[失败] --roles 为空等于把存量全部收成仅 admin；确认的话请逐文档处理")
            return 1

    from src.infra.milvus_client import get_milvus_client

    client = get_milvus_client()
    ensure_acl_field(client)
    rows = load_rows(client)

    # 按文档聚合看现状
    by_doc: dict[str, dict] = {}
    for r in rows:
        d = by_doc.setdefault(r["doc_name"], {"total": 0, "unset": 0})
        d["total"] += 1
        if not r.get(ACL_FIELD):
            d["unset"] += 1

    print("=" * 76)
    print(f"{'【现状】' if args.list else '回填计划'}  collection={COLLECTION} 共 {len(rows)} 块 / {len(by_doc)} 份文档")
    print("=" * 76)
    for name, d in sorted(by_doc.items()):
        mark = "✓" if d["unset"] == 0 else ("△ 部分未回填" if d["unset"] < d["total"] else "★ 未回填（非 admin 不可见）")
        print(f"  {mark} {name}: {d['total'] - d['unset']}/{d['total']} 已有 ACL")
    if args.list:
        return 0

    targets = [
        r for r in rows
        if not r.get(ACL_FIELD)
        and (not args.doc or args.doc in r["doc_name"])
    ]
    if args.limit:
        targets = targets[: args.limit]
    print(f"\n待回填 {len(targets)} 块 → acl_roles={roles}"
          + (f"（doc 过滤: {args.doc}）" if args.doc else "")
          + ("【dry-run，不写入】" if args.dry_run else ""))
    if not targets:
        return 0

    t0 = time.time()
    done = 0
    batch_size = 50
    for i in range(0, len(targets), batch_size):
        batch = targets[i : i + batch_size]
        if not args.dry_run:
            client.upsert(COLLECTION, data=[{**r, ACL_FIELD: roles} for r in batch])
        done += len(batch)
        print(f"  进度 {done}/{len(targets)}（{time.time() - t0:.1f}s）")

    if args.dry_run:
        print("\n[dry-run] 未写入。去掉 --dry-run 执行。")
        return 0

    # 回读核对（Strong 一致性：刚 upsert 的行立即可查）
    time.sleep(1)
    after = client.query(COLLECTION, filter="", output_fields=["doc_name", ACL_FIELD],
                         limit=16384, consistency_level="Strong")
    unset = sum(1 for r in after if not r.get(ACL_FIELD))
    matched = sum(1 for r in after if r.get(ACL_FIELD) and set(r[ACL_FIELD]) == set(roles))
    print(f"\n核对: {len(after)} 块中 {matched} 块 ACL={roles}，仍无 ACL 的 {unset} 块"
          f"（其余为按 --doc 过滤跳过的，或 ACL 不同的文档）")
    return 0 if unset == 0 or args.doc else 1


if __name__ == "__main__":
    raise SystemExit(main())
