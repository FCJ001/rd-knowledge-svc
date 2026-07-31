# ============================================================
# 离线构建脚本 — 读取 conf/nl2sql_meta.yaml 并写入三层存储：
#
#   1. PostgreSQL  → nl2sql_tables / nl2sql_columns / nl2sql_metrics / nl2sql_column_metrics
#   2. Milvus      → nl2sql_columns / nl2sql_metrics collection
#   3. Elasticsearch → nl2sql_values index（仅 sync: true 的列）
#
# 用法：
#   python scripts/build_nl2sql_meta.py                  # 增量更新（默认）
#   python scripts/build_nl2sql_meta.py --rebuild         # 全量重建（删光重建）
#   python scripts/build_nl2sql_meta.py --dry-run         # 只打印，不写入
# ============================================================

import asyncio
import json
import sys
from pathlib import Path

# 项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elasticsearch import AsyncElasticsearch
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import DataType, MilvusClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import get_settings
from src.nl2sql.conf.meta_config import MetaConfig
from src.nl2sql.models import Nl2sqlTable, Nl2sqlColumn, Nl2sqlMetric
from src.nl2sql.models.column_metric import ColumnMetricModel

# ── Milvus collection 定义 ──────────────────────────────────────

COLUMN_COLLECTION = "nl2sql_columns"
METRIC_COLLECTION = "nl2sql_metrics"
VECTOR_DIM = 1024  # text-embedding-v3 输出维度

# ── ES index 定义 ───────────────────────────────────────────────

VALUE_INDEX = "nl2sql_values"
VALUE_MAPPING = {
    "properties": {
        "id": {"type": "keyword"},
        "column_id": {"type": "keyword"},
        "column_name": {"type": "keyword"},
        "table_name": {"type": "keyword"},
        "value": {
            "type": "text",
            "analyzer": "standard",
        },
    }
}


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════

async def main(dry_run: bool = False, rebuild: bool = False):
    settings = get_settings()
    meta = MetaConfig.from_yaml(Path(__file__).parents[1] / "conf" / "nl2sql_meta.yaml")
    emb = DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )

    mode = "全量重建" if rebuild else "增量更新"
    print(f"模式: {mode}")
    print(f"  {len(meta.tables)} 张表, {sum(len(t.columns) for t in meta.tables)} 列, {len(meta.metrics)} 个指标\n")

    # ── PG ───────────────────────────────────────────────────
    pg_engine = create_async_engine(settings.DATABASE_URL)

    async with pg_engine.begin() as conn:
        # 建表（如不存在）
        await conn.run_sync(lambda c: Nl2sqlTable.metadata.create_all(c, checkfirst=True))
        await conn.run_sync(lambda c: ColumnMetricModel.metadata.create_all(c, checkfirst=True))

    pg_factory = async_sessionmaker(pg_engine, class_=AsyncSession, expire_on_commit=False)

    async with pg_factory() as db:
        if dry_run:
            _dry_print_pg(meta)
        elif rebuild:
            await _rebuild_pg(db, meta)
        else:
            await _upsert_pg(db, meta)

    # ── Milvus ───────────────────────────────────────────────
    milvus = MilvusClient(uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}")

    if dry_run:
        _dry_print_milvus(meta)
    elif rebuild:
        await _rebuild_milvus(milvus, meta, emb)
    else:
        await _upsert_milvus(milvus, meta, emb)

    # ── ES ──────────────────────────────────────────────────
    es = AsyncElasticsearch("http://localhost:9200")

    if dry_run:
        _dry_print_es(meta)
    elif rebuild:
        await _rebuild_es(es, meta)
    else:
        await _upsert_es(es, meta)

    await es.close()
    milvus.close()
    await pg_engine.dispose()
    print(f"\n{mode} 完成！")


# ════════════════════════════════════════════════════════════════
# PG — 全量重建
# ════════════════════════════════════════════════════════════════

async def _rebuild_pg(db: AsyncSession, meta: MetaConfig):
    """删除全部旧数据后重新写入"""
    await db.execute(delete(ColumnMetricModel))
    await db.execute(delete(Nl2sqlMetric))
    await db.execute(delete(Nl2sqlColumn))
    await db.execute(delete(Nl2sqlTable))

    table_map: dict[str, int] = {}
    for t in meta.tables:
        row = Nl2sqlTable(table_name=t.name, role=t.role, description=t.description)
        db.add(row)
        await db.flush()
        table_map[t.name] = row.id
        print(f"  PG 表: {t.name} (id={row.id})")

    for t in meta.tables:
        table_id = table_map[t.name]
        for c in t.columns:
            row = Nl2sqlColumn(
                table_id=table_id,
                column_name=c.name,
                column_type=c.type,
                role=c.role,
                description=c.description,
                aliases=c.alias,
            )
            db.add(row)
        print(f"  PG 列: {t.name} → {len(t.columns)} 列")

    metric_id_map: dict[str, int] = {}
    for m in meta.metrics:
        row = Nl2sqlMetric(
            metric_name=m.name,
            description=m.description,
            relevant_columns=m.relevant_columns,
            aliases=m.alias,
        )
        db.add(row)
        await db.flush()
        metric_id_map[m.name] = row.id
        print(f"  PG 指标: {m.name} (id={row.id})")

    for m in meta.metrics:
        for col_ref in m.relevant_columns:
            metric_id = str(metric_id_map[m.name])
            row = ColumnMetricModel(column_id=col_ref, metric_id=metric_id)
            db.add(row)
        print(f"  PG 列-指标关联: {m.name} → {len(m.relevant_columns)} 列")

    await db.commit()


# ════════════════════════════════════════════════════════════════
# PG — 增量更新
# ════════════════════════════════════════════════════════════════

async def _upsert_pg(db: AsyncSession, meta: MetaConfig):
    """按唯一键对比 YAML 与 DB，增/改/删"""
    # ── 表 ──
    existing_tables = {
        t.table_name: t
        for t in (await db.execute(select(Nl2sqlTable))).scalars()
    }
    yaml_table_names = {t.name for t in meta.tables}

    tables_added, tables_updated = 0, 0
    for t in meta.tables:
        if t.name in existing_tables:
            ex = existing_tables[t.name]
            if ex.role != t.role or ex.description != t.description:
                ex.role = t.role
                ex.description = t.description
                tables_updated += 1
        else:
            row = Nl2sqlTable(table_name=t.name, role=t.role, description=t.description)
            db.add(row)
            await db.flush()
            existing_tables[t.name] = row
            tables_added += 1

    # 删除 YAML 中已不存在的表（先删关联列和列-指标关联）
    tables_deleted = 0
    for name, table in existing_tables.items():
        if name not in yaml_table_names:
            # 删除该表下所有列的 ColumnMetric 关联
            col_ids = (
                await db.execute(
                    select(Nl2sqlColumn.column_name).where(Nl2sqlColumn.table_id == table.id)
                )
            ).scalars().all()
            for cname in col_ids:
                col_ref = f"{name}.{cname}"
                await db.execute(
                    delete(ColumnMetricModel).where(ColumnMetricModel.column_id == col_ref)
                )
            await db.execute(delete(Nl2sqlColumn).where(Nl2sqlColumn.table_id == table.id))
            await db.execute(delete(Nl2sqlTable).where(Nl2sqlTable.id == table.id))
            tables_deleted += 1

    await db.flush()

    # 重建 table_map（只含 YAML 中的表）
    table_map = {
        name: existing_tables[name].id
        for name in yaml_table_names
        if name in existing_tables
    }
    print(f"  PG 表: +{tables_added} ~{tables_updated} -{tables_deleted}")

    # ── 列 ──
    existing_columns: dict[tuple[int, str], Nl2sqlColumn] = {}
    for t_id in table_map.values():
        for c in (await db.execute(select(Nl2sqlColumn).where(Nl2sqlColumn.table_id == t_id))).scalars():
            existing_columns[(t_id, c.column_name)] = c

    yaml_col_keys: set[tuple[int, str]] = set()
    cols_added, cols_updated = 0, 0
    for t in meta.tables:
        table_id = table_map.get(t.name)
        if table_id is None:
            continue
        for c in t.columns:
            key = (table_id, c.name)
            yaml_col_keys.add(key)
            if key in existing_columns:
                ex = existing_columns[key]
                if (
                    ex.column_type != c.type
                    or ex.role != c.role
                    or ex.description != c.description
                    or ex.aliases != c.alias
                ):
                    ex.column_type = c.type
                    ex.role = c.role
                    ex.description = c.description
                    ex.aliases = c.alias
                    cols_updated += 1
            else:
                row = Nl2sqlColumn(
                    table_id=table_id,
                    column_name=c.name,
                    column_type=c.type,
                    role=c.role,
                    description=c.description,
                    aliases=c.alias,
                )
                db.add(row)
                cols_added += 1

    # 删除 YAML 中已不存在的列
    cols_deleted = 0
    for key, col in existing_columns.items():
        if key not in yaml_col_keys:
            # 同时删除 ColumnMetric 关联
            table_name = next((n for n, tid in table_map.items() if tid == key[0]), None)
            if table_name:
                col_ref = f"{table_name}.{key[1]}"
                await db.execute(
                    delete(ColumnMetricModel).where(ColumnMetricModel.column_id == col_ref)
                )
            await db.execute(delete(Nl2sqlColumn).where(Nl2sqlColumn.id == col.id))
            cols_deleted += 1

    await db.flush()
    print(f"  PG 列: +{cols_added} ~{cols_updated} -{cols_deleted}")

    # ── 指标 ──
    existing_metrics = {
        m.metric_name: m
        for m in (await db.execute(select(Nl2sqlMetric))).scalars()
    }
    yaml_metric_names = {m.name for m in meta.metrics}

    metrics_added, metrics_updated = 0, 0
    for m in meta.metrics:
        if m.name in existing_metrics:
            ex = existing_metrics[m.name]
            if (
                ex.description != m.description
                or ex.relevant_columns != m.relevant_columns
                or ex.aliases != m.alias
            ):
                ex.description = m.description
                ex.relevant_columns = m.relevant_columns
                ex.aliases = m.alias
                metrics_updated += 1
        else:
            row = Nl2sqlMetric(
                metric_name=m.name,
                description=m.description,
                relevant_columns=m.relevant_columns,
                aliases=m.alias,
            )
            db.add(row)
            await db.flush()
            existing_metrics[m.name] = row
            metrics_added += 1

    # 删除 YAML 中已不存在的指标（及其 ColumnMetric 关联）
    metrics_deleted = 0
    for name, metric in existing_metrics.items():
        if name not in yaml_metric_names:
            await db.execute(
                delete(ColumnMetricModel).where(
                    ColumnMetricModel.metric_id == str(metric.id)
                )
            )
            await db.execute(delete(Nl2sqlMetric).where(Nl2sqlMetric.id == metric.id))
            metrics_deleted += 1

    await db.flush()

    # 重建 metric_id_map
    metric_id_map = {
        name: existing_metrics[name].id
        for name in yaml_metric_names
        if name in existing_metrics
    }
    print(f"  PG 指标: +{metrics_added} ~{metrics_updated} -{metrics_deleted}")

    # ── 列-指标关联 ──
    cm_added, cm_deleted = 0, 0
    for m in meta.metrics:
        metric_id = str(metric_id_map.get(m.name, ""))
        if not metric_id:
            continue

        # 查询该指标现有的关联
        existing_cm = {
            row.column_id
            for row in (
                await db.execute(
                    select(ColumnMetricModel).where(
                        ColumnMetricModel.metric_id == metric_id
                    )
                )
            ).scalars()
        }
        yaml_cm = set(m.relevant_columns)

        # 删除不再关联的
        for col_ref in existing_cm - yaml_cm:
            await db.execute(
                delete(ColumnMetricModel).where(
                    ColumnMetricModel.column_id == col_ref,
                    ColumnMetricModel.metric_id == metric_id,
                )
            )
            cm_deleted += 1

        # 新增关联
        for col_ref in yaml_cm - existing_cm:
            row = ColumnMetricModel(column_id=col_ref, metric_id=metric_id)
            db.add(row)
            cm_added += 1

    print(f"  PG 列-指标关联: +{cm_added} -{cm_deleted}")

    await db.commit()


# ════════════════════════════════════════════════════════════════
# Milvus — 全量重建
# ════════════════════════════════════════════════════════════════

async def _rebuild_milvus(milvus: MilvusClient, meta: MetaConfig, emb: DashScopeEmbeddings):
    """drop collection 后重建 + 全量写入"""
    _ensure_collection_dropped(milvus, COLUMN_COLLECTION)
    _create_column_collection(milvus)
    col_data = _build_column_vectors(meta, emb)
    if col_data:
        milvus.insert(collection_name=COLUMN_COLLECTION, data=col_data)
        milvus.flush(collection_name=COLUMN_COLLECTION)
        print(f"  Milvus columns: {len(col_data)} 条向量")

    _ensure_collection_dropped(milvus, METRIC_COLLECTION)
    _create_metric_collection(milvus)
    metric_data = _build_metric_vectors(meta, emb)
    if metric_data:
        milvus.insert(collection_name=METRIC_COLLECTION, data=metric_data)
        milvus.flush(collection_name=METRIC_COLLECTION)
        print(f"  Milvus metrics: {len(metric_data)} 条向量")


# ════════════════════════════════════════════════════════════════
# Milvus — 增量更新
# ════════════════════════════════════════════════════════════════

async def _upsert_milvus(milvus: MilvusClient, meta: MetaConfig, emb: DashScopeEmbeddings):
    """collection 不存在则创建，存在则 upsert + 删除 YAML 中不存在的实体"""

    # ── columns ──
    if not milvus.has_collection(COLUMN_COLLECTION):
        _create_column_collection(milvus)
    elif not _collection_has_varchar_pk(milvus, COLUMN_COLLECTION):
        print("  [warn] Milvus columns collection 使用旧 schema（auto_id），降级为重建")
        milvus.drop_collection(COLUMN_COLLECTION)
        _create_column_collection(milvus)

    col_data = _build_column_vectors(meta, emb)
    if col_data:
        milvus.upsert(collection_name=COLUMN_COLLECTION, data=col_data)
        milvus.flush(collection_name=COLUMN_COLLECTION)

        # 删除 YAML 中不存在的实体
        yaml_col_ids = [f"{t.name}.{c.name}" for t in meta.tables for c in t.columns]
        await _delete_stale_milvus(milvus, COLUMN_COLLECTION, "column_id", yaml_col_ids)
        print(f"  Milvus columns: upsert {len(col_data)} 条")

    # ── metrics ──
    if not milvus.has_collection(METRIC_COLLECTION):
        _create_metric_collection(milvus)
    elif not _collection_has_varchar_pk(milvus, METRIC_COLLECTION):
        print("  [warn] Milvus metrics collection 使用旧 schema（auto_id），降级为重建")
        milvus.drop_collection(METRIC_COLLECTION)
        _create_metric_collection(milvus)

    metric_data = _build_metric_vectors(meta, emb)
    if metric_data:
        milvus.upsert(collection_name=METRIC_COLLECTION, data=metric_data)
        milvus.flush(collection_name=METRIC_COLLECTION)

        yaml_metric_ids = [m.name for m in meta.metrics]
        await _delete_stale_milvus(milvus, METRIC_COLLECTION, "metric_id", yaml_metric_ids)
        print(f"  Milvus metrics: upsert {len(metric_data)} 条")


# ════════════════════════════════════════════════════════════════
# ES — 全量重建
# ════════════════════════════════════════════════════════════════

async def _rebuild_es(es: AsyncElasticsearch, meta: MetaConfig):
    """删除 index 后重建 + 全量写入"""
    if await es.indices.exists(index=VALUE_INDEX):
        await es.indices.delete(index=VALUE_INDEX)
    await es.indices.create(index=VALUE_INDEX, mappings=VALUE_MAPPING)
    count = await _index_yaml_values(es, meta)
    await es.indices.refresh(index=VALUE_INDEX)
    print(f"  ES values: {count} 条枚举值")


# ════════════════════════════════════════════════════════════════
# ES — 增量更新
# ════════════════════════════════════════════════════════════════

async def _upsert_es(es: AsyncElasticsearch, meta: MetaConfig):
    """index 不存在则创建，存在则按 _id upsert + 删除 YAML 中不存在的 doc"""

    if not await es.indices.exists(index=VALUE_INDEX):
        await es.indices.create(index=VALUE_INDEX, mappings=VALUE_MAPPING)

    # 扫描现有 doc ID
    existing_ids: set[str] = set()
    try:
        result = await es.search(
            index=VALUE_INDEX,
            body={"query": {"match_all": {}}, "_source": False, "size": 10000},
            scroll="1m",
        )
        while True:
            hits = result["hits"]["hits"]
            if not hits:
                break
            for hit in hits:
                existing_ids.add(hit["_id"])
            scroll_id = result.get("_scroll_id")
            if not scroll_id:
                break
            result = await es.scroll(scroll_id=scroll_id, scroll="1m")
    except Exception:
        pass  # index 刚创建，还没有数据

    # 写入 YAML 数据
    yaml_ids = set()
    count = 0
    for t in meta.tables:
        for c in t.columns:
            if not c.sync or not c.alias:
                continue
            for alias in c.alias:
                doc_id = f"{t.name}.{c.name}.{alias}"
                yaml_ids.add(doc_id)
                doc = {
                    "id": doc_id,
                    "column_id": f"{t.name}.{c.name}",
                    "column_name": c.name,
                    "table_name": t.name,
                    "value": alias,
                }
                await es.index(index=VALUE_INDEX, id=doc_id, document=doc)
                count += 1

    # 删除 YAML 中不存在的 doc
    stale = existing_ids - yaml_ids
    for doc_id in stale:
        try:
            await es.delete(index=VALUE_INDEX, id=doc_id)
        except Exception:
            pass

    await es.indices.refresh(index=VALUE_INDEX)
    print(f"  ES values: +{count - len(existing_ids & yaml_ids)} ~{len(existing_ids & yaml_ids)} -{len(stale)} (总计 {count})")


# ════════════════════════════════════════════════════════════════
# 工具函数 — Milvus
# ════════════════════════════════════════════════════════════════

def _ensure_collection_dropped(milvus: MilvusClient, name: str):
    if milvus.has_collection(name):
        milvus.drop_collection(name)


def _create_column_collection(milvus: MilvusClient):
    """column_id (VARCHAR) 作为主键，支持 upsert"""
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("column_id", DataType.VARCHAR, max_length=256, is_primary=True)
    schema.add_field("column_name", DataType.VARCHAR, max_length=256)
    schema.add_field("column_type", DataType.VARCHAR, max_length=64)
    schema.add_field("role", DataType.VARCHAR, max_length=32)
    schema.add_field("description", DataType.VARCHAR, max_length=2048)
    schema.add_field("aliases", DataType.JSON)
    schema.add_field("table_name", DataType.VARCHAR, max_length=128)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type="COSINE",
        index_type="IVF_FLAT",
        params={"nlist": 128},
    )
    milvus.create_collection(COLUMN_COLLECTION, schema=schema, index_params=index_params)


def _create_metric_collection(milvus: MilvusClient):
    """metric_id (VARCHAR) 作为主键，支持 upsert"""
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("metric_id", DataType.VARCHAR, max_length=128, is_primary=True)
    schema.add_field("metric_name", DataType.VARCHAR, max_length=256)
    schema.add_field("description", DataType.VARCHAR, max_length=2048)
    schema.add_field("relevant_columns", DataType.JSON)
    schema.add_field("aliases", DataType.JSON)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type="COSINE",
        index_type="IVF_FLAT",
        params={"nlist": 32},
    )
    milvus.create_collection(METRIC_COLLECTION, schema=schema, index_params=index_params)


def _collection_has_varchar_pk(milvus: MilvusClient, name: str) -> bool:
    """检查 collection 是否使用 VARCHAR 主键（新 schema）"""
    try:
        info = milvus.describe_collection(name)
        for field in info.get("fields", []):
            if field.get("is_primary") and field.get("type") in ("VarChar", "VARCHAR"):
                return True
    except Exception:
        pass
    return False


async def _delete_stale_milvus(
    milvus: MilvusClient,
    collection: str,
    pk_field: str,
    yaml_ids: list[str],
):
    """删除 collection 中不在 yaml_ids 里的实体"""
    if not yaml_ids:
        return
    # 分页查询所有现有 ID
    existing_ids: set[str] = set()
    offset = 0
    batch_size = 1000
    while True:
        results = milvus.query(
            collection_name=collection,
            filter="",
            output_fields=[pk_field],
            offset=offset,
            limit=batch_size,
        )
        if not results:
            break
        for r in results:
            existing_ids.add(r[pk_field])
        if len(results) < batch_size:
            break
        offset += batch_size

    stale = existing_ids - set(yaml_ids)
    if stale:
        # Milvus delete by expression 限制：IN 列表最大 1000 个
        stale_list = list(stale)
        for i in range(0, len(stale_list), 1000):
            chunk = stale_list[i : i + 1000]
            expr = f'{pk_field} in {json.dumps(chunk)}'
            milvus.delete(collection_name=collection, filter=expr)
        print(f"  Milvus {collection}: 删除 {len(stale)} 条过期实体")


# ════════════════════════════════════════════════════════════════
# 工具函数 — 向量构建
# ════════════════════════════════════════════════════════════════

def _build_column_vectors(meta: MetaConfig, emb: DashScopeEmbeddings) -> list[dict]:
    """为所有列生成向量"""
    data = []
    for t in meta.tables:
        for c in t.columns:
            text_parts = [f"列名: {c.name}", f"类型: {c.type}", f"所属表: {t.name}"]
            if c.description:
                text_parts.append(f"说明: {c.description}")
            if c.alias:
                text_parts.append(f"别名: {', '.join(c.alias)}")
            text = "; ".join(text_parts)

            vector = emb.embed_query(text)
            data.append({
                "column_id": f"{t.name}.{c.name}",
                "column_name": c.name,
                "column_type": c.type,
                "role": c.role,
                "description": c.description,
                "aliases": c.alias,
                "table_name": t.name,
                "vector": vector,
            })
    return data


def _build_metric_vectors(meta: MetaConfig, emb: DashScopeEmbeddings) -> list[dict]:
    """为所有指标生成向量"""
    data = []
    for m in meta.metrics:
        text_parts = [f"指标: {m.name}"]
        if m.description:
            text_parts.append(f"说明: {m.description}")
        if m.alias:
            text_parts.append(f"别名: {', '.join(m.alias)}")
        text = "; ".join(text_parts)

        vector = emb.embed_query(text)
        data.append({
            "metric_id": m.name,
            "metric_name": m.name,
            "description": m.description,
            "relevant_columns": m.relevant_columns,
            "aliases": m.alias,
            "vector": vector,
        })
    return data


# ════════════════════════════════════════════════════════════════
# 工具函数 — ES 写入
# ════════════════════════════════════════════════════════════════

async def _index_yaml_values(es: AsyncElasticsearch, meta: MetaConfig) -> int:
    """将 YAML 中 sync:true 的列的 alias 写入 ES"""
    count = 0
    for t in meta.tables:
        for c in t.columns:
            if not c.sync or not c.alias:
                continue
            for alias in c.alias:
                doc = {
                    "id": f"{t.name}.{c.name}.{alias}",
                    "column_id": f"{t.name}.{c.name}",
                    "column_name": c.name,
                    "table_name": t.name,
                    "value": alias,
                }
                await es.index(index=VALUE_INDEX, document=doc)
                count += 1
    return count


# ════════════════════════════════════════════════════════════════
# Dry-run 打印
# ════════════════════════════════════════════════════════════════

def _dry_print_pg(meta: MetaConfig):
    print("[dry-run] === PG ===")
    for t in meta.tables:
        print(f"  {t.role:6s} {t.name:30s} ({len(t.columns)} 列)  {t.description}")
        for c in t.columns:
            tags = []
            if c.sync:
                tags.append("ES")
            if c.alias:
                tags.append(f"alias={c.alias}")
            print(f"    {c.role:12s} {c.name:25s} {c.type:15s} {c.description:20s} {', '.join(tags)}")
    for m in meta.metrics:
        print(f"  metric {m.name:30s} → {m.relevant_columns}  alias={m.alias}")


def _dry_print_milvus(meta: MetaConfig):
    total_cols = sum(len(t.columns) for t in meta.tables)
    print(f"[dry-run] === Milvus ===")
    print(f"  {COLUMN_COLLECTION}: {total_cols} 条向量 (dim={VECTOR_DIM}, COSINE)")
    print(f"  {METRIC_COLLECTION}: {len(meta.metrics)} 条向量 (dim={VECTOR_DIM}, COSINE)")


def _dry_print_es(meta: MetaConfig):
    sync_cols = [(t.name, c.name, c.alias) for t in meta.tables for c in t.columns if c.sync]
    total = sum(len(aliases) for _, _, aliases in sync_cols)
    print(f"[dry-run] === ES ===")
    print(f"  {VALUE_INDEX}: {total} 条枚举值 ({len(sync_cols)} 个 sync 列)")


# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    rebuild = "--rebuild" in sys.argv
    asyncio.run(main(dry_run=dry, rebuild=rebuild))
