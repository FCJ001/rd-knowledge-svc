# ============================================================
# 评估结果查询 API
#
# GET /api/v1/eval/scores           评测结果列表（在线 + TruLens 离线）
# GET /api/v1/eval/scores/summary   汇总统计
# ============================================================

from fastapi import APIRouter, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from src.core.base_schema import ResponseSchema
from src.core.config import get_settings

router = APIRouter(prefix="/api/v1/eval", tags=["评估结果"])

_eval_engine = None


def _get_eval_engine():
    global _eval_engine
    if _eval_engine is None:
        settings = get_settings()
        _eval_engine = create_async_engine(
            f"postgresql+asyncpg://{settings.DB_USER}:{settings.DB_PASSWORD}"
            f"@{settings.DB_HOST}:{settings.DB_PORT}/trulens_eval"
        )
    return _eval_engine


@router.get("/scores", response_model=ResponseSchema[dict])
async def get_eval_scores(
    limit: int = Query(default=50, ge=1, le=200, description="返回条数"),
    eval_type: str = Query(default="", description="nl2sql / knowledge / offline，空=全部"),
):
    """查询评测结果列表（在线 + TruLens 离线记录）"""
    engine = _get_eval_engine()
    items: list[dict] = []

    async with engine.begin() as conn:
        # ── 在线评测结果（NL2SQL + 知识检索）──
        if not eval_type or eval_type in ("online", "nl2sql", "knowledge"):
            try:
                type_filter = ""
                params = {"limit": limit}
                if eval_type in ("nl2sql", "knowledge"):
                    type_filter = "WHERE eval_type = :etype"
                    params["etype"] = eval_type

                online_rows = await conn.execute(
                    text(
                        f"SELECT id, COALESCE(eval_type, 'nl2sql') AS eval_type, "
                        "question, COALESCE(answer, '') AS answer, "
                        "sql, summary, error, "
                        "score_sql_valid, score_relevance, score_reason, "
                        "COALESCE(score_has_data, 0) AS score_has_data, "
                        "COALESCE(score_context_relevance, 0) AS score_context_relevance, "
                        "COALESCE(score_context_relevance_reason, '') AS score_context_relevance_reason, "
                        "COALESCE(score_groundedness, 0) AS score_groundedness, "
                        "COALESCE(score_groundedness_reason, '') AS score_groundedness_reason, "
                        "COALESCE(token_input, 0) AS token_input, "
                        "COALESCE(token_output, 0) AS token_output, "
                        "COALESCE(token_calls, 0) AS token_calls, "
                        "COALESCE(token_cost_usd, 0) AS token_cost_usd, "
                        "created_at AT TIME ZONE 'Asia/Shanghai' AS created_at "
                        "FROM online_scores "
                        f"{type_filter} "
                        "ORDER BY id DESC LIMIT :limit"
                    ),
                    params,
                )
                for row in online_rows:
                    items.append({
                        "id": f"online-{row.id}",
                        "eval_type": f"online-{row.eval_type}",
                        "question": row.question,
                        "answer": (row.answer or "")[:200],
                        "sql": (row.sql or "")[:200],
                        "summary": (row.summary or "")[:200],
                        "error": (row.error or "")[:200],
                        "score_sql_valid": row.score_sql_valid,
                        "score_has_data": row.score_has_data,
                        "score_relevance": row.score_relevance,
                        "score_reason": (row.score_reason or "")[:200],
                        "score_context_relevance": row.score_context_relevance,
                        "score_context_relevance_reason": (row.score_context_relevance_reason or "")[:200],
                        "score_groundedness": row.score_groundedness,
                        "score_groundedness_reason": (row.score_groundedness_reason or "")[:200],
                        "token_input": row.token_input,
                        "token_output": row.token_output,
                        "token_calls": row.token_calls,
                        "token_cost_usd": row.token_cost_usd,
                        "created_at": row.created_at.isoformat() if row.created_at else "",
                    })
            except Exception:
                pass  # online_scores 表可能还不存在

        # ── TruLens 离线评测记录 ──
        if not eval_type or eval_type == "offline":
            try:
                tru_rows = await conn.execute(
                    text(
                        "SELECT r.record_id, r.input, r.output, r.ts, "
                        "a.app_name, a.app_version "
                        "FROM records r "
                        "LEFT JOIN apps a ON r.app_id = a.app_id "
                        "ORDER BY r.ts DESC LIMIT :limit"
                    ),
                    {"limit": limit},
                )
                tru_items = []
                record_ids = []
                for row in tru_rows:
                    tru_items.append({
                        "id": row.record_id,
                        "question": (row.input or "")[:200],
                        "output": (row.output or "")[:200],
                        "app_name": row.app_name or "",
                        "app_version": row.app_version or "",
                        "eval_type": "offline",
                        "created_at": row.ts.isoformat() if row.ts else "",
                        "feedbacks": {},
                    })
                    record_ids.append(row.record_id)

                # 补充 feedback 分数
                if record_ids:
                    fb_rows = await conn.execute(
                        text(
                            "SELECT f.record_id, fd.name, f.result "
                            "FROM feedbacks f "
                            "JOIN feedback_defs fd ON f.feedback_definition_id = fd.feedback_definition_id "
                            "WHERE f.record_id = ANY(:ids) AND f.status = 'done'"
                        ),
                        {"ids": record_ids},
                    )
                    feedbacks = {}
                    for fb in fb_rows:
                        feedbacks.setdefault(fb.record_id, {})[fb.name] = round(fb.result, 3) if fb.result else 0
                    for item in tru_items:
                        item["feedbacks"] = feedbacks.get(item["id"], {})

                items.extend(tru_items)
            except Exception:
                pass

    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return ResponseSchema(data={"items": items[:limit], "total": len(items)})


@router.get("/scores/summary", response_model=ResponseSchema[dict])
async def get_eval_summary():
    """评测汇总统计"""
    engine = _get_eval_engine()
    async with engine.begin() as conn:
        data: dict = {"online": {}, "offline": {}}

        # ── 在线统计（按类型拆分）──
        try:
            row = await conn.execute(
                text(
                    "SELECT "
                    "  COUNT(*) AS total, "
                    "  COALESCE(AVG(score_sql_valid) FILTER (WHERE COALESCE(eval_type,'nl2sql') = 'nl2sql'), 0) AS avg_sql_valid, "
                    "  COALESCE(AVG(score_has_data) FILTER (WHERE COALESCE(eval_type,'nl2sql') = 'nl2sql'), 0) AS avg_has_data, "
                    "  COALESCE(AVG(score_relevance) FILTER (WHERE COALESCE(eval_type,'nl2sql') = 'nl2sql'), 0) AS avg_nl2sql_relevance, "
                    "  COUNT(*) FILTER (WHERE COALESCE(eval_type,'nl2sql') = 'nl2sql') AS nl2sql_total, "
                    "  COALESCE(AVG(score_relevance) FILTER (WHERE COALESCE(eval_type,'nl2sql') = 'knowledge'), 0) AS avg_knowledge_relevance, "
                    "  COALESCE(AVG(score_context_relevance) FILTER (WHERE COALESCE(eval_type,'nl2sql') = 'knowledge'), 0) AS avg_context_relevance, "
                    "  COALESCE(AVG(score_groundedness) FILTER (WHERE COALESCE(eval_type,'nl2sql') = 'knowledge'), 0) AS avg_groundedness, "
                    "  COUNT(*) FILTER (WHERE COALESCE(eval_type,'nl2sql') = 'knowledge') AS knowledge_total "
                    "FROM online_scores"
                )
            )
            r = row.first()
            data["online"] = {
                "total": r.total,
                "nl2sql": {
                    "total": r.nl2sql_total,
                    "avg_sql_valid": round(r.avg_sql_valid, 2),
                    "avg_has_data": round(r.avg_has_data, 2),
                    "avg_relevance": round(r.avg_nl2sql_relevance, 2),
                },
                "knowledge": {
                    "total": r.knowledge_total,
                    "avg_answer_relevance": round(r.avg_knowledge_relevance, 2),
                    "avg_context_relevance": round(r.avg_context_relevance, 2),
                    "avg_groundedness": round(r.avg_groundedness, 2),
                },
            }
        except Exception:
            pass

        # ── TruLens 离线统计 ──
        try:
            total_row = await conn.execute(text("SELECT COUNT(*) FROM records"))
            total_count = total_row.scalar()

            fb_rows = await conn.execute(
                text(
                    "SELECT fd.name, AVG(f.result) AS avg_score, COUNT(*) AS cnt "
                    "FROM feedbacks f "
                    "JOIN feedback_defs fd ON f.feedback_definition_id = fd.feedback_definition_id "
                    "WHERE f.status = 'done' "
                    "GROUP BY fd.name"
                )
            )
            by_metric = []
            for r in fb_rows:
                by_metric.append({
                    "metric": r.name,
                    "avg_score": round(r.avg_score, 3) if r.avg_score else 0,
                    "count": r.cnt,
                })

            data["offline"] = {
                "total_records": total_count,
                "by_metric": by_metric,
            }
        except Exception:
            pass

        return ResponseSchema(data=data)
