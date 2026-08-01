# ============================================================
# 在线异步评估器 — API 返回后不阻塞用户，后台调 LLM 打分并写入 trulens_eval
#
# 用法：
#   evaluator = get_async_evaluator()
#   evaluator.evaluate(question, sql, data, summary, error)       # NL2SQL
#   evaluator.evaluate_knowledge(question, answer, contexts)        # 知识检索（RAG Triad）
#   # ← 不 await！fire-and-forget
#
# 开关：TRULENS_ENABLED=false → 完全不跑，零开销
# ============================================================

from __future__ import annotations

import asyncio
import json
import random
import re

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import create_async_engine

from src.core.config import get_settings
from src.core.logger import logger
from src.rag.evaluation.nl2sql_metrics import score_sql_valid


# ── RAG Triad 中文 LLM-as-Judge prompts ────────────────────────────
# 使用 0.0-1.0 连续评分，要求 LLM 使用全范围，避免中庸

ANSWER_RELEVANCE_PROMPT = """你是知识检索质量评审专家。判断系统答案是否准确回答了用户问题。
请使用 0.0-1.0 的连续评分，**不要都打同一分数**。

## 用户问题
{question}

## 系统答案
{answer}

## 详细评判标准
- 0.9-1.0：答案精准、全面，所有关键点都被覆盖，信息完全正确，排版清晰
- 0.7-0.8：答案基本正确，主要问题被回答，但个别要点缺失或细节不足
- 0.5-0.6：答案部分正确，覆盖了部分问题，但遗漏了重要方面或包含少量不准确信息
- 0.3-0.4：答案与问题勉强相关，信息不完整或存在明显偏差
- 0.1-0.2：答案基本不相关，只有个别词汇沾边
- 0.0：完全答非所问，或返回错误信息

返回纯 JSON（score 可以是任意小数，如 0.85）：
{{
    "score": <0.0-1.0>,
    "reason": "一句话理由"
}}"""

CONTEXT_RELEVANCE_PROMPT = """你是检索质量评审专家。判断检索到的文档/数据是否与用户问题相关。
请使用 0.0-1.0 的连续评分，**不要都打同一分数**。

## 用户问题
{question}

## 检索到的内容（{chunk_count} 条）
{contexts}

## 详细评判标准
- 0.9-1.0：全部内容高度相关，每条都能直接用于回答问题
- 0.7-0.8：大部分内容相关，少量无关但不影响整体
- 0.5-0.6：约一半内容相关，一半是噪音
- 0.3-0.4：只有少量内容相关，大量噪音
- 0.1-0.2：只有个别词汇匹配，基本不相关
- 0.0：全部无关，检索完全失败

返回纯 JSON（score 可以是任意小数，如 0.65）：
{{
    "score": <0.0-1.0>,
    "reason": "一句话理由"
}}"""

GROUNDEDNESS_PROMPT = """你是答案有据性评审专家。逐条对比系统答案中的主张与检索到的参考内容，判断每个主张是否有明确依据。
请使用 0.0-1.0 的连续评分，**不要都打同一分数**。

## 用户问题
{question}

## 检索到的参考内容
{contexts}

## 系统答案
{answer}

## 详细评判标准
- 0.9-1.0：答案中每一个关键主张都能在参考内容中找到原文支撑，无编造
- 0.7-0.8：绝大部分主张有依据，个别细节或数值无法验证，但整体可信
- 0.5-0.6：约一半主张有依据，另一半缺乏明确支撑
- 0.3-0.4：多处主张无依据，可能包含编造或推测
- 0.1-0.2：只有个别主张能找到微弱关联，绝大部分无法验证
- 0.0：全部主张无依据，明显是编造的

返回纯 JSON（score 可以是任意小数，如 0.75）：
{{
    "score": <0.0-1.0>,
    "reason": "一句话理由"
}}"""


class AsyncEvaluator:
    """单例评估器：fire-and-forget 模式，后台调 LLM 打分存入 trulens_eval"""

    _instance: AsyncEvaluator | None = None

    def __init__(self):
        settings = get_settings()
        self.enabled = settings.TRULENS_ENABLED
        self.sample_rate = settings.EVAL_SAMPLE_RATE

        if self.enabled:
            self._llm = ChatOpenAI(
                model=settings.CHAT_MODEL,
                api_key=settings.DASHSCOPE_API_KEY,
                base_url=settings.BASE_URL_CHAT,
                temperature=0,
            )
            self._engine = create_async_engine(
                f"postgresql+asyncpg://{settings.DB_USER}:{settings.DB_PASSWORD}"
                f"@{settings.DB_HOST}:{settings.DB_PORT}/trulens_eval"
            )
            logger.info(
                f"[AsyncEvaluator] 已启用，采样率={self.sample_rate:.0%}"
            )
        else:
            logger.info("[AsyncEvaluator] 已禁用（TRULENS_ENABLED=false）")

    @classmethod
    def get_instance(cls) -> AsyncEvaluator:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def evaluate(
        self,
        question: str,
        sql: str = "",
        data: list[dict] | None = None,
        summary: str = "",
        error: str | None = None,
        token_usage: dict | None = None,
    ):
        """fire-and-forget：按采样率随机决定是否打分"""
        if not self.enabled:
            return

        if random.random() > self.sample_rate:
            return  # 不采样，跳过

        ctx = {
            "question": question,
            "sql": sql,
            "summary": summary,
            "error": error or "",
            "data": data or [],
            "row_count": len(data) if data else 0,
            "token_usage": token_usage or {},
        }
        asyncio.create_task(self._run(ctx))

    async def _run(self, ctx: dict):
        """后台任务：跑 NL2SQL 三指标 → 写入 trulens_eval"""
        try:
            # ① 客观指标 — SQL 可执行率 (0/1)
            sql_ok = score_sql_valid(ctx["error"] if ctx["error"] else None)

            # ② 客观指标 — 数据返回率 (有数据=1.0, 空结果=0.5, SQL失败=0)
            has_data = 1.0 if ctx.get("row_count", 0) > 0 else (0.5 if sql_ok > 0 else 0.0)

            # ③ 主观指标 — LLM 裁判结果相关性（只对成功的查询打分）
            relevance = 0.0
            reason = ""
            if sql_ok > 0 and ctx["summary"]:
                try:
                    from src.rag.evaluation.nl2sql_metrics import RESULT_RELEVANCE_PROMPT
                    data_preview = json.dumps(ctx.get("data", [])[:5], ensure_ascii=False, default=str)
                    prompt = RESULT_RELEVANCE_PROMPT.format(
                        question=ctx["question"],
                        summary=ctx["summary"][:500],
                        data_preview=data_preview[:2000],
                    )
                    response = await self._llm.ainvoke([SystemMessage(content=prompt)])
                    content = response.content.strip()
                    match = re.search(r"\{[^{}]*\}", content)
                    if match:
                        result = json.loads(match.group())
                        relevance = float(result.get("score", 0))
                        reason = result.get("reason", "")
                except Exception as e:
                    logger.warning(f"[AsyncEvaluator] 相关性评分失败: {e}")

            # ④ 写入 trulens_eval
            await self._save(ctx, sql_ok, has_data, relevance, reason)

        except Exception as e:
            logger.warning(f"[AsyncEvaluator] 评估异常（不影响主流程）: {e}")

    async def _save(self, ctx: dict, sql_ok: float, has_data: float, relevance: float, reason: str):
        """写入 trulens_eval.online_scores 表（NL2SQL 三指标）"""
        from sqlalchemy import text

        try:
            async with self._engine.begin() as conn:
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS online_scores (
                        id SERIAL PRIMARY KEY,
                        eval_type VARCHAR(20) DEFAULT 'nl2sql',
                        question TEXT NOT NULL,
                        answer TEXT DEFAULT '',
                        sql TEXT DEFAULT '',
                        summary TEXT DEFAULT '',
                        error TEXT DEFAULT '',
                        score_sql_valid REAL DEFAULT 0,
                        score_relevance REAL DEFAULT 0,
                        score_reason TEXT DEFAULT '',
                        score_has_data REAL DEFAULT 0,
                        score_context_relevance REAL DEFAULT 0,
                        score_context_relevance_reason TEXT DEFAULT '',
                        score_groundedness REAL DEFAULT 0,
                        score_groundedness_reason TEXT DEFAULT '',
                        token_input INTEGER DEFAULT 0,
                        token_output INTEGER DEFAULT 0,
                        token_calls INTEGER DEFAULT 0,
                        token_cost_usd REAL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """))
                for col_sql in [
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS eval_type VARCHAR(20) DEFAULT 'nl2sql'",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS answer TEXT DEFAULT ''",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS score_has_data REAL DEFAULT 0",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS score_context_relevance REAL DEFAULT 0",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS score_context_relevance_reason TEXT DEFAULT ''",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS score_groundedness REAL DEFAULT 0",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS score_groundedness_reason TEXT DEFAULT ''",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS token_input INTEGER DEFAULT 0",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS token_output INTEGER DEFAULT 0",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS token_calls INTEGER DEFAULT 0",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS token_cost_usd REAL DEFAULT 0",
                ]:
                    try:
                        await conn.execute(text(col_sql))
                    except Exception:
                        pass

                await conn.execute(
                    text("""
                        INSERT INTO online_scores
                            (eval_type, question, sql, summary, error,
                             score_sql_valid, score_has_data, score_relevance, score_reason,
                             token_input, token_output, token_calls, token_cost_usd)
                        VALUES ('nl2sql', :q, :sql, :summary, :error, :sql_ok, :has_data, :relevance, :reason,
                                :token_input, :token_output, :token_calls, :token_cost_usd)
                    """),
                    {
                        "q": ctx["question"][:500],
                        "sql": ctx["sql"][:2000],
                        "summary": ctx["summary"][:2000],
                        "error": ctx["error"][:500],
                        "sql_ok": sql_ok,
                        "has_data": has_data,
                        "relevance": relevance,
                        "reason": reason[:500],
                        "token_input": ctx.get("token_usage", {}).get("input_tokens", 0),
                        "token_output": ctx.get("token_usage", {}).get("output_tokens", 0),
                        "token_calls": ctx.get("token_usage", {}).get("calls", 0),
                        "token_cost_usd": ctx.get("token_usage", {}).get("cost_usd", 0),
                    },
                )
            logger.info(
                f"[AsyncEvaluator] NL2SQL 评分已保存: sql_valid={sql_ok:.0f} "
                f"has_data={has_data:.1f} relevance={relevance:.2f} q={ctx['question'][:40]}"
            )
        except Exception as e:
            logger.warning(f"[AsyncEvaluator] 写入 trulens_eval 失败: {e}")

    # ── 知识检索评估 ────────────────────────────────────────────────

    def evaluate_knowledge(self, question: str, answer: str, contexts: list[str] | None = None, token_usage: dict | None = None):
        """fire-and-forget：评估知识检索 RAG Triad（答案相关性 + 上下文相关性 + 有据性）"""
        if not self.enabled:
            return

        if random.random() > self.sample_rate:
            return

        ctx = {
            "question": question,
            "answer": answer,
            "contexts": contexts or [],
            "token_usage": token_usage or {},
        }
        asyncio.create_task(self._run_knowledge(ctx))

    async def _run_knowledge(self, ctx: dict):
        """后台任务：LLM 裁判 RAG Triad → 写入 trulens_eval"""
        contexts_text = "\n---\n".join(ctx["contexts"]) if ctx["contexts"] else "（无检索内容）"
        scores = {
            "answer_relevance": 0.0,
            "answer_relevance_reason": "",
            "context_relevance": 0.0,
            "context_relevance_reason": "",
            "groundedness": 0.0,
            "groundedness_reason": "",
        }

        try:
            n_ctx = len(ctx["contexts"])
            logger.info(
                f"[AsyncEvaluator] 开始知识评估: q={ctx['question'][:40]} "
                f"contexts={n_ctx} answer_len={len(ctx['answer'])}"
            )

            # ① 答案相关性
            prompt = ANSWER_RELEVANCE_PROMPT.format(
                question=ctx["question"],
                answer=ctx["answer"][:2000],
            )
            response = await self._llm.ainvoke([SystemMessage(content=prompt)])
            scores["answer_relevance"], scores["answer_relevance_reason"] = self._parse_score(response.content)
            logger.info(
                f"[AsyncEvaluator] ① 答案相关性: score={scores['answer_relevance']:.2f} "
                f"reason={scores['answer_relevance_reason'][:60]} "
                f"raw={response.content[:200]}"
            )

            # ② 上下文相关性
            prompt = CONTEXT_RELEVANCE_PROMPT.format(
                question=ctx["question"],
                chunk_count=n_ctx,
                contexts=contexts_text[:3000],
            )
            response = await self._llm.ainvoke([SystemMessage(content=prompt)])
            scores["context_relevance"], scores["context_relevance_reason"] = self._parse_score(response.content)
            logger.info(
                f"[AsyncEvaluator] ② 上下文相关性: score={scores['context_relevance']:.2f} "
                f"reason={scores['context_relevance_reason'][:60]} "
                f"raw={response.content[:200]}"
            )

            # ③ 有据性
            if n_ctx > 0:
                prompt = GROUNDEDNESS_PROMPT.format(
                    question=ctx["question"],
                    contexts=contexts_text[:3000],
                    answer=ctx["answer"][:2000],
                )
                response = await self._llm.ainvoke([SystemMessage(content=prompt)])
                scores["groundedness"], scores["groundedness_reason"] = self._parse_score(response.content)
                logger.info(
                    f"[AsyncEvaluator] ③ 有据性: score={scores['groundedness']:.2f} "
                    f"reason={scores['groundedness_reason'][:60]} "
                    f"raw={response.content[:200]}"
                )
            else:
                logger.warning("[AsyncEvaluator] ③ 有据性: 跳过（无检索内容）")

            await self._save_knowledge(ctx, scores)

        except Exception as e:
            logger.warning(f"[AsyncEvaluator] 知识评估异常: {e}")

    @staticmethod
    def _parse_score(content: str) -> tuple[float, str]:
        """从 LLM 返回的 JSON 中提取 score 和 reason"""
        match = re.search(r"\{[^{}]*\}", content.strip())
        if match:
            result = json.loads(match.group())
            return float(result.get("score", 0)), result.get("reason", "")
        return 0.0, ""

    async def _save_knowledge(self, ctx: dict, scores: dict):
        """写入 trulens_eval.online_scores 表（eval_type=knowledge, RAG Triad）"""
        from sqlalchemy import text

        try:
            async with self._engine.begin() as conn:
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS online_scores (
                        id SERIAL PRIMARY KEY,
                        eval_type VARCHAR(20) DEFAULT 'nl2sql',
                        question TEXT NOT NULL,
                        answer TEXT DEFAULT '',
                        sql TEXT DEFAULT '',
                        summary TEXT DEFAULT '',
                        error TEXT DEFAULT '',
                        score_sql_valid REAL DEFAULT 0,
                        score_relevance REAL DEFAULT 0,
                        score_reason TEXT DEFAULT '',
                        score_has_data REAL DEFAULT 0,
                        score_context_relevance REAL DEFAULT 0,
                        score_context_relevance_reason TEXT DEFAULT '',
                        score_groundedness REAL DEFAULT 0,
                        score_groundedness_reason TEXT DEFAULT '',
                        token_input INTEGER DEFAULT 0,
                        token_output INTEGER DEFAULT 0,
                        token_calls INTEGER DEFAULT 0,
                        token_cost_usd REAL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """))
                # 兼容旧表：尝试加列
                for col_sql in [
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS eval_type VARCHAR(20) DEFAULT 'nl2sql'",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS answer TEXT DEFAULT ''",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS score_has_data REAL DEFAULT 0",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS score_context_relevance REAL DEFAULT 0",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS score_context_relevance_reason TEXT DEFAULT ''",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS score_groundedness REAL DEFAULT 0",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS score_groundedness_reason TEXT DEFAULT ''",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS token_input INTEGER DEFAULT 0",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS token_output INTEGER DEFAULT 0",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS token_calls INTEGER DEFAULT 0",
                    "ALTER TABLE online_scores ADD COLUMN IF NOT EXISTS token_cost_usd REAL DEFAULT 0",
                ]:
                    try:
                        await conn.execute(text(col_sql))
                    except Exception:
                        pass

                await conn.execute(
                    text("""
                        INSERT INTO online_scores
                            (eval_type, question, answer,
                             score_relevance, score_reason,
                             score_context_relevance, score_context_relevance_reason,
                             score_groundedness, score_groundedness_reason,
                             token_input, token_output, token_calls, token_cost_usd)
                        VALUES ('knowledge', :q, :answer,
                                :rel, :rel_reason,
                                :ctx_rel, :ctx_rel_reason,
                                :gnd, :gnd_reason,
                                :token_input, :token_output, :token_calls, :token_cost_usd)
                    """),
                    {
                        "q": ctx["question"][:500],
                        "answer": ctx["answer"][:2000],
                        "rel": scores["answer_relevance"],
                        "rel_reason": scores["answer_relevance_reason"][:500],
                        "ctx_rel": scores["context_relevance"],
                        "ctx_rel_reason": scores["context_relevance_reason"][:500],
                        "gnd": scores["groundedness"],
                        "gnd_reason": scores["groundedness_reason"][:500],
                        "token_input": ctx.get("token_usage", {}).get("input_tokens", 0),
                        "token_output": ctx.get("token_usage", {}).get("output_tokens", 0),
                        "token_calls": ctx.get("token_usage", {}).get("calls", 0),
                        "token_cost_usd": ctx.get("token_usage", {}).get("cost_usd", 0),
                    },
                )
            logger.info(
                f"[AsyncEvaluator] 知识评分已保存: "
                f"answer_relevance={scores['answer_relevance']:.2f} "
                f"context_relevance={scores['context_relevance']:.2f} "
                f"groundedness={scores['groundedness']:.2f} "
                f"q={ctx['question'][:40]}"
            )
        except Exception as e:
            logger.warning(f"[AsyncEvaluator] 写入知识评分失败: {e}")

    async def close(self):
        if self.enabled and hasattr(self, "_engine"):
            await self._engine.dispose()


def get_async_evaluator() -> AsyncEvaluator:
    return AsyncEvaluator.get_instance()
