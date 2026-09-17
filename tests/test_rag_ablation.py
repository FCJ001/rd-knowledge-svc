# ============================================================
# 消融结果落盘 & 单因子对比 单元测试
#
# 覆盖 run_rag_experiments 里两个纯函数（不连库、不调 LLM）：
#   _save_retrieval_results —— jsonl 追加，字段完整
#   _compare_with_history  —— 单因子优先、同参数不误比、多变体告警
#
# 为什么值得测：这两个函数是"改一个参数→跑→看涨跌"闭环的关键，
# 写错了会让人按错误的对比结论去调参（比调不出参更糟）。
# ============================================================

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_rag_experiments.py"


@pytest.fixture()
def rexp(tmp_path, monkeypatch):
    """加载实验脚本模块，并把结果落盘路径重定向到临时目录。"""
    spec = importlib.util.spec_from_file_location("rexp_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rexp_under_test"] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "RESULTS_PATH", tmp_path / "rag_eval_results.jsonl")
    return mod


def _summary(hit5=0.6, recall5=0.45, mrr_=0.52, latency=820.0):
    return {
        "doc_rag": {
            "count": 10,
            "hit_rate@5": hit5,
            "recall@5": recall5,
            "mrr": mrr_,
            "latency_ms_avg": latency,
        }
    }


def _params(use_hyde=False, top_k=20, rerank_top_k=5):
    return {"use_hyde": use_hyde, "top_k": top_k,
            "rerank_top_k": rerank_top_k, "label": ""}


# ── 落盘 ────────────────────────────────────────────────────────────────

def test_save_appends_one_jsonl_line_per_run(rexp):
    rexp._save_retrieval_results(_summary(), _params(), "scripts/eval_dataset.csv")
    rexp._save_retrieval_results(_summary(hit5=0.8), _params(use_hyde=True),
                                 "scripts/eval_dataset.csv")

    lines = rexp.RESULTS_PATH.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    rec = json.loads(lines[0])
    assert rec["dataset"] == "eval_dataset.csv"
    assert rec["params"] == _params()
    assert rec["results"]["doc_rag"]["hit_rate@5"] == 0.6
    assert "timestamp" in rec


def test_save_skips_when_no_summaries(rexp):
    """全部通道被跳过时不写文件，避免污染历史对比。"""
    rexp._save_retrieval_results({}, _params(), "scripts/eval_dataset.csv")
    assert not rexp.RESULTS_PATH.exists()


# ── 单因子对比 ───────────────────────────────────────────────────────────

def test_compare_ignores_same_params(rexp, capsys):
    """同参数重跑不应拿自己跟自己比。"""
    rexp._save_retrieval_results(_summary(), _params(), "scripts/eval_dataset.csv")
    rexp._compare_with_history(_summary(), _params())
    out = capsys.readouterr().out
    assert "还没有其它参数组合的记录" in out


def test_compare_prefers_single_factor(rexp, capsys):
    """历史里同时有 base 和 hyde 时，HyDE+top_k50 应与"只差 top_k"的 hyde 比。"""
    rexp._save_retrieval_results(_summary(), _params(), "scripts/eval_dataset.csv")
    rexp._save_retrieval_results(_summary(hit5=0.8), _params(use_hyde=True),
                                 "scripts/eval_dataset.csv")

    rexp._compare_with_history(_summary(hit5=0.85), _params(use_hyde=True, top_k=50))
    out = capsys.readouterr().out

    assert "变更项" in out
    assert "top_k: 20 → 50" in out
    assert "use_hyde" not in out.split("变更项")[1].split("\n")[0]  # 只报这一个变量
    assert "0.800" in out and "0.850" in out


def test_compare_warns_when_no_single_factor_history(rexp, capsys):
    """只有多参数差异的历史时，要显式告警，避免错误归因。"""
    rexp._save_retrieval_results(_summary(), _params(), "scripts/eval_dataset.csv")

    rexp._compare_with_history(_summary(hit5=0.85), _params(use_hyde=True, top_k=50))
    out = capsys.readouterr().out

    assert "只差一个参数" in out
    assert "相差 2 个参数" in out


def test_compare_shows_direction_and_delta(rexp, capsys):
    """对比表要能看出涨还是跌，涨跌方向不能反。"""
    rexp._save_retrieval_results(_summary(hit5=0.80), _params(), "scripts/eval_dataset.csv")
    rexp._compare_with_history(_summary(hit5=0.60), _params(top_k=50))
    out = capsys.readouterr().out

    # ★ 必须定位到 hit_rate@5 那一行再断言方向。
    #   只写 `assert "↓" in out` 是假测试——延迟行也有 ↓，恒为真。
    row = next(line for line in out.splitlines() if "hit_rate@5" in line)
    assert "↓" in row, f"0.80→0.60 应显示下降箭头，实际: {row!r}"
    assert "0.200" in row


def test_compare_missing_history_file_is_silent(rexp, capsys):
    """首次运行没有历史文件，不应报错。"""
    rexp._compare_with_history(_summary(), _params())
    assert capsys.readouterr().out == ""
