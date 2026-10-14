"""阶段 5 Agent 评测的确定性测试（§9）。

★ 分两层：
  1. 指标纯函数（normalize / match / build_agent_result）—— 无依赖。
  2. eval_agent.evaluate —— 注入假 store + 假 llm + 内存 SQLite，**不连 PG、不跑真实模型**
     就能钉住"逐例 → 聚合"的接线（同 test_eval_runner 的思路）。
"""

from types import SimpleNamespace

from app.evals.metrics_agent import (
    build_agent_result,
    normalize_tool_spec,
    tool_selection_match,
)
from scripts.eval_agent import _apply_fail_under, evaluate


# ---------------------------------------------------------------- 指标纯函数


def test_normalize_tool_spec():
    assert normalize_tool_spec(None) == set()
    assert normalize_tool_spec("calc") == {"calc"}
    assert normalize_tool_spec(["calc", "now"]) == {"calc", "now"}


def test_tool_selection_match_is_set_equality():
    # 顺序无关
    assert tool_selection_match(["calc", "search_kb"], ["search_kb", "calc"])
    # 漏调 → 不等
    assert not tool_selection_match(["calc", "search_kb"], ["calc"])
    # 多调 → 不等（该只 calc 却还检索）
    assert not tool_selection_match(["calc"], ["calc", "search_kb"])


def _o(cid, expect, tools, steps=1, stopped=None, cost=0.0):
    return {"id": cid, "expect_tool": expect, "tools": tools,
            "steps": steps, "stopped_reason": stopped, "cost_yuan": cost}


def test_build_agent_result_metrics():
    outcomes = [
        _o("a", ["calc"], ["calc"], steps=1, cost=0.001),        # match
        _o("b", ["search_kb", "calc"], ["calc"], steps=1, cost=0.003),   # 漏检索 → miss
        _o("c", [], [], steps=0, stopped="max_steps", cost=0.002),         # match, 保护触发
    ]
    r = build_agent_result(outcomes, fast_path_cost_yuan=0.001)
    m = r["metrics"]
    assert m["tool_selection_accuracy"] == round(2 / 3, 4)
    assert m["avg_steps"] == round(2 / 3, 4)
    assert m["protection_trigger_rate"] == round(1 / 3, 4)
    assert m["avg_cost_yuan"] == round(0.006 / 3, 4)
    assert m["cost_multiple_vs_fast_path"] == round(m["avg_cost_yuan"] / 0.001, 2)
    by_id = {c["id"]: c for c in r["cases"]}
    assert by_id["a"]["tool_match"] is True
    assert by_id["b"]["tool_match"] is False
    assert by_id["c"]["tool_match"] is True


def test_build_agent_result_empty():
    r = build_agent_result([])
    assert r["metrics"]["tool_selection_accuracy"] is None
    assert r["cases"] == []


def test_apply_fail_under():
    r = build_agent_result([_o("a", ["calc"], ["calc"])])   # accuracy 1.0
    args = SimpleNamespace(fail_under_tool=0.8)
    assert _apply_fail_under(r, args) == 0
    args = SimpleNamespace(fail_under_tool=1.1)
    assert _apply_fail_under(r, args) == 1
    args = SimpleNamespace(fail_under_tool=None)
    assert _apply_fail_under(r, args) == 0


# ---------------------------------------------------------------- evaluate 接线


class _AlwaysAnswerLLM:
    """永远直接给最终答案、不调工具——把 evaluate 变成确定性、零检索。"""

    def __init__(self):
        self.calls = 0

    async def chat(self, messages, **kwargs):
        from app.clients.llm import LLMResult
        self.calls += 1
        return LLMResult(
            content="答", prompt_tokens=100, completion_tokens=10,
            cost_yuan=0.0005, raw={"tool_calls": None, "reasoning_content": None},
        )


async def test_evaluate_aggregates_without_db_or_real_llm(session_factory):
    store = SimpleNamespace(owner_id=9999)     # 假 store：owned_store=False，跳过 seed/cleanup
    llm = _AlwaysAnswerLLM()
    cases = [
        {"id": "c1", "question": "随便问一", "expect_tool": []},          # 不调工具 → match
        {"id": "c2", "question": "随便问二", "expect_tool": ["calc"]},    # 期望 calc → miss
    ]
    result, code = await evaluate(
        llm=llm, embedder=None, reranker=None, store=store,
        session_factory=session_factory, cases=cases, documents=[],
    )
    assert code == 0
    m = result["metrics"]
    assert result["config"]["case_count"] == 2
    assert m["tool_selection_accuracy"] == 0.5      # c1 对、c2 错
    assert m["avg_steps"] == 0.0                    # 全程没调工具
    assert m["protection_trigger_rate"] == 0.0
    assert llm.calls == 2                            # 每条用例各一次 LLM 调用
