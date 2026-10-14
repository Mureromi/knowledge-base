"""Agent 专属评测指标（阶段 5 §9.3）。纯函数，零依赖，便于确定性测试。

这四个指标是"用阶段 6 评测体系量 Agent"的落点——它们量的不是检索质量
（阶段 6 已覆盖），而是 **Agent 这条路径本身好不好**：

| 指标 | 定义 | 为什么重要 |
| --- | --- | --- |
| 工具选择准确率 | 实际调用的工具集合 == 期望集合 | 定位"选错工具"这类失败，比只看答案更能归因 |
| 步数 | 平均 tool 调用次数 | 效率；异常高说明在来回试 |
| 保护触发率 | stopped_reason != null 的比例 | **高说明项目不可用**——用户频繁碰上限 |
| 成本倍率 | Agent 平均成本 / 快速路径平均成本 | 决定"Agent 该不该默认开" |
"""

from typing import Any, Iterable


def normalize_tool_spec(expect_tool: Any) -> set[str]:
    """把 expect_tool（裸字符串 / 列表 / None）统一成集合。"""
    if expect_tool is None:
        return set()
    if isinstance(expect_tool, str):
        return {expect_tool}
    return set(expect_tool)


def tool_selection_match(expected: Iterable[str], actual: Iterable[str]) -> bool:
    """工具选择是否正确：实际调用的工具集合 == 期望集合（严格相等）。

    ★ 用集合相等而非"包含"：多调（该只 calc 却还检索）与漏调同样是有问题的行为，
      只是这里以"集合是否一致"作为第一道粗筛；细粒度诊断看逐例的 tools 明细。
    """
    return set(expected) == set(actual)


def build_agent_result(
    outcomes: list[dict],
    *,
    fast_path_cost_yuan: float | None = None,
) -> dict[str, Any]:
    """把逐例结果聚合成结果 JSON（纯函数，test_eval_agent 不连库不跑模型即可验证）。

    每个 outcome 约定字段：
        id, question, expect_tool(list|str), tools(list 实际调用名),
        steps(int), stopped_reason(str|None), cost_yuan(float), passed(bool 可选)
    """
    def _mean(vals: list[float]) -> float | None:
        return round(sum(vals) / len(vals), 4) if vals else None

    n = len(outcomes)
    matches = [
        tool_selection_match(normalize_tool_spec(o.get("expect_tool")), o.get("tools") or [])
        for o in outcomes
    ]
    agent_avg_cost = _mean([float(o.get("cost_yuan") or 0.0) for o in outcomes])
    cost_multiple = (
        round(agent_avg_cost / fast_path_cost_yuan, 2)
        if fast_path_cost_yuan and agent_avg_cost is not None
        else None
    )
    return {
        "config": {"case_count": n},
        "metrics": {
            "tool_selection_accuracy": round(sum(matches) / n, 4) if n else None,
            "avg_steps": _mean([o.get("steps") or 0 for o in outcomes]),
            "protection_trigger_rate": (
                round(sum(1 for o in outcomes if o.get("stopped_reason")) / n, 4)
                if n else None
            ),
            "avg_cost_yuan": agent_avg_cost,
            "cost_multiple_vs_fast_path": cost_multiple,
        },
        "cases": [
            {
                "id": o["id"],
                "expect_tool": sorted(normalize_tool_spec(o.get("expect_tool"))),
                "tools": sorted(set(o.get("tools") or [])),
                "tool_match": m,
                "steps": o.get("steps"),
                "stopped_reason": o.get("stopped_reason"),
                "cost_yuan": round(float(o.get("cost_yuan") or 0.0), 6),
            }
            for o, m in zip(outcomes, matches, strict=False)
        ],
    }
