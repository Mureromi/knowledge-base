"""LLM judge 测试（阶段 6，docs/项目实现文档-阶段6.md §4.3 / §7.1 / §7.2）。

★ 分两层，互不混淆（§7.2①）：
  - **计算/解析逻辑**：注入喂固定 JSON 的假 LLM，确定性验证 faithfulness/completeness 的打分。
  - **judge 本身可不可靠**：由 validate_judge() 负责——非 CI 用鸭子类型的"完美裁判"验证聚合，
    真·裁判判对率用 @pytest.mark.real 标出（默认被 `-m 'not real'`  deselect）。
"""

import json
from types import SimpleNamespace

import pytest

from app.evals.judge import (
    FAITHFULNESS_PROMPT,
    Judge,
    build_completeness_prompt,
    build_faithfulness_prompt,
    validate_judge,
    _safe_json,
)


def _llm_returning(payload) -> SimpleNamespace:
    """构造一个假 llm_client.chat 返回值：content 为给定 JSON（或原始串）。"""
    content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return SimpleNamespace(content=content, cost_yuan=0.0002)


class _ScriptedLLM:
    """按预设顺序吐响应；记录每次收到的 prompt，便于断言注入正确。"""

    def __init__(self, *payloads):
        self._payloads = list(payloads)
        self.prompts: list[str] = []

    async def chat(self, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        payload = self._payloads.pop(0) if self._payloads else {}
        return _llm_returning(payload)


# ---------------------------------------------------------------------------
# 判定解析（§7.1：缺字段/多字段/非法 JSON → 不崩，返回合理默认值）
# ---------------------------------------------------------------------------

def test_safe_json_plain():
    assert _safe_json('{"faithful": true}') == {"faithful": True}


def test_safe_json_markdown_wrapped():
    """模型偶尔会裹 ```json ...``` 代码块，正则要能捞出首个 {…}。"""
    text = '这是我的判断：\n```json\n{"faithful": false, "reason": "编造"}\n```'
    assert _safe_json(text) == {"faithful": False, "reason": "编造"}


def test_safe_json_with_prefix_suffix():
    text = '废话前缀 {"covered": [1]} 废话后缀'
    assert _safe_json(text) == {"covered": [1]}


def test_safe_json_garbage_returns_empty_dict():
    assert _safe_json("完全不是 JSON") == {}
    assert _safe_json("") == {}
    assert _safe_json(None) == {}


def test_safe_json_non_object_returns_empty():
    """合法 JSON 但不是对象（数组/标量）→ 也兜底成 {}。"""
    assert _safe_json("[1, 2, 3]") == {}
    assert _safe_json("42") == {}


@pytest.mark.asyncio
async def test_faithfulness_true_and_false():
    j_yes = Judge(llm=_ScriptedLLM({"faithful": True, "reason": "有依据"}))
    assert (await j_yes.faithfulness("答案", ["资料"]))["score"] == 1.0

    j_no = Judge(llm=_ScriptedLLM({"faithful": False, "reason": "编造了 30 天"}))
    res = await j_no.faithfulness("答案", ["资料"])
    assert res["score"] == 0.0
    assert "30 天" in res["reason"]


@pytest.mark.asyncio
async def test_faithfulness_defaults_to_unfaithful_on_parse_failure():
    """解析不出 → 视为不忠实（保守：坏尺子宁可误杀不可放过幻觉）。"""
    j = Judge(llm=_ScriptedLLM("模型抽风，没返回 JSON"))
    assert (await j.faithfulness("答案", ["资料"]))["score"] == 0.0


@pytest.mark.asyncio
async def test_completeness_scores_by_covered_ratio():
    j = Judge(llm=_ScriptedLLM({"covered": [1], "missing": [2], "reason": "第2点未提及"}))
    res = await j.completeness("答案", ["要点一", "要点二"])
    assert res["score"] == 0.5
    assert res["missing"] == ["要点二"]


@pytest.mark.asyncio
async def test_completeness_ignores_out_of_range_indices():
    """越界编号被过滤，绝不让 points[i-1] 抛 IndexError。"""
    j = Judge(llm=_ScriptedLLM({"covered": [], "missing": [1, 99]}))
    res = await j.completeness("答案", ["只有这一点"])
    assert res["missing"] == ["只有这一点"]      # 99 越界被丢弃，1 合法保留


@pytest.mark.asyncio
async def test_judge_accumulates_cost_and_calls():
    j = Judge(llm=_ScriptedLLM({"faithful": True}, {"covered": [1]}))
    await j.faithfulness("a", ["c"])
    await j.completeness("a", ["p"])
    assert j.calls == 2
    assert j.cost_yuan == pytest.approx(0.0004)


# ---------------------------------------------------------------------------
# ★ §7.2② 反脆弱断言：把"我踩过两次的 .format 大括号坑"永久钉住
# ---------------------------------------------------------------------------

def test_naive_format_would_raise_keyerror():
    """证明坑是真的：模板含单大括号 JSON 示例时，str.format 会把 { 当占位符 → KeyError。"""
    with pytest.raises(KeyError):
        FAITHFULNESS_PROMPT.format(ctx="x", out="y")     # 示例里的 {"faithful": ...} 触发


def test_judge_prompt_avoids_format_trap():
    """我们的规避方案：占位符 <<...>> + .replace()，含 JSON 示例也不炸，且大括号原样保留。"""
    out = build_faithfulness_prompt("回答含大括号 { 也是普通字符", ["资料 A", "资料 B"])
    # 占位符已替换
    assert "<<CTX>>" not in out and "<<OUT>>" not in out
    # 发给模型的 JSON 示例保持合法单大括号（没被 format 吃掉）
    assert '{"faithful": false' in out
    # 回答与资料都注入进来了
    assert "回答含大括号 { 也是普通字符" in out and "资料 A" in out

    out2 = build_completeness_prompt("答案", ["要点一", "要点二"])
    assert "<<POINTS>>" not in out2 and "<<OUT>>" not in out2
    assert '"covered"' in out2      # 完整 prompt 未被 format 破坏


# ---------------------------------------------------------------------------
# validate_judge：把"尺子准不准"变成可执行检查（§4.3）
# ---------------------------------------------------------------------------

class _PerfectJudge:
    """鸭子类型完美裁判：答案与资料完全一致才判忠实——恰合 SELF_CHECK_CASES 的期望。"""

    cost_yuan = 0.0

    async def faithfulness(self, answer, contexts):
        joined = "\n".join(contexts)
        faithful = answer.strip() == joined.strip()
        return {"score": 1.0 if faithful else 0.0, "reason": ""}


class _AlwaysFaithfulJudge:
    """坏尺子：什么都判忠实（正是 §1.1 里 DeepEval 的症状）。"""

    cost_yuan = 0.0

    async def faithfulness(self, answer, contexts):
        return {"score": 1.0, "reason": "看起来都对"}


@pytest.mark.asyncio
async def test_validate_judge_perfect_is_one():
    assert await validate_judge(_PerfectJudge()) == 1.0


@pytest.mark.asyncio
async def test_validate_judge_always_faithful_fails():
    """1 好 3 坏里，全判忠实的坏裁判只对 1/4 → 判对率 < 1.0，评测不可信。"""
    ratio = await validate_judge(_AlwaysFaithfulJudge())
    assert ratio < 1.0
    assert ratio == pytest.approx(1 / 4)


# ---------------------------------------------------------------------------
# 真·裁判判对率：需真实 LLM Key，默认不进 CI（addopts -m 'not real'）
# ---------------------------------------------------------------------------

@pytest.mark.real
@pytest.mark.asyncio
async def test_validate_real_judge_is_trustworthy():
    """§8 验收项：真实 DeepSeek 当裁判，validate_judge() 判对率必须 = 1.0。"""
    from app.clients.llm import llm_client

    await llm_client.startup()
    try:
        assert await validate_judge(Judge(llm=llm_client)) == 1.0
    finally:
        await llm_client.shutdown()
