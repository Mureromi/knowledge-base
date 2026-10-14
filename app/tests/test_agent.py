"""阶段 5 Agent 循环测试（docs/项目实现文档-阶段5.md §8.2）。

★ 全部用可编程的 ScriptedLLM 驱动，**不需要真实 LLM**：
  它能精确构造「调工具→再调工具→给答案」「调不存在的工具」「循环调用」「参数非法 JSON」
  等路径（§8.1）。断言聚焦循环本身：步数、成本累计、四层保护、引用回表、思考关闭。
"""

from app.clients.llm import LLMResult
from app.core.config import settings
from app.models.chat_log import ChatLog
from app.services.agent import AgentRunner
from sqlalchemy import select


class ScriptedLLM:
    """按剧本返回响应的假 LLM：第 N 次调用返回第 N 个剧本（用尽则重复最后一个）。"""

    def __init__(self, script):
        self.script = script
        self.calls = []
        self._i = 0

    async def chat(self, messages, **kwargs):
        self.calls.append({"messages": [dict(m) for m in messages], **kwargs})
        step = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        return LLMResult(
            content=step.get("content", ""),
            prompt_tokens=100,
            completion_tokens=20,
            cost_yuan=step.get("cost_yuan", 0.0002),
            raw={
                "tool_calls": step.get("tool_calls"),
                "reasoning_content": step.get("reasoning"),
            },
        )


def _call(name, args_json, cid="c1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args_json}}


def _runner(db, llm, owner_id=1, embedder=None, reranker=None):
    return AgentRunner(db, llm, owner_id=owner_id, embedder=embedder, reranker=reranker)


# ---------------------------------------------------------------- §8.2 基础路径


async def test_direct_answer_no_tool(db_session):
    llm = ScriptedLLM([{"content": "直接回答", "tool_calls": None}])
    run = await _runner(db_session, llm).run("你好")
    assert run.answer == "直接回答"
    assert run.steps == []
    assert run.llm_calls == 1
    assert run.stopped_reason is None


async def test_single_tool(db_session, seed_document, fake_embedder, fake_reranker):
    await seed_document(owner_id=1, chunks=[("退款政策：7 天内可退货", None)])
    llm = ScriptedLLM([
        {"tool_calls": [_call("search_kb", '{"query": "退款"}')]},
        {"content": "可以 7 天退货 [1]", "tool_calls": None},
    ])
    run = await _runner(
        db_session, llm, embedder=fake_embedder, reranker=fake_reranker
    ).run("能退货吗")
    assert len(run.steps) == 1
    assert run.tool_counts == {"search_kb": 1}
    assert run.llm_calls == 2
    assert run.citations, "search_kb 召回后必须回填 citations"


async def test_multi_tool_chain(db_session, seed_document, fake_embedder):
    await seed_document(owner_id=1, chunks=[("单价 89 元", None)])
    llm = ScriptedLLM([
        {"tool_calls": [_call("search_kb", '{"query": "单价"}', "c1")]},
        {"tool_calls": [_call("calc", '{"expr": "89 * 2"}', "c2")]},
        {"content": "两件 178 元", "tool_calls": None},
    ])
    run = await _runner(db_session, llm, embedder=fake_embedder).run("两件多少钱")
    assert run.llm_calls == 3
    assert [s.tool for s in run.steps] == ["search_kb", "calc"]


async def test_tool_error_then_self_correct(db_session):
    """§1.2 正面用例：第一次 calc 非法（错误当结果返回），模型换合法的再调一次。"""
    llm = ScriptedLLM([
        {"tool_calls": [_call("calc", '{"expr": "1 / 0"}', "c1")]},
        {"tool_calls": [_call("calc", '{"expr": "2 + 2"}', "c2")]},
        {"content": "等于 4", "tool_calls": None},
    ])
    run = await _runner(db_session, llm).run("先算 1/0 再算 2+2")
    assert run.steps[0].is_error is True
    assert run.steps[1].is_error is False
    assert run.stopped_reason is None           # 工具报错不算保护触发


async def test_nonexistent_tool_does_not_crash(db_session):
    llm = ScriptedLLM([
        {"tool_calls": [_call("bogus", "{}")]},
        {"content": "算了", "tool_calls": None},
    ])
    run = await _runner(db_session, llm).run("试试不存在工具")
    assert run.steps[0].is_error is True
    assert "不存在的工具" in run.steps[0].result_preview
    assert run.answer == "算了"                  # 循环继续、正常给出答案


async def test_invalid_json_arguments(db_session):
    llm = ScriptedLLM([
        {"tool_calls": [_call("calc", "not json")]},
        {"content": "重来", "tool_calls": None},
    ])
    run = await _runner(db_session, llm).run("参数坏掉")
    assert run.steps[0].is_error is True
    assert "合法 JSON" in run.steps[0].result_preview
    assert run.answer == "重来"


# ---------------------------------------------------------------- §5.1 四层保护


async def test_max_steps_protection(db_session, monkeypatch):
    monkeypatch.setattr(settings, "agent_max_steps", 2)
    monkeypatch.setattr(settings, "agent_repeat_call_limit", 100)   # 排除 repeat 干扰
    monkeypatch.setattr(settings, "agent_max_cost_yuan", 10.0)      # 排除 cost 干扰
    llm = ScriptedLLM([{"tool_calls": [_call("now", "{}")]}])       # 永远在调工具
    run = await _runner(db_session, llm).run(" endless")
    assert run.stopped_reason == "max_steps"
    # 2 次循环 + 1 次兜底作答
    assert run.llm_calls == 3
    assert run.answer                             # 兜底必须有答案，不能空


async def test_cost_limit_early_exit(db_session, monkeypatch):
    monkeypatch.setattr(settings, "agent_max_cost_yuan", 0.0005)
    llm = ScriptedLLM([
        {"tool_calls": [_call("now", "{}")], "cost_yuan": 0.0004},
        {"tool_calls": [_call("now", "{}")], "cost_yuan": 0.0004},   # 累计 0.0008 超限
    ])
    run = await _runner(db_session, llm).run("贵问题")
    assert run.stopped_reason == "cost_limit"
    assert run.llm_calls == 2                       # 超限当轮即停，无兜底额外调用
    assert run.answer                               # 有降级答案


async def test_timeout_protection(db_session, monkeypatch):
    monkeypatch.setattr(settings, "agent_timeout_seconds", -1.0)
    llm = ScriptedLLM([{"content": "不该走到这", "tool_calls": None}])
    run = await _runner(db_session, llm).run("慢")
    assert run.stopped_reason == "timeout"
    assert run.llm_calls == 1                       # 循环里 0 次，仅兜底 1 次


async def test_repeat_call_detection(db_session, monkeypatch):
    monkeypatch.setattr(settings, "agent_repeat_call_limit", 2)
    monkeypatch.setattr(settings, "agent_max_steps", 10)
    llm = ScriptedLLM([
        {"tool_calls": [_call("calc", '{"expr": "2 + 2"}')]},   # 每次同参数
    ])
    run = await _runner(db_session, llm).run("重复")
    assert run.stopped_reason == "repeat_call"
    assert run.tool_counts.get("calc", 0) == 2       # 前 2 次执行，第 3 次被拒


# ---------------------------------------------------------------- 契约与防线


async def test_thinking_off_and_tools_passed(db_session):
    llm = ScriptedLLM([{"content": "hi", "tool_calls": None}])
    await _runner(db_session, llm).run("问题")
    call0 = llm.calls[0]
    assert call0["thinking"] is False                    # §1.4 成本护栏
    names = {t["function"]["name"] for t in call0["tools"]}
    assert names == {"search_kb", "calc", "now"}         # §5.3 tools 透传


async def test_reasoning_and_tool_calls_appended(db_session):
    llm = ScriptedLLM([
        {"tool_calls": [_call("now", "{}")], "reasoning": "我要看时间"},
        {"content": "答", "tool_calls": None},
    ])
    await _runner(db_session, llm).run("今天几号")
    second_call_msgs = llm.calls[1]["messages"]
    assistant = next(m for m in second_call_msgs if m["role"] == "assistant")
    assert assistant["tool_calls"]
    assert assistant["reasoning_content"] == "我要看时间"    # §1.1 质量优化回传


async def test_citations_owner_isolation(db_session, seed_document):
    """★ 越权防线：_load_citations 只回本人且 ready 文档的切片。"""
    _, own = await seed_document(owner_id=1, chunks=[("我的资料", None)])
    _, foreign = await seed_document(owner_id=999, chunks=[("别人的资料", None)])
    llm = ScriptedLLM([{"content": "x", "tool_calls": None}])
    runner = _runner(db_session, llm, owner_id=1)
    got = await runner._load_citations([own[0].id, foreign[0].id])
    ids = {c.chunk_id for c in got}
    assert own[0].id in ids
    assert foreign[0].id not in ids


# ---------------------------------------------------------------- 端点接线


async def test_agent_endpoint_records_chatlog(client, auth_headers, db_session):
    r = await client.post(
        "/api/v1/chat/agent",
        json={"question": "你好呀"},
        headers=auth_headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["retrieval_mode"] == "agent"
    assert body["agent_steps"] == 0                    # FakeLLM 不调工具
    assert body["stopped_reason"] is None
    # 落库带 agent 标记
    log = (await db_session.scalars(
        select(ChatLog).where(ChatLog.is_agent.is_(True))
    )).first()
    assert log is not None and log.agent_steps == 0


async def test_agent_endpoint_disabled_returns_404(
    client, auth_headers, monkeypatch
):
    monkeypatch.setattr(settings, "agent_enabled", False)
    r = await client.post(
        "/api/v1/chat/agent", json={"question": "hi"}, headers=auth_headers
    )
    assert r.status_code == 404
