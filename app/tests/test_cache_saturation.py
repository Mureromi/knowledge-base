"""★ 滑动窗口边界下的缓存行为测试（补 test_multiturn_cache.py 的缺口）。

## 本文件修正了一个错误认知（重要）

最初的假设是"窗口饱和后，紧接着重复同一问题会命中缓存"。
**实测证明这个假设是错的。** 滑动窗口是**固定容量队列**（容量 = 轮数 × 2 = 12 条），
不是累积列表：每轮问答把 2 条新消息推入队尾、同时把 2 条旧消息挤出队首。

后果：每问一次，窗口内容都变 → 历史指纹每轮都不同 → **会话内永远不可能命中缓存**。

| 场景 | 历史指纹 | 是否命中 |
| --- | --- | --- |
| 全新会话（不传 session_id，窗口恒为空） | 恒定 | ✅ 命中（`test_multiturn_cache.py` 已覆盖） |
| 会话内重复同一问题（第 2 轮起） | 每轮都变 | ❌ 不命中 |
| 会话内重复同一问题（窗口已饱和） | **仍然每轮都变**（队列移位） | ❌ 不命中 |

**这不是 bug，是"历史进缓存键"这个设计决策的直接推论**：
键覆盖了实际发给 LLM 的上下文，而上下文每轮都在变，所以键每轮都新。
设计上是安全优先（宁可多花钱，不可串上下文）。

## 那这个文件测什么

既然"会话内命中"不可能发生，就不能写一条断言它必须命中的测试（那是自欺）。
本文件改为锁死三条**真实且重要**的不变量：
1. 会话内重复 → 必须未命中（若命中，说明缓存键丢了历史，会串上下文）；
2. 会话内重复 → 必须带上历史重新生成（否则就是忽略上下文的错误答案）；
3. 流式与非流式必须使用同一套键（谁改了一边立刻红）。

> 已知优化机会（未实现，见文末 §优化机会）：同一窗口里连续重复同一问题时，
> 窗口内容其实是相同的（只是位置移位），缓存本可命中。当前实现放弃了这部分收益。
"""

import pytest
from httpx import AsyncClient

from app.core.config import settings
from app.tests.conftest import FakeLLM, parse_sse

WINDOW_CAPACITY = settings.session_window_turns * 2   # 12 条消息
# 灌满窗口所需轮数：容量 12 条 = 6 轮；+1 轮确保队列真正进入滑动状态
SATURATE_TURNS = settings.session_window_turns + 1

PROBE_QUESTION = "退款要多久到账？"


async def _upload(client: AsyncClient, headers: dict, content: str) -> None:
    r = await client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": ("policy.md", content.encode("utf-8"), "text/plain")},
    )
    # 阶段 3 §5.1：上传改为 202 Accepted（异步）。txt/md 快路径历史上是 201，
    # 两者都在语义上表示"文档已创建"，故都接受；真正的错误码仍会被这条断言挡下。
    assert r.status_code in (201, 202), r.text


async def _ask(
    client: AsyncClient, headers: dict, question: str, session_id: int | None = None
):
    body: dict = {"question": question}
    if session_id is not None:
        body["session_id"] = session_id
    r = await client.post("/api/v1/chat/sync", headers=headers, json=body)
    assert r.status_code == 200, r.text
    return r.json()


async def _saturate(client: AsyncClient, headers: dict) -> int:
    """把会话的滑动窗口灌满（第 7 轮起队列开始滑动），返回 session_id。"""
    sid = None
    for i in range(SATURATE_TURNS):
        out = await _ask(client, headers, f"退款问题{i}？", session_id=sid)
        sid = out["session_id"]
    return sid


@pytest.mark.asyncio
async def test_saturated_window_repeat_still_misses_cache(
    client, auth_headers, fake_llm: FakeLLM
):
    """★ 窗口饱和后重复同一问题 → **仍然未命中**（队列移位导致指纹变化）。

    这条测试的价值在于：它锁死了"历史指纹真的参与了键计算"。
    阶段 1 的键（问题 + 检索指纹）在此场景下**会命中**，而阶段 2 的键不会——
    所以一旦有人把 `window_messages` 参数从 `answer_cache_key` 里去掉，
    这条测试立刻变红。**这是唯一能发现该退化的测试。**

    （`test_multiturn_cache.py::test_repeated_question_across_fresh_sessions_hits_cache`
     用的是空窗口，退化后照样命中，发现不了这个问题。）
    """
    await _upload(
        client, auth_headers, "退款流程：提交申请后客服 3 个工作日审核，通过后 5-7 个工作日到账。"
    )
    sid = await _saturate(client, auth_headers)

    first = await _ask(client, auth_headers, PROBE_QUESTION, session_id=sid)
    assert first["usage"]["cache_hit"] is False

    calls_before = len(fake_llm.calls)
    second = await _ask(client, auth_headers, PROBE_QUESTION, session_id=sid)

    assert second["usage"]["cache_hit"] is False, (
        f"窗口容量 {WINDOW_CAPACITY} 条，每轮推入 2 条并挤出 2 条，历史指纹每轮都变，"
        "因此会话内重复提问必须未命中。若此处命中，说明缓存键忽略了会话历史——"
        "那会把上一个上下文的答案串到当前上下文（阶段 2 §4.1 明确禁止）。"
    )
    assert len(fake_llm.calls) == calls_before + 1, "未命中就必须真的调一次 LLM"


@pytest.mark.asyncio
async def test_saturated_window_repeat_carries_history(
    client, auth_headers, fake_llm: FakeLLM
):
    """★ 会话内重复提问虽然不命中，但**必须携带历史**重新生成。

    这条防的是另一种错误：为了"提高命中率"而把历史从 messages 里去掉。
    那样虽然会命中缓存，但答案会丢失上下文（"那要多久？"这类追问直接失效）。
    """
    await _upload(
        client, auth_headers, "退款流程：提交申请后客服 3 个工作日审核，通过后 5-7 个工作日到账。"
    )
    sid = await _saturate(client, auth_headers)
    await _ask(client, auth_headers, PROBE_QUESTION, session_id=sid)

    # 第二次请求发给 LLM 的 messages：system + 窗口历史 + 本轮
    msgs = fake_llm.calls[-1]["messages"]
    assert msgs[0]["role"] == "system", "system 必须恒在最前（阶段 0 缓存友好设计）"
    assert msgs[-1]["role"] == "user", "最后一条必须是本轮问题"
    assert len(msgs) > 2, (
        "第二次请求必须带上窗口历史（system + 历史 + 本轮），否则追问会丢失上下文"
    )
    # 窗口已饱和 = 12 条历史 + system + 本轮 = 14 条
    assert len(msgs) == WINDOW_CAPACITY + 2, (
        f"饱和窗口应携带 {WINDOW_CAPACITY} 条历史，实际 {len(msgs) - 2} 条"
    )
    # 注意：这里**不应**断言 PROBE_QUESTION 出现在历史里——
    # 窗口饱和时首个探针问答会被立即挤出（这正是滑动窗口的语义）。
    # 断言历史非空且都是 user/assistant 交替即可。
    assert len(history_text := [m["role"] for m in msgs[1:-1]]) == WINDOW_CAPACITY
    assert set(history_text) <= {"user", "assistant"}


@pytest.mark.asyncio
async def test_stream_uses_session_window_fingerprint(
    client, auth_headers, fake_llm: FakeLLM
):
    """★ 流式路径必须真的把会话窗口纳入缓存键（防两条路径键不一致）。

    ## 构造思路（这条测试经过变异测试验证）

    在**有窗口**的会话里连续两次问同一问题：
      - 第 1 次：未命中 → 写入缓存（键含当时的窗口）
      - 第 2 次：窗口已移位 → 未命中 → **再写一份新键**

    如果流式路径漏传了 window（例如把 `window=window` 写成 `window=None`），
    它算出的键会等于"空窗口"的键——而空窗口的键恰好是**第 1 次写入的那份**，
    于是第 2 次会错误地命中，本条立刻变红。

    ⚠️ 注意：**不能**写成"先 sync 再 stream 比较 cache_hit 是否相同"。
    那种写法在两条路径都返回 False 时也会通过，抓不到键不一致
    （本文件初版就犯了这个错，变异测试时暴露出来）。
    """
    await _upload(
        client, auth_headers, "退款流程：提交申请后客服 3 个工作日审核，通过后 5-7 个工作日到账。"
    )
    sid = await _saturate(client, auth_headers)

    async def stream_hit(question: str) -> bool:
        r = await client.post(
            "/api/v1/chat",
            headers=auth_headers,
            json={"question": question, "session_id": sid},
        )
        assert r.status_code == 200, r.text
        return dict(parse_sse(r.text))["meta"]["cache_hit"]

    assert await stream_hit(PROBE_QUESTION) is False, "第 1 次必然未命中"
    assert await stream_hit(PROBE_QUESTION) is False, (
        "第 2 次：窗口已移位，流式路径必须同样未命中。"
        "若此处命中，说明流式路径没有把会话窗口传给 answer_cache_key——"
        "它的键退化成了'空窗口键'，会与其他上下文串答案。"
    )


@pytest.mark.asyncio
async def test_stream_hit_does_not_call_llm_in_fresh_session(
    client, auth_headers, fake_llm: FakeLLM
):
    """流式在**空窗口**场景下能命中缓存，且命中时一次 LLM 都不调。

    与上面几条互为补充：会话内不命中，但无会话（空窗口）时必须命中——
    这证明缓存功能本身是好的，只是被"历史进键"的设计限制在特定场景。
    """
    await _upload(client, auth_headers, "退款流程：提交申请后客服 3 个工作日审核。")

    # 非流式先灌缓存（不传 session_id，窗口恒为空）
    first = await _ask(client, auth_headers, PROBE_QUESTION)
    assert first["usage"]["cache_hit"] is False

    chat_before = len(fake_llm.calls)
    stream_before = len(fake_llm.stream_calls)

    r = await client.post(
        "/api/v1/chat", headers=auth_headers, json={"question": PROBE_QUESTION}
    )
    assert r.status_code == 200, r.text
    events = parse_sse(r.text)
    meta = dict(events)["meta"]
    assert meta["cache_hit"] is True, "空窗口下流式必须能命中非流式写入的缓存"

    deltas = "".join(data["text"] for name, data in events if name == "delta")
    assert deltas, "命中缓存也要逐块推送答案，不能只发 meta"

    assert len(fake_llm.calls) == chat_before, "命中时不得调用 chat()"
    assert len(fake_llm.stream_calls) == stream_before, "命中时不得调用 chat_stream()"


# ---------------------------------------------------------------------------
# 优化机会（本阶段未实现，记录在此避免遗忘）
#
# 现象：在饱和窗口里连续问同一个问题 4 次，第 3、4 次的窗口内容**其实完全相同**
#       （都是同一组 Q/A 重复 6 遍），只是位置发生了移位，导致指纹不同、缓存未命中。
#
# 可选的改法：把 `history_fingerprint` 改成对窗口内容做**规范化去重**
#       （例如把连续重复的相同 (role, content) 压缩），使"内容相同但位置不同"
#       的窗口得到同一指纹。这样连续的重复提问就能命中缓存。
#
# 为什么现在不做：
#   1. 需要先有真实的缓存命中率数据，才能判断这笔优化值不值得；
#   2. 规范化会放宽键的约束，必须配套更严格的"不串上下文"测试。
#
# 触发条件：真实使用中统计到"同一会话内高频重复提问"占比较高时再做。
# ---------------------------------------------------------------------------
