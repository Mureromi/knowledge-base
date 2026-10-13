"""真实 LLM 冒烟测试（手动跑，不进 CI）。

用法：
    pytest app/tests/manual -m real -s

必须做一次：它是唯一能同时验证 key / base_url / 模型名 / 缓存字段名 / 成本计算的方式。
跑完把打印出来的 usage 结构对照 §6.3 的 `_parse()`，字段名不一致就回填。
"""

import pytest

from app.clients.llm import llm_client


@pytest.mark.real
async def test_real_llm_call():
    await llm_client.startup()
    try:
        r = await llm_client.chat(
            [{"role": "user", "content": "只回复：OK"}], max_tokens=10
        )
        print(f"\n内容={r.content!r}")
        print(
            f"tokens: prompt={r.prompt_tokens} completion={r.completion_tokens} "
            f"cached={r.cached_tokens}"
        )
        print(f"耗时={r.latency_ms}ms 成本={r.cost_yuan}元 高峰={r.peak}")
        print(f"原始 usage 字段：{r.raw.get('usage')}")
    finally:
        await llm_client.shutdown()
