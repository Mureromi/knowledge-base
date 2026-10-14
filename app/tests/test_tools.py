"""阶段 5 工具层单测（docs/项目实现文档-阶段5.md §8.3 / §8.4）。

纯函数优先：calc / now / to_openai_schema 不需要数据库；
search_kb 用真实内存库 + 假 embedder/reranker，并覆盖"错误当结果返回、不冒泡"。
"""

import pytest

from app.services.retrieval import RetrievedChunk
from app.tools import build_tools
from app.tools.base import ToolResult
from app.tools.calc import CalcTool, safe_eval
from app.tools.now import NowTool
from app.tools.search_kb import SearchKBTool


# ---------------------------------------------------------------- calc（§8.3）


class TestCalcNormal:
    async def test_basic_arithmetic(self):
        assert (await CalcTool().run(expr="2 + 3")).content == "2 + 3 = 5"

    async def test_discount_expression(self):
        r = await CalcTool().run(expr="(99 * 0.9) + 15")
        assert r.is_error is False
        assert "104.1" in r.content

    async def test_result_not_noisy_float(self):
        # 89.10000000000001 这类浮点噪声必须被 :g 收敛掉
        r = await CalcTool().run(expr="99 * 0.9")
        assert "0.900000000" not in r.content


class TestCalcSafety:
    """__import__ / open / 下标 / 条件表达式 / 名字 全部被拒（§8.3）。"""

    @pytest.mark.parametrize("expr", [
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "[1,2,3][0]",
        "1 if True else 2",
        "x + 1",                 # 自由变量
        "True + 1",              # bool 不是数字字面量
        "'abc'",                 # 字符串
        "abs(-1)",               # 函数调用
    ])
    async def test_dangerous_rejected_as_error(self, expr):
        r = await CalcTool().run(expr=expr)
        assert r.is_error is True
        assert "计算失败" in r.content


class TestCalcOverflow:
    """★ §1.3 实测的核心边界：这些必须返回可读错误，绝不能让 OverflowError 逃出工具。"""

    @pytest.mark.parametrize("expr", [
        "2 ** 1000",
        "2 ** 100000",
        "2 ** 10 ** 10",
        "9 ** 9 ** 9",
    ])
    async def test_large_exponent_no_overflow_escape(self, expr):
        # 直接调 safe_eval：OverflowError 若逃出，这里会炸而不是抛 ValueError
        with pytest.raises(ValueError):
            safe_eval(expr)
        # 走工具：转成可读错误，绝不 is_error 之外的异常
        r = await CalcTool().run(expr=expr)
        assert r.is_error is True

    async def test_division_by_zero(self):
        r = await CalcTool().run(expr="1 / 0")
        assert r.is_error is True
        assert "除数为零" in r.content

    async def test_empty_expr(self):
        r = await CalcTool().run(expr="")
        assert r.is_error is True


# ---------------------------------------------------------------- now（§8.3）


class TestNow:
    async def test_beijing_time_and_weekday(self):
        r = await NowTool().run()
        assert "北京时间" in r.content
        assert "星期" in r.content
        assert "T" in r.meta["iso"]           # iso8601


# ---------------------------------------------------------------- schema（§8.3）


class TestSchema:
    def test_all_tools_schema_valid(self):
        tools = build_tools(db=None, embedder=None, reranker=None, owner_id=1)
        for tool in tools.values():
            s = tool.to_openai_schema()
            assert s["type"] == "function"
            fn = s["function"]
            assert fn["name"] == tool.name
            assert fn["parameters"]["type"] == "object"
            # required 必须是 properties 的子集
            props = set(fn["parameters"].get("properties", {}))
            for req in fn["parameters"].get("required", []):
                assert req in props


# ---------------------------------------------------------------- §8.4 反脆弱


def test_tool_descriptions_state_when_not_to_use():
    """★ §1.2 实测：工具描述重叠会导致错调。
    强制每个工具都声明禁用场景——这是防"新增工具时忘记写边界"的机械检查。"""
    for tool in build_tools(db=None, embedder=None, reranker=None, owner_id=1).values():
        assert "不要" in tool.description or "仅在" in tool.description, (
            f"工具 {tool.name} 的描述缺少使用边界声明（§1.2：描述重叠会导致错调）"
        )


# ---------------------------------------------------------------- search_kb


class TestSearchKB:
    async def test_empty_query_is_readable_error(self):
        tool = SearchKBTool(db=None, embedder=None, reranker=None, owner_id=1)
        r = await tool.run(query="   ")
        assert r.is_error is True
        assert "query" in r.content

    async def test_search_exception_does_not_bubble(self, monkeypatch):
        """检索内部抛异常 → 转成可读错误，绝不冒泡（§1.2 失败模式 2 的反面）。"""
        async def _boom(*a, **k):
            raise RuntimeError("索引挂了")

        monkeypatch.setattr("app.tools.search_kb.search", _boom)
        tool = SearchKBTool(db=None, embedder=None, reranker=None, owner_id=1)
        r = await tool.run(query="退款")
        assert r.is_error is True
        assert "检索失败" in r.content

    async def test_no_hits_message(self, monkeypatch):
        async def _empty(*a, **k):
            return []

        monkeypatch.setattr("app.tools.search_kb.search", _empty)
        tool = SearchKBTool(db=None, embedder=None, reranker=None, owner_id=1)
        r = await tool.run(query="无关词")
        assert r.is_error is False
        assert "没有找到" in r.content
        assert r.meta["hits"] == 0

    async def test_long_result_truncated(self, monkeypatch):
        async def _many(*a, **k):
            return [
                RetrievedChunk(
                    chunk_id=i, document_id=1, document_title="长文档",
                    seq=i, content="内" * 5000, score=1.0,
                )
                for i in range(3)
            ]

        monkeypatch.setattr("app.tools.search_kb.search", _many)
        tool = SearchKBTool(db=None, embedder=None, reranker=None, owner_id=1)
        r = await tool.run(query="退款")
        from app.core.config import settings
        assert "已截断" in r.content
        assert len(r.content) <= settings.agent_tool_result_max_chars + len("\n…（已截断）")

    async def test_citations_accumulate_across_calls(self, monkeypatch):
        """多跳问答：两次检索的 chunk_ids 去重累加，不是覆盖（本阶段对文档伪代码的改进）。"""

        def _mk(ids):
            async def _s(*a, **k):
                return [
                    RetrievedChunk(chunk_id=cid, document_id=1, document_title="d",
                                   seq=cid, content="内容", score=1.0)
                    for cid in ids
                ]
            return _s

        tool = SearchKBTool(db=None, embedder=None, reranker=None, owner_id=1)
        monkeypatch.setattr("app.tools.search_kb.search", _mk([]))
        await tool.run(query="a")
        assert tool.last_chunk_ids == []
        monkeypatch.setattr("app.tools.search_kb.search", _mk([3, 1, 2]))
        await tool.run(query="b")
        assert tool.last_chunk_ids == [3, 1, 2]
        # 再检索一次含重叠：去重且保序
        monkeypatch.setattr("app.tools.search_kb.search", _mk([2, 9]))
        await tool.run(query="c")
        assert tool.last_chunk_ids == [3, 1, 2, 9]

    async def test_result_is_tool_result_type(self, monkeypatch):
        async def _empty(*a, **k):
            return []

        monkeypatch.setattr("app.tools.search_kb.search", _empty)
        tool = SearchKBTool(db=None, embedder=None, reranker=None, owner_id=1)
        assert isinstance(await tool.run(query="x"), ToolResult)
