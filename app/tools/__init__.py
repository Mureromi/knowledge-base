"""工具注册表（阶段 5 §4.5）。新增工具只需在这里登记。

★ 注册顺序会影响模型的选择倾向（靠前的更容易被选），
  把最常用的 search_kb 放第一位。
"""

from app.tools.base import Tool, ToolResult, openai_schema
from app.tools.calc import CalcTool
from app.tools.now import NowTool
from app.tools.search_kb import SearchKBTool

__all__ = [
    "Tool",
    "ToolResult",
    "openai_schema",
    "CalcTool",
    "NowTool",
    "SearchKBTool",
    "build_tools",
]


def build_tools(*, db, embedder, reranker, owner_id) -> dict[str, Tool]:
    """构造本次请求可用的工具集。

    ★ 每次请求新建：search_kb 需要请求级的 db session 和 owner_id，
      不能做成全局单例（否则会串用户、串 session）。
    """
    tools = [
        SearchKBTool(db, embedder, reranker, owner_id),
        CalcTool(),
        NowTool(),
    ]
    return {t.name: t for t in tools}
