"""当前时间工具（阶段 5 §4.4）。

为什么需要它：LLM 的训练数据有截止日期，**它不知道"今天"是哪天**。
一旦用户问"最近的""今年新出的"，模型只能猜。给一个时间工具能消除这类错误。
"""

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.tools.base import ToolResult, openai_schema

CST = ZoneInfo("Asia/Shanghai")


class NowTool:
    name = "now"
    description = (
        "获取当前日期与时间（北京时间）。"
        "当问题涉及「今天」「最近」「今年」等相对时间时使用。"
        "不要用它做计算或查资料。"
    )
    parameters = {"type": "object", "properties": {}}

    def to_openai_schema(self) -> dict:
        return openai_schema(self)

    async def run(self, **kwargs: Any) -> ToolResult:
        now = datetime.now(CST)
        weekday = "一二三四五六日"[now.weekday()]
        return ToolResult(
            content=f"现在是 {now.strftime('%Y-%m-%d %H:%M')}（北京时间，星期{weekday}）",
            meta={"iso": now.isoformat()},
        )
