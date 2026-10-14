"""工具的统一契约（阶段 5 §4.1）。

★ 三条设计原则（都由实测得出，见文档 §1.2）：
1. **错误是结果，不是异常**：工具内部异常必须转成 ToolResult(is_error=True) 返回，
   让模型读懂并自我修正（实测：模型确实会修正）。绝不让异常冒泡到 agent 循环。
2. **描述必须互斥且写明禁用场景**：实测模型会因描述重叠而错调工具（§1.2 失败模式 1）。
3. **返回值要有长度上限**：工具结果会进 context，超长会挤掉真正的参考资料。
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolResult:
    """工具执行结果。永远不会抛异常给 agent 循环。"""

    content: str                      # 给模型看的文本
    is_error: bool = False
    # 供 chat_logs.agent_tools 统计与 trace 使用
    meta: dict[str, Any] | None = None


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str                  # ★ 必须写明"什么时候不要用"
    parameters: dict                  # JSON Schema

    async def run(self, **kwargs: Any) -> ToolResult: ...

    def to_openai_schema(self) -> dict:
        ...


def openai_schema(tool: Tool) -> dict:
    """把工具转成 OpenAI/DeepSeek Function Calling 的 tools 数组元素。

    做成模块级函数而非 Protocol 默认方法：Protocol 的默认实现不会被"结构化子类"
    自动继承（各工具类没有显式继承 Tool），所以统一在这里组装，避免每个工具重复写。
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }
