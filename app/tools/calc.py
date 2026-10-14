"""数学计算工具（阶段 5 §4.3）。

★ 安全实现：用 ast 白名单求值，**绝不用 eval()**。
  eval("__import__('os').system('rm -rf /')") 是真实存在的攻击面，
  而这里的表达式来自**模型生成的字符串**——不能信任。

★ 两处实测修正（文档 §1.3）：
  1. 指数在**计算之前**限制（2**100000 会在 float() 转换时抛 OverflowError，
     若只事后判断结果，异常会逃出工具、崩掉 agent 循环）。
  2. except 显式带 OverflowError——它是独立异常类型，不在 (SyntaxError, ValueError,
     TypeError) 之列，漏了就会冒泡成 500。
"""

import ast
import operator
from typing import Any

from app.tools.base import ToolResult, openai_schema

# 只允许这些运算符
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# 指数上限：既防计算爆炸，也让 2**10**10 这类在解析阶段就被拒
_MAX_EXPONENT = 1000


def safe_eval(expr: str) -> float:
    """对算术表达式求值。不允许任何名字、属性访问、函数调用、容器。"""

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            # bool 是 int 的子类，必须显式排除——否则 True 会被当 1 参与运算
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            raise ValueError("只允许数字字面量")
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            # ★ 指数必须在计算之前限制（§1.3）：2**100000 抛的是 OverflowError，
            #   发生在 float() 转换时；提前按常量限制更省、也更早失败。
            if isinstance(node.op, ast.Pow):
                right = node.right
                if not (
                    isinstance(right, ast.Constant)
                    and isinstance(right.value, int)
                    and not isinstance(right.value, bool)
                    and abs(right.value) <= _MAX_EXPONENT
                ):
                    raise ValueError(f"指数过大（上限 {_MAX_EXPONENT}）")
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"不允许的语法：{type(node).__name__}")

    tree = ast.parse(expr, mode="eval")          # 解析失败会抛 SyntaxError
    value = _eval(tree)
    if abs(value) > 1e15:
        raise ValueError("结果超出可计算范围")
    return value


class CalcTool:
    name = "calc"
    description = (
        "对数学表达式求值，支持 + - * / // % ** 和括号。"
        "仅用于需要精确数值结果的场景（折扣后价格、单价乘数量、百分比、差值）。"
        "不要用它查资料——查资料请用 search_kb；也不要传入单位或文字。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "expr": {
                "type": "string",
                "description": "纯数学表达式，例如 (99 * 0.9) + 15",
            }
        },
        "required": ["expr"],
    }

    def to_openai_schema(self) -> dict:
        return openai_schema(self)

    async def run(self, **kwargs: Any) -> ToolResult:
        expr = (kwargs.get("expr") or "").strip()
        if not expr:
            return ToolResult(content="错误：缺少必填参数 expr", is_error=True)
        try:
            value = safe_eval(expr)
        except ZeroDivisionError:
            return ToolResult(content="计算错误：除数为零", is_error=True)
        except (SyntaxError, ValueError, TypeError, OverflowError) as exc:
            # ★ OverflowError 必须显式捕获（§1.3）：漏了它，异常会冒泡出工具
            #   → agent 循环崩 → 用户看到 500。
            return ToolResult(content=f"计算失败：{exc}", is_error=True)
        # 结果同时给"可读值"，避免模型把 89.10000000000001 抄进回答
        return ToolResult(content=f"{expr} = {round(value, 10):g}", meta={"value": value})
