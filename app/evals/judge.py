"""LLM judge：用 DeepSeek 自己当裁判（阶段 6，docs/项目实现文档-阶段6.md §4.3）。

★ 本模块的存在理由（§1，本阶段最重要的实测结论）：
  DeepEval 的忠实度指标在同一组"已知坏答案"上只判对 1/4（把完全无关、部分编造
  都评为"完全忠实"）。**一个未经验证的指标比没有指标更危险——它给虚假的信心。**
  所以这里自写 judge（逻辑透明、可针对我们的数据分布调 prompt），并且**自带自检**
  `validate_judge()`：用 4 个已知答案（1 好 3 坏）跑一遍，判对率必须 = 1.0，
  否则整份评测结果不可信（结果里带 judge_validated 标记，§5.2）。

★ §2 那个我踩了两次的坑，这里是**规避**而非重演：
  prompt 里含 JSON 示例时若用 `str.format()`，示例的 `{` 会被当占位符 → KeyError。
  本模块采用文档 §2 的"更稳的做法"：占位符用 `<<CTX>>/<<OUT>>`，一律 `.replace()` 注入，
  从根上绕开；因此 JSON 示例保持**单大括号**（发给模型时才是合法的 JSON 样例）。
  test_judge.py 里 `test_judge_prompt_avoids_format_trap` 把这条永久钉住。
"""

import json
import logging
import re

from app.clients.llm import llm_client
from app.core.config import settings

logger = logging.getLogger(__name__)

# JSON 示例用单大括号（不经 .format，无需转义）；占位符是 <<...>>，用 .replace 注入。
FAITHFULNESS_PROMPT = """你是严格的 RAG 答案校验员。判断【回答】是否完全由【参考资料】支撑。

规则：
1. 回答中任何参考资料里没有的事实、数字、条件、承诺，都算不忠实。
2. 回答与参考资料矛盾，算不忠实。
3. 只做判断，不要自行补充知识。
4. 输出 JSON 对象，含两个字段：
   - "faithful"：布尔值
   - "reason"：一句话说明

示例输出：{"faithful": false, "reason": "回答提到的 30 天到账在资料中无依据"}

【参考资料】
<<CTX>>

【回答】
<<OUT>>"""

COMPLETENESS_PROMPT = """你是严格的 RAG 答案校验员。判断【回答】是否**完整覆盖**了【要点】。

规则：
1. 逐条核对【要点】是否在【回答】中体现（同义表述也算）。
2. 输出 JSON 对象，含三个字段：
   - "covered"：被覆盖的要点编号列表（从 1 开始）
   - "missing"：未被覆盖的要点编号列表
   - "reason"：一句话说明

示例输出：{"covered": [1], "missing": [2], "reason": "第 2 点未提及"}

【要点】
<<POINTS>>

【回答】
<<OUT>>"""


def _safe_json(text: str) -> dict:
    """宽松解析模型返回的 JSON：直接 loads → 抽取首个 {...} 块 → 兜底空 dict。

    §7.1 要求"喂入各种模型输出（缺字段/多字段/非法 JSON）不崩，返回合理默认值"。
    即使声明了 response_format=json_object，模型偶尔仍会裹 markdown 代码块或加前后缀，
    这里必须容错——评测不能因为一次解析失败就整体崩掉。
    """
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def build_faithfulness_prompt(answer: str, contexts: list[str]) -> str:
    return (
        FAITHFULNESS_PROMPT
        .replace("<<CTX>>", "\n".join(contexts) or "（无参考资料）")
        .replace("<<OUT>>", answer)
    )


def build_completeness_prompt(answer: str, points: list[str]) -> str:
    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(points, start=1))
    return (
        COMPLETENESS_PROMPT
        .replace("<<POINTS>>", numbered)
        .replace("<<OUT>>", answer)
    )


class Judge:
    """裁判。llm 可注入（§7.2①：把响应做成可喂固定 JSON 的，用于确定性测计算逻辑）；
    不注入时复用项目的 `llm_client`（唯一的 LLM 出口，阶段 0 定的约定）。

    ★ 实现说明：文档 §4.3 伪代码直连 `openai.AsyncOpenAI`，但本项目**没有 openai 依赖**
      （llm.py 是手写 httpx 客户端）——所以这里走 llm_client.chat(json_mode=True)，
      重试/成本计算/思考关闭全部复用现有实现，不另起炉灶。
    """

    def __init__(self, llm: object | None = None) -> None:
        self._llm = llm
        self.model = settings.eval_judge_model or settings.llm_model
        self.calls = 0            # 便于统计评测成本
        self.cost_yuan = 0.0      # judge 自己的花费也要能算（§5.2 cost_yuan）

    @property
    def _client(self):
        # 懒取全局单例：避免 import 时就要求 lifespan 已 startup
        return self._llm if self._llm is not None else llm_client

    async def _ask_json(self, prompt: str) -> dict:
        self.calls += 1
        result = await self._client.chat(
            [{"role": "user", "content": prompt}],
            temperature=settings.eval_judge_temperature,   # ★ 评测必须确定性（§4.3）
            max_tokens=500,
            json_mode=True,                                 # 走现有 response_format 通道
            thinking=False,                                 # 评测不需要思考（省 75% 成本）
        )
        self.cost_yuan += float(getattr(result, "cost_yuan", 0.0) or 0.0)
        return _safe_json(getattr(result, "content", "") or "")

    async def faithfulness(self, answer: str, contexts: list[str]) -> dict:
        """答案是否有资料支撑。返回 {"score": 0/1, "reason": str}。"""
        data = await self._ask_json(build_faithfulness_prompt(answer, contexts))
        return {
            "score": 1.0 if data.get("faithful") else 0.0,
            "reason": str(data.get("reason", "")),
        }

    async def completeness(self, answer: str, points: list[str]) -> dict:
        """是否覆盖了标注要点。返回 {"score": 0~1, "missing": [...], "reason": str}。"""
        data = await self._ask_json(build_completeness_prompt(answer, points))
        covered = len(data.get("covered") or [])
        total = max(len(points), 1)
        missing_idx = [i for i in (data.get("missing") or []) if isinstance(i, int)]
        return {
            "score": covered / total,
            "missing": [points[i - 1] for i in missing_idx if 1 <= i <= len(points)],
            "reason": str(data.get("reason", "")),
        }


# ---------------------------------------------------------------------------
# 自检：把 §1.1 的人工发现变成可重复执行的检查（"先证明尺子是准的"）
# ---------------------------------------------------------------------------
SELF_CHECK_CASES: list[tuple[str, list[str], bool]] = [
    # (答案, 资料, 期望是否忠实) —— 1 好 + 3 坏（§8 验收项）
    ("审核通过后，款项将在 5-7 个工作日退回原支付渠道。",
     ["审核通过后，款项将在 5-7 个工作日退回原支付渠道。"], True),
    ("巴黎是法国首都，人口约 214 万。",
     ["审核通过后，款项将在 5-7 个工作日退回原支付渠道。"], False),
    ("退款需要 30 天才能到账，并且要收 5% 手续费。",
     ["审核通过后，款项将在 5-7 个工作日退回原支付渠道。"], False),
    ("审核通过后 5-7 个工作日到账，另外需要支付 20 元提现费。",
     ["审核通过后，款项将在 5-7 个工作日退回原支付渠道。"], False),
]


async def validate_judge(judge: Judge) -> float:
    """★ 用已知答案验证 judge 是否可用。返回判对率；< 1.0 时评测结果不可信。

    为什么必须有它：DeepEval 的忠实度在同一组用例上只判对 1/4（§1.1），而它看起来
    完全正常。**没有这个自检，你会拿一把坏尺子量很久。**
    """
    correct = 0
    for answer, ctx, expect_faithful in SELF_CHECK_CASES:
        result = await judge.faithfulness(answer, ctx)
        got_faithful = result["score"] == 1.0
        if got_faithful == expect_faithful:
            correct += 1
        else:
            logger.error(
                "judge 自检失败：答案=%r 期望忠实=%s 实际=%s（理由：%s）",
                answer[:30], expect_faithful, got_faithful, result["reason"][:60],
            )
    return correct / len(SELF_CHECK_CASES)
