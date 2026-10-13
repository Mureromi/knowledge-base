"""DeepSeek 价格表（元/百万 tokens）。

核实于 2026 年，官方定价页会变动，用前请核对：
https://api-docs.deepseek.com/zh-cn/quick_start/pricing
"""

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class ModelPrice:
    input_cached: float
    input_miss: float
    output: float


# 空闲时段价格（高峰为其 2 倍）
PRICES: dict[str, ModelPrice] = {
    "deepseek-flash": ModelPrice(input_cached=0.02, input_miss=1.0, output=4.0),
    "deepseek-v4-pro": ModelPrice(input_cached=0.15, input_miss=4.5, output=13.5),
}

CST = ZoneInfo("Asia/Shanghai")


def is_peak_now(now: datetime | None = None) -> bool:
    """高峰：北京时间周一至周五 9:00-12:00、14:00-18:00（未处理法定节假日）。"""
    now = now or datetime.now(CST)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return (time(9, 0) <= t < time(12, 0)) or (time(14, 0) <= t < time(18, 0))


def calc_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    peak: bool | None = None,
) -> float:
    price = PRICES.get(model)
    if price is None:
        return 0.0
    peak = is_peak_now() if peak is None else peak
    multiplier = 2.0 if peak else 1.0
    miss = max(prompt_tokens - cached_tokens, 0)
    cost = (
        cached_tokens * price.input_cached
        + miss * price.input_miss
        + completion_tokens * price.output
    ) / 1_000_000
    return round(cost * multiplier, 6)
