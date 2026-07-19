"""EMA / ADX trend-following strategy."""
from __future__ import annotations

import pandas as pd

from indicators.technical import add_all_indicators
from strategies.base import BaseStrategy, StrategySignal


class TrendFollowingStrategy(BaseStrategy):
    name = "trend_following"

    def __init__(self, adx_min: float = 20.0, rsi_long_max: float = 70.0, rsi_short_min: float = 30.0):
        self.adx_min = adx_min
        self.rsi_long_max = rsi_long_max
        self.rsi_short_min = rsi_short_min

    def generate(self, symbol: str, df: pd.DataFrame) -> StrategySignal:
        data = add_all_indicators(df)
        row = data.iloc[-1]
        direction = "flat"
        confidence = 0.0

        bullish = (
            row["ema_9"] > row["ema_21"] > row["ema_55"]
            and row["adx_14"] >= self.adx_min
            and row["rsi_14"] < self.rsi_long_max
            and row["macd_hist"] > 0
        )
        bearish = (
            row["ema_9"] < row["ema_21"] < row["ema_55"]
            and row["adx_14"] >= self.adx_min
            and row["rsi_14"] > self.rsi_short_min
            and row["macd_hist"] < 0
        )
        if bullish:
            direction = "long"
            confidence = min(0.95, 0.55 + (row["adx_14"] / 100) + min(abs(row["macd_hist"]) * 10, 0.2))
        elif bearish:
            direction = "short"
            confidence = min(0.95, 0.55 + (row["adx_14"] / 100) + min(abs(row["macd_hist"]) * 10, 0.2))

        return StrategySignal(symbol, direction, float(confidence), self.name)

    def params(self):
        return {
            "adx_min": self.adx_min,
            "rsi_long_max": self.rsi_long_max,
            "rsi_short_min": self.rsi_short_min,
        }
