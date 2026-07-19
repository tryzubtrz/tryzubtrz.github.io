"""Short-term scalping using EMA cross + stochastic."""
from __future__ import annotations

import pandas as pd

from indicators.technical import add_all_indicators
from strategies.base import BaseStrategy, StrategySignal


class ScalpingStrategy(BaseStrategy):
    name = "scalping"

    def __init__(self, stoch_low: float = 25.0, stoch_high: float = 75.0):
        self.stoch_low = stoch_low
        self.stoch_high = stoch_high

    def generate(self, symbol: str, df: pd.DataFrame) -> StrategySignal:
        data = add_all_indicators(df)
        if len(data) < 3:
            return StrategySignal(symbol, "flat", 0.0, self.name)
        prev, row = data.iloc[-2], data.iloc[-1]
        direction = "flat"
        confidence = 0.0

        bullish_cross = prev["ema_9"] <= prev["ema_21"] and row["ema_9"] > row["ema_21"]
        bearish_cross = prev["ema_9"] >= prev["ema_21"] and row["ema_9"] < row["ema_21"]

        if bullish_cross and row["stoch_k"] < self.stoch_high:
            direction = "long"
            confidence = 0.65 + (0.1 if row["stoch_k"] < self.stoch_low else 0.0)
        elif bearish_cross and row["stoch_k"] > self.stoch_low:
            direction = "short"
            confidence = 0.65 + (0.1 if row["stoch_k"] > self.stoch_high else 0.0)

        return StrategySignal(symbol, direction, float(confidence), self.name)

    def params(self):
        return {"stoch_low": self.stoch_low, "stoch_high": self.stoch_high}
