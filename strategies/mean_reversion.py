"""Bollinger + RSI mean reversion strategy."""
from __future__ import annotations

import pandas as pd

from indicators.technical import add_all_indicators
from strategies.base import BaseStrategy, StrategySignal


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"

    def __init__(self, rsi_oversold: float = 30.0, rsi_overbought: float = 70.0):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate(self, symbol: str, df: pd.DataFrame) -> StrategySignal:
        data = add_all_indicators(df)
        row = data.iloc[-1]
        direction = "flat"
        confidence = 0.0
        close = row["close"]

        if close <= row["bb_lower"] and row["rsi_14"] <= self.rsi_oversold:
            direction = "long"
            depth = (self.rsi_oversold - row["rsi_14"]) / 30.0
            confidence = min(0.9, 0.6 + max(depth, 0))
        elif close >= row["bb_upper"] and row["rsi_14"] >= self.rsi_overbought:
            direction = "short"
            depth = (row["rsi_14"] - self.rsi_overbought) / 30.0
            confidence = min(0.9, 0.6 + max(depth, 0))

        return StrategySignal(symbol, direction, float(confidence), self.name)

    def params(self):
        return {"rsi_oversold": self.rsi_oversold, "rsi_overbought": self.rsi_overbought}
