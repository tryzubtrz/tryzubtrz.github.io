"""Volume-confirmed range breakout strategy."""
from __future__ import annotations

import pandas as pd

from indicators.technical import add_all_indicators
from strategies.base import BaseStrategy, StrategySignal


class BreakoutStrategy(BaseStrategy):
    name = "breakout"

    def __init__(self, lookback: int = 20, volume_mult: float = 1.5):
        self.lookback = lookback
        self.volume_mult = volume_mult

    def generate(self, symbol: str, df: pd.DataFrame) -> StrategySignal:
        data = add_all_indicators(df)
        if len(data) < self.lookback + 2:
            return StrategySignal(symbol, "flat", 0.0, self.name)

        window = data.iloc[-(self.lookback + 1) : -1]
        row = data.iloc[-1]
        high = window["high"].max()
        low = window["low"].min()
        direction = "flat"
        confidence = 0.0

        vol_ok = row["volume_ratio"] >= self.volume_mult
        if row["close"] > high and vol_ok:
            direction = "long"
            breakout_strength = (row["close"] - high) / (high + 1e-12)
            confidence = min(0.92, 0.6 + breakout_strength * 20 + min(row["volume_ratio"] / 10, 0.15))
        elif row["close"] < low and vol_ok:
            direction = "short"
            breakout_strength = (low - row["close"]) / (low + 1e-12)
            confidence = min(0.92, 0.6 + breakout_strength * 20 + min(row["volume_ratio"] / 10, 0.15))

        return StrategySignal(symbol, direction, float(confidence), self.name)

    def params(self):
        return {"lookback": self.lookback, "volume_mult": self.volume_mult}
