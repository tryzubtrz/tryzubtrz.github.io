"""Weighted ensemble of rule-based strategies (+ optional ML vote)."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from strategies.base import BaseStrategy, StrategySignal
from strategies.breakout import BreakoutStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.scalping import ScalpingStrategy
from strategies.trend_following import TrendFollowingStrategy


class EnsembleStrategy(BaseStrategy):
    name = "ensemble"

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.strategies: List[BaseStrategy] = [
            TrendFollowingStrategy(),
            MeanReversionStrategy(),
            BreakoutStrategy(),
            ScalpingStrategy(),
        ]
        self.weights = weights or {
            "trend_following": 0.35,
            "mean_reversion": 0.25,
            "breakout": 0.25,
            "scalping": 0.15,
        }
        self.ml_vote: Optional[StrategySignal] = None
        self.ml_weight = 0.4

    def set_ml_vote(self, signal: Optional[StrategySignal]) -> None:
        self.ml_vote = signal

    def generate(self, symbol: str, df: pd.DataFrame) -> StrategySignal:
        votes = {"long": 0.0, "short": 0.0, "flat": 0.0}
        details = []
        for strat in self.strategies:
            sig = strat.generate(symbol, df)
            w = self.weights.get(strat.name, 0.2)
            votes[sig.direction] += w * sig.confidence
            details.append(sig.to_dict())

        if self.ml_vote is not None:
            votes[self.ml_vote.direction] += self.ml_weight * self.ml_vote.confidence
            details.append(self.ml_vote.to_dict())

        direction = max(votes, key=votes.get)
        confidence = votes[direction]
        # Normalize roughly into [0, 1]
        total = sum(votes.values()) + 1e-12
        confidence = min(0.99, confidence / max(total, 1.0) * (1 if direction != "flat" else 0.3))
        if direction != "flat":
            confidence = max(confidence, votes[direction] / (sum(self.weights.values()) + self.ml_weight))

        return StrategySignal(
            symbol,
            direction if confidence >= 0.35 else "flat",
            float(confidence if direction != "flat" else 0.0),
            self.name,
            meta={"votes": votes, "children": details},
        )

    def get_child(self, name: str) -> Optional[BaseStrategy]:
        for s in self.strategies:
            if s.name == name:
                return s
        return None
