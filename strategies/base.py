"""Base strategy interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class StrategySignal:
    symbol: str
    direction: str  # long / short / flat
    confidence: float
    strategy: str
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": float(self.confidence),
            "strategy": self.strategy,
            "meta": self.meta or {},
        }


class BaseStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, symbol: str, df: pd.DataFrame) -> StrategySignal:
        raise NotImplementedError

    def params(self) -> Dict[str, Any]:
        return {}

    def set_params(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
