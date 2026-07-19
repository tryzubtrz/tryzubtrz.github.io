"""Grid Trading strategy — place layered buy/sell levels around mid price."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

log = logging.getLogger("tryzub.grid")


@dataclass
class GridLevel:
    side: str  # Buy / Sell
    price: float
    qty_pct: float


class GridStrategy:
    name = "grid"

    def __init__(
        self,
        levels: int = 5,
        spacing_pct: float = 0.4,
        enabled: bool = False,
    ) -> None:
        self.levels = max(2, levels)
        self.spacing_pct = spacing_pct
        self.enabled = enabled

    def generate(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Return grid plan + soft signal for ensemble voting."""
        try:
            if not self.enabled or df is None or df.empty:
                return {
                    "symbol": symbol,
                    "direction": "flat",
                    "confidence": 0.0,
                    "strategy": self.name,
                    "grid": [],
                }
            mid = float(df["close"].iloc[-1])
            atr = float((df["high"] - df["low"]).tail(14).mean() or mid * 0.002)
            step = max(mid * (self.spacing_pct / 100.0), atr * 0.5)
            grid: List[Dict[str, Any]] = []
            for i in range(1, self.levels + 1):
                grid.append({"side": "Buy", "price": round(mid - i * step, 6), "qty_pct": 1.0 / self.levels})
                grid.append({"side": "Sell", "price": round(mid + i * step, 6), "qty_pct": 1.0 / self.levels})

            # Soft bias: if price near lower half of recent range → lean long
            lo, hi = float(df["low"].tail(50).min()), float(df["high"].tail(50).max())
            pos = (mid - lo) / (hi - lo + 1e-12)
            if pos < 0.35:
                direction, conf = "long", 0.55 + (0.35 - pos)
            elif pos > 0.65:
                direction, conf = "short", 0.55 + (pos - 0.65)
            else:
                direction, conf = "flat", 0.0

            log.info("Grid %s mid=%.4f levels=%s bias=%s", symbol, mid, len(grid), direction)
            return {
                "symbol": symbol,
                "direction": direction,
                "confidence": float(min(conf, 0.85)),
                "strategy": self.name,
                "grid": grid,
            }
        except Exception as exc:
            log.error("Grid strategy error %s: %s", symbol, exc)
            return {
                "symbol": symbol,
                "direction": "flat",
                "confidence": 0.0,
                "strategy": self.name,
                "grid": [],
                "error": str(exc),
            }
