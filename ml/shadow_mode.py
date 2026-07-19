"""
Shadow Mode: run a candidate strategy in parallel without placing orders.
Track hypothetical PnL and recommend switch when it outperforms live.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import DATA_DIR
from strategies.base import StrategySignal
from strategies.ensemble import EnsembleStrategy

logger = logging.getLogger("ai.shadow")
STATE_PATH = DATA_DIR / "shadow_state.json"


@dataclass
class ShadowTrade:
    symbol: str
    side: str
    entry: float
    qty: float = 1.0


@dataclass
class ShadowState:
    equity: float = 0.0
    trades: int = 0
    wins: int = 0
    open: Dict[str, ShadowTrade] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)


class ShadowMode:
    def __init__(self, strategy: Optional[EnsembleStrategy] = None) -> None:
        self.strategy = strategy or EnsembleStrategy()
        self.state = ShadowState()
        self.load()

    def load(self) -> None:
        if STATE_PATH.exists():
            try:
                raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
                self.state.equity = float(raw.get("equity", 0))
                self.state.trades = int(raw.get("trades", 0))
                self.state.wins = int(raw.get("wins", 0))
                self.state.history = raw.get("history", [])[-200:]
            except Exception:
                logger.warning("Failed to load shadow state")

    def save(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(
            json.dumps(
                {
                    "equity": self.state.equity,
                    "trades": self.state.trades,
                    "wins": self.state.wins,
                    "history": self.state.history[-200:],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def step(self, symbol: str, df: pd.DataFrame, price: float) -> StrategySignal:
        sig = self.strategy.generate(symbol, df)
        open_pos = self.state.open.get(symbol)

        if open_pos is None:
            if sig.direction in {"long", "short"} and sig.confidence >= 0.6:
                side = "Buy" if sig.direction == "long" else "Sell"
                self.state.open[symbol] = ShadowTrade(symbol=symbol, side=side, entry=price)
        else:
            direction = 1 if open_pos.side == "Buy" else -1
            pnl_pct = (price - open_pos.entry) / open_pos.entry * direction
            should_exit = (
                sig.direction == "flat"
                or (open_pos.side == "Buy" and sig.direction == "short")
                or (open_pos.side == "Sell" and sig.direction == "long")
                or abs(pnl_pct) >= 0.012
            )
            if should_exit:
                self.state.equity += pnl_pct
                self.state.trades += 1
                if pnl_pct > 0:
                    self.state.wins += 1
                self.state.history.append(
                    {
                        "symbol": symbol,
                        "side": open_pos.side,
                        "entry": open_pos.entry,
                        "exit": price,
                        "pnl_pct": pnl_pct,
                    }
                )
                del self.state.open[symbol]
        self.save()
        return sig

    def should_switch(self, live_equity: float, min_trades: int = 20, edge: float = 0.02) -> Dict[str, Any]:
        """Recommend switching live strategy if shadow outperforms by `edge`."""
        recommend = (
            self.state.trades >= min_trades
            and self.state.equity > live_equity + edge
            and (self.state.wins / max(self.state.trades, 1)) >= 0.5
        )
        return {
            "recommend_switch": recommend,
            "shadow_equity": self.state.equity,
            "live_equity": live_equity,
            "shadow_trades": self.state.trades,
            "shadow_winrate": self.state.wins / max(self.state.trades, 1),
        }
