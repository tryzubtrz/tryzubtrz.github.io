"""Shadow Mode — paper portfolio that tracks signals without placing live orders."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("tryzub.shadow")


class ShadowPortfolio:
    def __init__(self, path: Path, enabled: bool = True, start_equity: float = 10_000.0) -> None:
        self.path = path
        self.enabled = enabled
        self.equity = start_equity
        self.start_equity = start_equity
        self.open: Dict[str, Dict[str, Any]] = {}
        self.trades = 0
        self.wins = 0
        self.history: list = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.equity = float(raw.get("equity", self.equity))
            self.start_equity = float(raw.get("start_equity", self.start_equity))
            self.open = raw.get("open") or {}
            self.trades = int(raw.get("trades", 0))
            self.wins = int(raw.get("wins", 0))
            self.history = raw.get("history") or []
        except Exception as exc:
            log.warning("Shadow load failed: %s", exc)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "equity": self.equity,
                    "start_equity": self.start_equity,
                    "open": self.open,
                    "trades": self.trades,
                    "wins": self.wins,
                    "history": self.history[-200:],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def step(self, symbol: str, direction: str, confidence: float, price: float) -> None:
        if not self.enabled:
            return
        try:
            pos = self.open.get(symbol)
            if pos is None:
                if direction in {"long", "short"} and confidence >= 0.55:
                    self.open[symbol] = {"side": "Buy" if direction == "long" else "Sell", "entry": price}
                    log.info("Shadow OPEN %s %s @ %.4f", symbol, direction, price)
            else:
                d = 1 if pos["side"] == "Buy" else -1
                pnl_pct = (price - pos["entry"]) / pos["entry"] * d
                exit_now = (
                    direction == "flat"
                    or (pos["side"] == "Buy" and direction == "short")
                    or (pos["side"] == "Sell" and direction == "long")
                    or abs(pnl_pct) >= 0.012
                )
                if exit_now:
                    self.equity *= 1 + pnl_pct
                    self.trades += 1
                    if pnl_pct > 0:
                        self.wins += 1
                    self.history.append(
                        {"symbol": symbol, "side": pos["side"], "entry": pos["entry"], "exit": price, "pnl_pct": pnl_pct}
                    )
                    del self.open[symbol]
                    log.info("Shadow CLOSE %s pnl_pct=%.4f equity=%.2f", symbol, pnl_pct, self.equity)
            self.save()
        except Exception as exc:
            log.error("Shadow step error: %s", exc)

    def snapshot(self) -> Dict[str, Any]:
        ret = (self.equity / self.start_equity - 1.0) * 100 if self.start_equity else 0.0
        return {
            "enabled": self.enabled,
            "equity": self.equity,
            "return_pct": ret,
            "trades": self.trades,
            "wins": self.wins,
            "open": self.open,
        }
