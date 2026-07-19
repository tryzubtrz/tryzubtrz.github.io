"""Funding Rate Harvesting — bias toward receiving funding on Bybit linear."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests

log = logging.getLogger("tryzub.funding")


class FundingStrategy:
    name = "funding"

    def __init__(self, enabled: bool = False, min_rate: float = 0.0001, testnet: bool = True) -> None:
        self.enabled = enabled
        self.min_rate = min_rate
        self.testnet = testnet
        self._session = requests.Session()

    def _base(self) -> str:
        return "https://api-testnet.bybit.com" if self.testnet else "https://api.bybit.com"

    def fetch_rate(self, symbol: str) -> Optional[float]:
        try:
            url = f"{self._base()}/v5/market/funding/history"
            r = self._session.get(url, params={"category": "linear", "symbol": symbol, "limit": 1}, timeout=15)
            r.raise_for_status()
            lst = r.json().get("result", {}).get("list", []) or []
            if not lst:
                return None
            return float(lst[0].get("fundingRate") or 0)
        except Exception as exc:
            log.warning("Funding fetch %s failed: %s", symbol, exc)
            return None

    def generate(self, symbol: str) -> Dict[str, Any]:
        try:
            if not self.enabled:
                return {"symbol": symbol, "direction": "flat", "confidence": 0.0, "strategy": self.name}
            rate = self.fetch_rate(symbol)
            if rate is None:
                return {"symbol": symbol, "direction": "flat", "confidence": 0.0, "strategy": self.name, "funding_rate": None}
            # Positive funding → shorts receive; negative → longs receive
            direction = "flat"
            conf = 0.0
            if rate >= self.min_rate:
                direction, conf = "short", min(0.8, 0.5 + abs(rate) * 200)
            elif rate <= -self.min_rate:
                direction, conf = "long", min(0.8, 0.5 + abs(rate) * 200)
            log.info("Funding %s rate=%s → %s conf=%.3f", symbol, rate, direction, conf)
            return {
                "symbol": symbol,
                "direction": direction,
                "confidence": float(conf),
                "strategy": self.name,
                "funding_rate": rate,
            }
        except Exception as exc:
            log.error("Funding strategy error %s: %s", symbol, exc)
            return {"symbol": symbol, "direction": "flat", "confidence": 0.0, "strategy": self.name, "error": str(exc)}
