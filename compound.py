"""Compound interest position sizing — grow size with equity curve."""
from __future__ import annotations

import logging
from typing import Any, Dict

log = logging.getLogger("tryzub.compound")


class CompoundSizer:
    def __init__(
        self,
        enabled: bool = True,
        base_balance: float = 10_000.0,
        max_position_pct: float = 5.0,
        max_mult: float = 3.0,
    ) -> None:
        self.enabled = enabled
        self.base_balance = base_balance
        self.max_position_pct = max_position_pct
        self.max_mult = max_mult

    def size_usd(self, current_balance: float, confidence: float, flat_size_fn) -> float:
        """
        flat_size_fn(balance, confidence) -> usd without compound.
        With compound: scale by equity / base, capped.
        """
        try:
            base = float(flat_size_fn(current_balance, confidence))
            if not self.enabled or self.base_balance <= 0:
                return base
            mult = max(0.5, min(self.max_mult, current_balance / self.base_balance))
            sized = round(base * mult, 4)
            # Hard cap vs current equity
            hard = current_balance * (self.max_position_pct / 100.0) * max(0.4, min(1.0, confidence))
            out = min(sized, hard)
            log.info("Compound size base=%.2f mult=%.3f → %.2f (bal=%.2f)", base, mult, out, current_balance)
            return round(out, 4)
        except Exception as exc:
            log.error("Compound sizing error: %s", exc)
            try:
                return float(flat_size_fn(current_balance, confidence))
            except Exception:
                return 0.0

    def snapshot(self, current_balance: float) -> Dict[str, Any]:
        mult = (current_balance / self.base_balance) if self.base_balance else 1.0
        return {
            "enabled": self.enabled,
            "base_balance": self.base_balance,
            "current_balance": current_balance,
            "multiplier": float(max(0.5, min(self.max_mult, mult))),
        }
