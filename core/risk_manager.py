"""Position sizing, daily loss limits, and goal tracking."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

import pytz

from config import get_settings
from core.database import DailyStats, audit, session_scope, utcnow
from sqlalchemy import select

logger = logging.getLogger("trading.risk")


@dataclass
class RiskState:
    day: str
    start_balance: float
    realized_pnl: float = 0.0
    open_risk: float = 0.0
    goals_hit: Set[float] = field(default_factory=set)
    loss_limit_hit: bool = False
    trading_halted: bool = False


class RiskManager:
    def __init__(self, start_balance: Optional[float] = None) -> None:
        self.settings = get_settings()
        self.tz = pytz.timezone(self.settings.timezone)
        self.state = RiskState(
            day=self._today(),
            start_balance=start_balance or self.settings.initial_balance,
        )
        self._load_or_create_daily()

    def _today(self) -> str:
        return datetime.now(self.tz).strftime("%Y-%m-%d")

    def _load_or_create_daily(self) -> None:
        day = self._today()
        with session_scope() as s:
            row = s.execute(select(DailyStats).where(DailyStats.day == day)).scalar_one_or_none()
            if row is None:
                s.add(
                    DailyStats(
                        day=day,
                        start_balance=self.state.start_balance,
                        end_balance=self.state.start_balance,
                    )
                )
            else:
                self.state.day = row.day
                self.state.start_balance = row.start_balance or self.settings.initial_balance
                self.state.realized_pnl = row.pnl or 0.0
                self.state.loss_limit_hit = bool(row.loss_limit_hit)
                if row.goals_hit:
                    self.state.goals_hit = {float(x) for x in row.goals_hit.split(",") if x}
                if self.state.loss_limit_hit:
                    self.state.trading_halted = True

    def rollover_if_needed(self, current_balance: float) -> None:
        today = self._today()
        if today == self.state.day:
            return
        self._persist_daily(current_balance)
        self.state = RiskState(day=today, start_balance=current_balance)
        self._load_or_create_daily()
        audit("day_rollover", details=f"day={today};balance={current_balance}")

    def current_pnl_pct(self) -> float:
        if self.state.start_balance <= 0:
            return 0.0
        return (self.state.realized_pnl / self.state.start_balance) * 100.0

    def can_open_trade(self, open_positions: int) -> tuple[bool, str]:
        self.rollover_if_needed(self.state.start_balance + self.state.realized_pnl)
        if self.state.trading_halted or self.state.loss_limit_hit:
            return False, "daily_loss_limit"
        if open_positions >= self.settings.max_open_positions:
            return False, "max_open_positions"
        if self.current_pnl_pct() <= -abs(self.settings.max_daily_loss_pct):
            self.trigger_loss_limit()
            return False, "daily_loss_limit"
        return True, "ok"

    def position_size_usd(self, balance: float, confidence: float = 0.7) -> float:
        """Kelly-inspired fractional sizing capped by max_position_pct."""
        base = balance * (self.settings.max_position_pct / 100.0)
        # Scale by confidence between 0.4x and 1.0x
        scale = max(0.4, min(1.0, confidence))
        return round(base * scale, 4)

    def qty_from_usd(self, usd: float, price: float, leverage: int) -> float:
        if price <= 0:
            return 0.0
        notional = usd * max(leverage, 1)
        qty = notional / price
        # Round to reasonable precision for linear USDT contracts
        if price >= 1000:
            return round(qty, 3)
        if price >= 1:
            return round(qty, 2)
        return round(qty, 4)

    def stop_take_prices(self, side: str, entry: float) -> Dict[str, float]:
        sl_pct = self.settings.stop_loss_pct / 100.0
        tp_pct = self.settings.take_profit_pct / 100.0
        if side.lower() in {"buy", "long"}:
            return {
                "stop_loss": round(entry * (1 - sl_pct), 6),
                "take_profit": round(entry * (1 + tp_pct), 6),
            }
        return {
            "stop_loss": round(entry * (1 + sl_pct), 6),
            "take_profit": round(entry * (1 - tp_pct), 6),
        }

    def register_closed_pnl(self, pnl: float, balance: float) -> List[float]:
        """Update daily PnL; return newly achieved goal percentages."""
        self.state.realized_pnl += pnl
        pct = self.current_pnl_pct()
        newly: List[float] = []
        for goal in self.settings.daily_goals:
            if pct >= goal and goal not in self.state.goals_hit:
                self.state.goals_hit.add(goal)
                newly.append(goal)
                audit("daily_goal_hit", details=f"goal={goal}%;pnl_pct={pct:.3f}")
        if pct <= -abs(self.settings.max_daily_loss_pct):
            self.trigger_loss_limit()
        self._persist_daily(balance)
        return newly

    def trigger_loss_limit(self) -> None:
        if self.state.loss_limit_hit:
            return
        self.state.loss_limit_hit = True
        self.state.trading_halted = True
        logger.warning("Daily loss limit triggered at %.3f%%", self.current_pnl_pct())
        audit("daily_loss_limit", details=f"pnl_pct={self.current_pnl_pct():.3f}")
        self._persist_daily(self.state.start_balance + self.state.realized_pnl)

    def _persist_daily(self, end_balance: float) -> None:
        with session_scope() as s:
            row = s.execute(
                select(DailyStats).where(DailyStats.day == self.state.day)
            ).scalar_one_or_none()
            goals = ",".join(str(g) for g in sorted(self.state.goals_hit))
            if row is None:
                s.add(
                    DailyStats(
                        day=self.state.day,
                        start_balance=self.state.start_balance,
                        end_balance=end_balance,
                        pnl=self.state.realized_pnl,
                        pnl_pct=self.current_pnl_pct(),
                        goals_hit=goals,
                        loss_limit_hit=self.state.loss_limit_hit,
                    )
                )
            else:
                row.end_balance = end_balance
                row.pnl = self.state.realized_pnl
                row.pnl_pct = self.current_pnl_pct()
                row.goals_hit = goals
                row.loss_limit_hit = self.state.loss_limit_hit

    def snapshot(self) -> Dict:
        return {
            "day": self.state.day,
            "start_balance": self.state.start_balance,
            "realized_pnl": self.state.realized_pnl,
            "pnl_pct": self.current_pnl_pct(),
            "goals_hit": sorted(self.state.goals_hit),
            "loss_limit_hit": self.state.loss_limit_hit,
            "trading_halted": self.state.trading_halted,
            "max_daily_loss_pct": self.settings.max_daily_loss_pct,
        }
