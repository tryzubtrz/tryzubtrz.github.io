"""Track open/closed trades in DB and sync with exchange positions."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from core.database import Trade, audit, session_scope, utcnow

logger = logging.getLogger("trading.positions")


class PositionManager:
    def open_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        leverage: int,
        strategy: str,
        confidence: float,
        order_id: str = "",
        notes: str = "",
    ) -> int:
        with session_scope() as s:
            trade = Trade(
                symbol=symbol,
                side=side,
                qty=qty,
                entry_price=entry_price,
                leverage=leverage,
                strategy=strategy,
                confidence=confidence,
                order_id=order_id,
                status="open",
                notes=notes,
                opened_at=utcnow(),
            )
            s.add(trade)
            s.flush()
            trade_id = trade.id
        audit(
            "trade_opened",
            details=f"{side} {qty} {symbol} @{entry_price} strategy={strategy}",
        )
        return int(trade_id)

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str,
        fee: float = 0.0,
    ) -> Dict[str, Any]:
        with session_scope() as s:
            trade = s.get(Trade, trade_id)
            if trade is None:
                raise ValueError(f"Trade {trade_id} not found")
            if trade.status != "open":
                raise ValueError(f"Trade {trade_id} already {trade.status}")

            direction = 1.0 if trade.side.lower() in {"buy", "long"} else -1.0
            pnl = (exit_price - trade.entry_price) * trade.qty * direction
            # Approximate leverage effect on %-return of margin
            margin = (trade.entry_price * trade.qty) / max(trade.leverage or 1, 1)
            pnl_pct = (pnl / margin * 100.0) if margin else 0.0
            trade.exit_price = exit_price
            trade.pnl = pnl - fee
            trade.pnl_pct = pnl_pct
            trade.fee = fee
            trade.exit_reason = exit_reason
            trade.status = "closed"
            trade.closed_at = utcnow()
            result = {
                "id": trade.id,
                "symbol": trade.symbol,
                "side": trade.side,
                "qty": trade.qty,
                "entry_price": trade.entry_price,
                "exit_price": exit_price,
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "exit_reason": exit_reason,
                "strategy": trade.strategy,
            }
        audit(
            "trade_closed",
            details=(
                f"id={result['id']} {result['symbol']} pnl={result['pnl']:.4f} "
                f"reason={exit_reason}"
            ),
        )
        return result

    def close_by_symbol(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
    ) -> List[Dict[str, Any]]:
        closed = []
        with session_scope() as s:
            rows = list(
                s.execute(
                    select(Trade).where(Trade.symbol == symbol, Trade.status == "open")
                ).scalars()
            )
            ids = [r.id for r in rows]
        for tid in ids:
            closed.append(self.close_trade(tid, exit_price, exit_reason))
        return closed

    def list_open(self) -> List[Dict[str, Any]]:
        with session_scope() as s:
            rows = list(s.execute(select(Trade).where(Trade.status == "open")).scalars())
            return [
                {
                    "id": r.id,
                    "symbol": r.symbol,
                    "side": r.side,
                    "qty": r.qty,
                    "entry_price": r.entry_price,
                    "leverage": r.leverage,
                    "strategy": r.strategy,
                    "confidence": r.confidence,
                    "opened_at": r.opened_at.isoformat() if r.opened_at else None,
                }
                for r in rows
            ]

    def list_closed(self, limit: int = 50) -> List[Dict[str, Any]]:
        with session_scope() as s:
            rows = list(
                s.execute(
                    select(Trade)
                    .where(Trade.status == "closed")
                    .order_by(Trade.closed_at.desc())
                    .limit(limit)
                ).scalars()
            )
            return [
                {
                    "id": r.id,
                    "symbol": r.symbol,
                    "side": r.side,
                    "qty": r.qty,
                    "entry_price": r.entry_price,
                    "exit_price": r.exit_price,
                    "pnl": r.pnl,
                    "pnl_pct": r.pnl_pct,
                    "exit_reason": r.exit_reason,
                    "strategy": r.strategy,
                    "closed_at": r.closed_at.isoformat() if r.closed_at else None,
                }
                for r in rows
            ]

    def open_count(self) -> int:
        return len(self.list_open())
