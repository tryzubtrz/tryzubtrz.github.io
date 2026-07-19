"""Execute strategy signals through risk checks → Bybit → DB."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from config import get_settings
from core.bybit_client import BybitClient, get_bybit_client
from core.market_data import MarketDataService
from core.position_manager import PositionManager
from core.risk_manager import RiskManager

logger = logging.getLogger("trading.executor")


class OrderExecutor:
    def __init__(
        self,
        client: Optional[BybitClient] = None,
        risk: Optional[RiskManager] = None,
        positions: Optional[PositionManager] = None,
        market: Optional[MarketDataService] = None,
    ) -> None:
        self.settings = get_settings()
        self.client = client or get_bybit_client()
        self.risk = risk or RiskManager()
        self.positions = positions or PositionManager()
        self.market = market or MarketDataService(self.client)

    def _balance_usd(self) -> float:
        try:
            bal = self.client.get_wallet_balance()
            lst = bal.get("result", {}).get("list", []) or []
            if not lst:
                return self.settings.initial_balance
            coins = lst[0].get("coin", []) or []
            for c in coins:
                if c.get("coin") == "USDT":
                    return float(c.get("walletBalance") or c.get("equity") or 0)
            total = lst[0].get("totalEquity") or lst[0].get("totalWalletBalance")
            if total:
                return float(total)
        except Exception as exc:
            logger.warning("Balance fetch failed, using start_balance: %s", exc)
        return self.risk.state.start_balance + self.risk.state.realized_pnl

    def execute_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        signal keys: symbol, direction (long/short/flat), confidence, strategy
        """
        symbol = signal["symbol"]
        direction = signal.get("direction", "flat").lower()
        confidence = float(signal.get("confidence") or 0)
        strategy = signal.get("strategy") or "ensemble"

        result: Dict[str, Any] = {"executed": False, "reason": "", "trade": None}

        if direction == "flat" or confidence < self.settings.min_confidence:
            result["reason"] = "low_confidence_or_flat"
            return result

        ok, why = self.risk.can_open_trade(self.positions.open_count())
        if not ok:
            result["reason"] = why
            return result

        # Avoid stacking same-symbol exposure
        for pos in self.positions.list_open():
            if pos["symbol"] == symbol:
                result["reason"] = "already_open"
                return result

        price = self.market.get_last_price(symbol)
        balance = self._balance_usd()
        usd = self.risk.position_size_usd(balance, confidence)
        leverage = self.settings.default_leverage
        qty = self.risk.qty_from_usd(usd, price, leverage)
        if qty <= 0:
            result["reason"] = "qty_zero"
            return result

        side = "Buy" if direction == "long" else "Sell"
        levels = self.risk.stop_take_prices(side, price)

        try:
            self.client.set_leverage(symbol, leverage)
            order = self.client.place_order(
                symbol=symbol,
                side=side,
                qty=qty,
                order_type="Market",
                take_profit=levels["take_profit"],
                stop_loss=levels["stop_loss"],
            )
            order_id = (
                (order.get("result") or {}).get("orderId")
                if isinstance(order, dict)
                else ""
            ) or ""
            trade_id = self.positions.open_trade(
                symbol=symbol,
                side=side,
                qty=qty,
                entry_price=price,
                leverage=leverage,
                strategy=strategy,
                confidence=confidence,
                order_id=order_id,
                notes=f"tp={levels['take_profit']};sl={levels['stop_loss']}",
            )
            result.update(
                {
                    "executed": True,
                    "reason": "ok",
                    "trade": {
                        "id": trade_id,
                        "symbol": symbol,
                        "side": side,
                        "qty": qty,
                        "entry_price": price,
                        "order_id": order_id,
                        **levels,
                    },
                }
            )
        except Exception as exc:
            logger.exception("Order execution failed")
            result["reason"] = f"error:{exc}"
        return result

    def close_position(
        self,
        symbol: str,
        exit_reason: str = "manual",
    ) -> Dict[str, Any]:
        price = self.market.get_last_price(symbol)
        try:
            self.client.close_position(symbol)
        except Exception as exc:
            logger.warning("Exchange close failed for %s: %s", symbol, exc)
        closed = self.positions.close_by_symbol(symbol, price, exit_reason)
        balance = self._balance_usd()
        goals = []
        for c in closed:
            goals.extend(self.risk.register_closed_pnl(c["pnl"], balance))
        return {"closed": closed, "goals_hit": goals, "risk": self.risk.snapshot()}

    def sync_exits_from_exchange(self) -> list:
        """Mark local open trades closed if exchange position size is 0."""
        closed_events = []
        open_local = self.positions.list_open()
        for pos in open_local:
            try:
                remote = self.client.get_positions(symbol=pos["symbol"])
                size = 0.0
                for r in remote:
                    size += float(r.get("size") or 0)
                if size > 0:
                    continue
                price = self.market.get_last_price(pos["symbol"])
                # Infer exit reason from TP/SL proximity
                reason = "exchange_flat"
                entry = pos["entry_price"]
                levels = self.risk.stop_take_prices(pos["side"], entry)
                if abs(price - levels["take_profit"]) / entry < 0.002:
                    reason = "take_profit"
                elif abs(price - levels["stop_loss"]) / entry < 0.002:
                    reason = "stop_loss"
                result = self.positions.close_trade(pos["id"], price, reason)
                balance = self._balance_usd()
                goals = self.risk.register_closed_pnl(result["pnl"], balance)
                closed_events.append({"trade": result, "goals_hit": goals})
            except Exception as exc:
                logger.error("sync_exits failed for %s: %s", pos["symbol"], exc)
        return closed_events
