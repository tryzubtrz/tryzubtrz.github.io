"""
Bybit V5 API client (REST + helpers).
Uses pybit; Testnet by default. Keys only from settings/.env.
IP whitelist is enforced on Bybit account side — we bind outbound requests
to the current host IP when possible and log the egress IP for verification.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import socket
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests
from pybit.unified_trading import HTTP

from config import get_settings
from core.database import audit

logger = logging.getLogger("trading.bybit")


class BybitClient:
    """Thin wrapper around Bybit V5 unified trading API."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.testnet = self.settings.bybit_testnet
        self.session: Optional[HTTP] = None
        self._public = requests.Session()
        self._egress_ip: Optional[str] = None
        self._connect()

    def _connect(self) -> None:
        key = self.settings.bybit_api_key
        secret = self.settings.bybit_api_secret
        self.session = HTTP(
            testnet=self.testnet,
            api_key=key or None,
            api_secret=secret or None,
            timeout=30,
        )
        self._egress_ip = self.detect_egress_ip()
        logger.info(
            "Bybit client ready testnet=%s egress_ip=%s",
            self.testnet,
            self._egress_ip,
        )
        audit("bybit_connect", details=f"testnet={self.testnet};ip={self._egress_ip}")

    @property
    def egress_ip(self) -> Optional[str]:
        return self._egress_ip

    def detect_egress_ip(self) -> Optional[str]:
        """Resolve current public IP used for outbound requests (for whitelist checks)."""
        for url in (
            "https://api.ipify.org",
            "https://ifconfig.me/ip",
            "https://icanhazip.com",
        ):
            try:
                resp = self._public.get(url, timeout=5)
                if resp.ok:
                    return resp.text.strip()
            except Exception:
                continue
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return None

    def _ensure_session(self) -> HTTP:
        if self.session is None:
            self._connect()
        assert self.session is not None
        return self.session

    # ── Public market data ─────────────────────────────────────────────

    def _public_http(self, testnet: Optional[bool] = None) -> HTTP:
        """Separate public client — can fall back to mainnet market data if testnet geo-blocked."""
        use_testnet = self.testnet if testnet is None else testnet
        return HTTP(testnet=use_testnet, timeout=30)

    def _market_call(self, method: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Call a public market endpoint.
        Prefer configured network; on 403/geo/rate-limit fall back to the other host.
        """
        order = [True, False] if self.testnet else [False, True]
        last_exc: Optional[Exception] = None
        for tn in order:
            try:
                client = self._public_http(testnet=tn)
                fn = getattr(client, method)
                return fn(**kwargs)
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                if "403" in msg or "usa" in msg or "rate limit" in msg or "forbidden" in msg:
                    logger.warning(
                        "Public %s failed on testnet=%s (%s) — trying fallback",
                        method,
                        tn,
                        exc,
                    )
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def get_server_time(self) -> Dict[str, Any]:
        return self._market_call("get_server_time")

    def get_tickers(self, symbol: Optional[str] = None, category: str = "linear") -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"category": category}
        if symbol:
            kwargs["symbol"] = symbol
        return self._market_call("get_tickers", **kwargs)

    def get_kline(
        self,
        symbol: str,
        interval: str = "15",
        limit: int = 200,
        category: str = "linear",
    ) -> List[List[str]]:
        resp = self._market_call(
            "get_kline",
            category=category,
            symbol=symbol,
            interval=interval,
            limit=limit,
        )
        return resp.get("result", {}).get("list", []) or []

    def get_orderbook(self, symbol: str, limit: int = 50, category: str = "linear") -> Dict[str, Any]:
        return self._market_call("get_orderbook", category=category, symbol=symbol, limit=limit)

    def get_instruments(self, symbol: Optional[str] = None, category: str = "linear") -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"category": category}
        if symbol:
            kwargs["symbol"] = symbol
        return self._ensure_session().get_instruments_info(**kwargs)

    # ── Private account ────────────────────────────────────────────────

    def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict[str, Any]:
        return self._ensure_session().get_wallet_balance(accountType=account_type)

    def get_positions(self, symbol: Optional[str] = None, category: str = "linear") -> List[Dict[str, Any]]:
        kwargs: Dict[str, Any] = {"category": category, "settleCoin": "USDT"}
        if symbol:
            kwargs["symbol"] = symbol
        resp = self._ensure_session().get_positions(**kwargs)
        return resp.get("result", {}).get("list", []) or []

    def set_leverage(self, symbol: str, leverage: int, category: str = "linear") -> Dict[str, Any]:
        try:
            return self._ensure_session().set_leverage(
                category=category,
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
        except Exception as exc:
            # Bybit returns error if leverage unchanged — treat as OK
            if "not modified" in str(exc).lower() or "110043" in str(exc):
                return {"retCode": 0, "retMsg": "leverage unchanged"}
            raise

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "Market",
        price: Optional[float] = None,
        reduce_only: bool = False,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        category: str = "linear",
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "side": side,  # Buy / Sell
            "orderType": order_type,
            "qty": str(qty),
            "reduceOnly": reduce_only,
            "timeInForce": "GTC" if order_type == "Limit" else "IOC",
        }
        if price is not None and order_type == "Limit":
            params["price"] = str(price)
        if take_profit is not None:
            params["takeProfit"] = str(take_profit)
        if stop_loss is not None:
            params["stopLoss"] = str(stop_loss)

        logger.info("Placing order %s", params)
        audit("place_order", details=str(params))
        return self._ensure_session().place_order(**params)

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        category: str = "linear",
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"category": category, "symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if order_link_id:
            params["orderLinkId"] = order_link_id
        audit("cancel_order", details=str(params))
        return self._ensure_session().cancel_order(**params)

    def close_position(self, symbol: str, category: str = "linear") -> Optional[Dict[str, Any]]:
        positions = self.get_positions(symbol=symbol, category=category)
        for pos in positions:
            size = float(pos.get("size") or 0)
            if size <= 0:
                continue
            side = pos.get("side", "")
            close_side = "Sell" if side == "Buy" else "Buy"
            return self.place_order(
                symbol=symbol,
                side=close_side,
                qty=size,
                order_type="Market",
                reduce_only=True,
                category=category,
            )
        return None

    def test_connection(self) -> Dict[str, Any]:
        """Smoke-test: server time + optional wallet if keys present."""
        result: Dict[str, Any] = {
            "ok": False,
            "testnet": self.testnet,
            "egress_ip": self._egress_ip,
            "server_time": None,
            "balance": None,
            "error": None,
        }
        try:
            st = self.get_server_time()
            result["server_time"] = st
            if self.settings.bybit_api_key and self.settings.bybit_api_secret:
                bal = self.get_wallet_balance()
                result["balance"] = bal
            result["ok"] = True
        except Exception as exc:
            result["error"] = str(exc)
            logger.exception("Bybit connection test failed")
        return result

    # ── Raw signed request (fallback) ──────────────────────────────────

    def signed_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Manual V5 signed REST call — useful when pybit lacks an endpoint."""
        settings = self.settings
        params = params or {}
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        if method.upper() == "GET":
            query = urlencode(params)
            payload = f"{timestamp}{settings.bybit_api_key}{recv_window}{query}"
            url = f"{settings.rest_base}{path}?{query}" if query else f"{settings.rest_base}{path}"
            body = None
        else:
            import json

            body = json.dumps(params)
            payload = f"{timestamp}{settings.bybit_api_key}{recv_window}{body}"
            url = f"{settings.rest_base}{path}"

        sign = hmac.new(
            settings.bybit_api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        headers = {
            "X-BAPI-API-KEY": settings.bybit_api_key,
            "X-BAPI-SIGN": sign,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
        }
        resp = self._public.request(method.upper(), url, headers=headers, data=body, timeout=30)
        resp.raise_for_status()
        return resp.json()


# Singleton-ish factory
_client: Optional[BybitClient] = None


def get_bybit_client() -> BybitClient:
    global _client
    if _client is None:
        _client = BybitClient()
    return _client
