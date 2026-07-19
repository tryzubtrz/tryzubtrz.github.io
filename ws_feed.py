"""Bybit public WebSocket ticker feed (thread) with REST fallback awareness."""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Dict, List, Optional

log = logging.getLogger("tryzub.ws")


class BybitWSFeed:
    def __init__(self, pairs: List[str], testnet: bool = True) -> None:
        self.pairs = [p.upper() for p in pairs]
        self.testnet = testnet
        self.prices: Dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.connected = False

    @property
    def url(self) -> str:
        if self.testnet:
            return "wss://stream-testnet.bybit.com/v5/public/linear"
        return "wss://stream.bybit.com/v5/public/linear"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="bybit-ws", daemon=True)
        self._thread.start()
        log.info("WS feed starting url=%s pairs=%s", self.url, self.pairs)

    def stop(self) -> None:
        self._stop.set()
        self.connected = False

    def get_price(self, symbol: str) -> Optional[float]:
        return self.prices.get(symbol.upper())

    def _run(self) -> None:
        try:
            import websocket  # websocket-client
        except ImportError:
            log.warning("websocket-client missing — WS feed disabled")
            return

        while not self._stop.is_set():
            try:
                ws = websocket.WebSocket()
                ws.connect(self.url, timeout=20)
                args = [f"tickers.{s}" for s in self.pairs]
                ws.send(json.dumps({"op": "subscribe", "args": args}))
                self.connected = True
                log.info("WS connected, subscribed %s", args)
                ws.settimeout(30)
                while not self._stop.is_set():
                    try:
                        raw = ws.recv()
                    except Exception:
                        break
                    if not raw:
                        break
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    topic = msg.get("topic") or ""
                    data = msg.get("data")
                    if topic.startswith("tickers.") and isinstance(data, dict):
                        sym = data.get("symbol") or topic.split(".", 1)[-1]
                        last = data.get("lastPrice") or data.get("markPrice")
                        if sym and last:
                            self.prices[str(sym).upper()] = float(last)
                try:
                    ws.close()
                except Exception:
                    pass
                self.connected = False
            except Exception as exc:
                self.connected = False
                log.warning("WS loop error: %s — retry in 5s", exc)
                time.sleep(5)
