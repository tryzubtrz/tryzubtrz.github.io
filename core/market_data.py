"""Market data fetching, caching, and OHLCV DataFrame helpers."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.bybit_client import BybitClient, get_bybit_client
from config import CACHE_DIR

logger = logging.getLogger("trading.market")


def _synthetic_ohlcv(symbol: str, limit: int = 300, interval_minutes: int = 15) -> pd.DataFrame:
    """Deterministic synthetic candles for offline / geo-blocked environments."""
    seed = sum(ord(c) for c in symbol) % 10_000
    rng = np.random.default_rng(seed)
    # Rough spot anchors
    anchors = {"BTCUSDT": 65000.0, "ETHUSDT": 3400.0, "SOLUSDT": 145.0}
    price = anchors.get(symbol.upper(), 100.0)
    rows = []
    ts = pd.Timestamp.utcnow().floor("min") - pd.Timedelta(minutes=interval_minutes * limit)
    for _ in range(limit):
        ret = float(rng.normal(0, 0.0025))
        open_p = price
        close_p = max(0.01, price * (1 + ret))
        high_p = max(open_p, close_p) * (1 + abs(float(rng.normal(0, 0.001))))
        low_p = min(open_p, close_p) * (1 - abs(float(rng.normal(0, 0.001))))
        vol = float(abs(rng.normal(100, 25)))
        rows.append(
            {
                "timestamp": ts,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": vol,
                "turnover": vol * close_p,
            }
        )
        price = close_p
        ts += pd.Timedelta(minutes=interval_minutes)
    return pd.DataFrame(rows)


INTERVAL_MAP = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "4h": "240",
    "1d": "D",
    "1w": "W",
}


class MarketDataService:
    def __init__(self, client: Optional[BybitClient] = None) -> None:
        self.client = client or get_bybit_client()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "15m",
        limit: int = 300,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        bybit_interval = INTERVAL_MAP.get(interval, interval)
        cache_path = CACHE_DIR / f"{symbol}_{bybit_interval}.csv"

        try:
            raw = self.client.get_kline(symbol=symbol, interval=bybit_interval, limit=limit)
        except Exception as exc:
            logger.warning("Kline fetch failed for %s: %s — trying cache/synthetic", symbol, exc)
            if cache_path.exists():
                return pd.read_csv(cache_path, parse_dates=["timestamp"])
            # Parse minutes from interval key when possible
            minutes = 15
            if interval.endswith("m") and interval[:-1].isdigit():
                minutes = int(interval[:-1])
            elif interval.endswith("h") and interval[:-1].isdigit():
                minutes = int(interval[:-1]) * 60
            df = _synthetic_ohlcv(symbol, limit=limit, interval_minutes=minutes)
            if use_cache:
                df.to_csv(cache_path, index=False)
            return df

        # Bybit returns newest-first: [start, open, high, low, close, volume, turnover]
        rows = []
        for item in raw:
            rows.append(
                {
                    "timestamp": pd.to_datetime(int(item[0]), unit="ms", utc=True),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "turnover": float(item[6]) if len(item) > 6 else 0.0,
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            if cache_path.exists():
                return pd.read_csv(cache_path, parse_dates=["timestamp"])
            minutes = 15
            if interval.endswith("m") and interval[:-1].isdigit():
                minutes = int(interval[:-1])
            return _synthetic_ohlcv(symbol, limit=limit, interval_minutes=minutes)
        df = df.sort_values("timestamp").reset_index(drop=True)
        if use_cache:
            df.to_csv(cache_path, index=False)
        return df

    def get_last_price(self, symbol: str) -> float:
        try:
            tickers = self.client.get_tickers(symbol=symbol)
            lst = tickers.get("result", {}).get("list", []) or []
            if lst:
                return float(lst[0].get("lastPrice") or lst[0].get("markPrice") or 0)
        except Exception as exc:
            logger.warning("Ticker failed for %s: %s — using OHLCV close", symbol, exc)
        df = self.fetch_ohlcv(symbol, interval="15m", limit=5)
        if df.empty:
            raise ValueError(f"No price for {symbol}")
        return float(df["close"].iloc[-1])

    def get_multi_ohlcv(
        self,
        symbols: List[str],
        interval: str = "15m",
        limit: int = 300,
    ) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                out[sym] = self.fetch_ohlcv(sym, interval=interval, limit=limit)
            except Exception as exc:
                logger.error("Failed OHLCV %s: %s", sym, exc)
        return out

    def correlation_matrix(
        self,
        symbols: List[str],
        interval: str = "1h",
        limit: int = 200,
    ) -> pd.DataFrame:
        closes = {}
        for sym in symbols:
            df = self.fetch_ohlcv(sym, interval=interval, limit=limit)
            if not df.empty:
                closes[sym] = df.set_index("timestamp")["close"]
        if not closes:
            return pd.DataFrame()
        prices = pd.DataFrame(closes).dropna()
        returns = prices.pct_change().dropna()
        return returns.corr()
