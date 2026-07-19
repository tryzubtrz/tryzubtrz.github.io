"""Technical indicators used by strategies and ML feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(series, fast) - ema(series, slow)
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / (mid + 1e-12)
    return upper, mid, lower, width


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3):
    low_min = df["low"].rolling(k).min()
    high_max = df["high"].rolling(k).max()
    stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-12)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = atr(df, period=1) * period  # rough; use true range series
    # Recalculate TR properly
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_n = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / (atr_n + 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / (atr_n + 1e-12)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12) * 100
    return dx.rolling(period).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    return (typical * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-12)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["close"]
    out["ema_9"] = ema(c, 9)
    out["ema_21"] = ema(c, 21)
    out["ema_55"] = ema(c, 55)
    out["sma_50"] = sma(c, 50)
    out["sma_200"] = sma(c, 200)
    out["rsi_14"] = rsi(c, 14)
    macd_line, macd_sig, macd_hist = macd(c)
    out["macd"] = macd_line
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist
    upper, mid, lower, width = bollinger(c)
    out["bb_upper"] = upper
    out["bb_mid"] = mid
    out["bb_lower"] = lower
    out["bb_width"] = width
    out["atr_14"] = atr(out, 14)
    out["adx_14"] = adx(out, 14)
    sk, sd = stochastic(out)
    out["stoch_k"] = sk
    out["stoch_d"] = sd
    out["vwap"] = vwap(out)
    out["returns"] = c.pct_change()
    out["log_returns"] = np.log(c / c.shift(1))
    out["volatility_20"] = out["returns"].rolling(20).std()
    out["volume_sma_20"] = sma(out["volume"], 20)
    out["volume_ratio"] = out["volume"] / (out["volume_sma_20"] + 1e-12)
    return out
