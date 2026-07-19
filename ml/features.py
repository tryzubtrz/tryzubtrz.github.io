"""Feature engineering & selection helpers for ML models."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from indicators.technical import add_all_indicators

FEATURE_COLUMNS = [
    "returns",
    "log_returns",
    "volatility_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_width",
    "atr_14",
    "adx_14",
    "stoch_k",
    "stoch_d",
    "volume_ratio",
    "ema_9",
    "ema_21",
    "ema_55",
    "sma_50",
]


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = add_all_indicators(df)
    # Relative EMAs to price for scale invariance
    data["ema9_rel"] = data["ema_9"] / data["close"] - 1
    data["ema21_rel"] = data["ema_21"] / data["close"] - 1
    data["ema55_rel"] = data["ema_55"] / data["close"] - 1
    data["sma50_rel"] = data["sma_50"] / data["close"] - 1
    data["bb_pos"] = (data["close"] - data["bb_lower"]) / (
        data["bb_upper"] - data["bb_lower"] + 1e-12
    )
    return data


EXTENDED_FEATURES = FEATURE_COLUMNS + [
    "ema9_rel",
    "ema21_rel",
    "ema55_rel",
    "sma50_rel",
    "bb_pos",
]


def make_supervised(
    df: pd.DataFrame,
    horizon: int = 1,
    threshold: float = 0.0015,
    feature_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build X, y_class, y_reg.
    y_class: 1=up, 0=flat/down-ish mapped later to {-1,0,1} via threshold.
    """
    data = build_feature_frame(df)
    cols = feature_cols or EXTENDED_FEATURES
    future_ret = data["close"].shift(-horizon) / data["close"] - 1.0
    labels = np.where(future_ret > threshold, 1, np.where(future_ret < -threshold, -1, 0))
    frame = data[cols].copy()
    frame["y_class"] = labels
    frame["y_reg"] = future_ret
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    X = frame[cols]
    y_class = frame["y_class"].astype(int)
    y_reg = frame["y_reg"].astype(float)
    return X, y_class, y_reg


def sequence_windows(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i - seq_len : i])
        ys.append(y[i])
    if not xs:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0,))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys)
