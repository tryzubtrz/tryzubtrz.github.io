"""Market anomaly detection (volatility spikes, volume shocks, gaps)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from core.database import MarketAnomaly, audit, session_scope

logger = logging.getLogger("trading.anomaly")


class AnomalyDetector:
    def __init__(
        self,
        vol_z_threshold: float = 3.5,
        volume_z_threshold: float = 4.0,
        gap_pct_threshold: float = 1.5,
    ) -> None:
        self.vol_z_threshold = vol_z_threshold
        self.volume_z_threshold = volume_z_threshold
        self.gap_pct_threshold = gap_pct_threshold

    def analyze(self, symbol: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
        if df is None or len(df) < 40:
            return []
        anomalies: List[Dict[str, Any]] = []
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        returns = close.pct_change()
        vol = returns.rolling(20).std()
        vol_z = (vol - vol.mean()) / (vol.std() + 1e-12)
        vol_z_last = float(vol_z.iloc[-1]) if not np.isnan(vol_z.iloc[-1]) else 0.0
        if abs(vol_z_last) >= self.vol_z_threshold:
            anomalies.append(
                {
                    "symbol": symbol,
                    "anomaly_type": "volatility_spike",
                    "severity": "high" if abs(vol_z_last) > 5 else "medium",
                    "details": f"vol_z={vol_z_last:.2f}",
                }
            )

        vol_mean = volume.rolling(30).mean()
        vol_std = volume.rolling(30).std()
        volume_z = float(((volume.iloc[-1] - vol_mean.iloc[-1]) / (vol_std.iloc[-1] + 1e-12)))
        if abs(volume_z) >= self.volume_z_threshold:
            anomalies.append(
                {
                    "symbol": symbol,
                    "anomaly_type": "volume_shock",
                    "severity": "high" if abs(volume_z) > 6 else "medium",
                    "details": f"volume_z={volume_z:.2f}",
                }
            )

        # Gap vs previous close
        gap_pct = abs(float(df["open"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100)
        if gap_pct >= self.gap_pct_threshold:
            anomalies.append(
                {
                    "symbol": symbol,
                    "anomaly_type": "price_gap",
                    "severity": "high" if gap_pct > 3 else "medium",
                    "details": f"gap_pct={gap_pct:.2f}",
                }
            )

        for a in anomalies:
            self._persist(a)
        return anomalies

    def _persist(self, anomaly: Dict[str, Any]) -> None:
        with session_scope() as s:
            s.add(
                MarketAnomaly(
                    symbol=anomaly["symbol"],
                    anomaly_type=anomaly["anomaly_type"],
                    severity=anomaly["severity"],
                    details=anomaly["details"],
                )
            )
        audit(
            "market_anomaly",
            details=f"{anomaly['symbol']} {anomaly['anomaly_type']} {anomaly['details']}",
        )
        logger.warning("Anomaly detected: %s", anomaly)
