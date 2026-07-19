"""Combine LSTM + XGBoost into a single ML vote."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ml.features import EXTENDED_FEATURES, build_feature_frame, make_supervised, sequence_windows
from ml.models.lstm_model import LSTMModel
from ml.models.xgboost_model import XGBoostDirectionModel
from strategies.base import StrategySignal


class MLEnsemble:
    def __init__(self, seq_len: int = 32) -> None:
        self.seq_len = seq_len
        self.xgb = XGBoostDirectionModel()
        self.lstm = LSTMModel(n_features=len(EXTENDED_FEATURES), seq_len=seq_len)
        self.xgb_weight = 0.55
        self.lstm_weight = 0.45
        self.selected_features = list(EXTENDED_FEATURES)

    def train_on_df(self, df: pd.DataFrame, epochs: int = 12) -> Dict[str, Dict[str, float]]:
        X, y_class, _ = make_supervised(df, feature_cols=self.selected_features)
        xgb_metrics = self.xgb.fit(X, y_class)
        X_np = X.to_numpy(dtype=np.float32)
        # Standardize for LSTM
        mean, std = X_np.mean(axis=0), X_np.std(axis=0) + 1e-8
        self._mean, self._std = mean, std
        X_norm = (X_np - mean) / std
        X_seq, y_seq = sequence_windows(X_norm, y_class.to_numpy(), seq_len=self.seq_len)
        self.lstm.n_features = X_norm.shape[1]
        self.lstm.model = self.lstm.model.__class__(self.lstm.n_features).to(self.lstm.device)
        lstm_metrics = self.lstm.fit(X_seq, y_seq, epochs=epochs)
        return {"xgboost": xgb_metrics, "lstm": lstm_metrics}

    def predict(self, symbol: str, df: pd.DataFrame) -> StrategySignal:
        data = build_feature_frame(df)
        cols = [c for c in self.selected_features if c in data.columns]
        row = data[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if row.empty:
            return StrategySignal(symbol, "flat", 0.0, "ml_ensemble")

        xgb_dir, xgb_conf = ("flat", 0.0)
        lstm_dir, lstm_conf = ("flat", 0.0)
        if self.xgb.trained:
            xgb_dir, xgb_conf = self.xgb.predict_proba_direction(row.iloc[[-1]][self.xgb.feature_names])

        if self.lstm.trained and hasattr(self, "_mean"):
            X_np = row[cols].to_numpy(dtype=np.float32)
            X_norm = (X_np - self._mean[: X_np.shape[1]]) / self._std[: X_np.shape[1]]
            if len(X_norm) >= self.seq_len:
                window = X_norm[-self.seq_len :][None, ...]
                lstm_dir, lstm_conf = self.lstm.predict_proba_direction(window)

        scores = {"long": 0.0, "short": 0.0, "flat": 0.0}
        scores[xgb_dir] += self.xgb_weight * xgb_conf
        scores[lstm_dir] += self.lstm_weight * lstm_conf
        direction = max(scores, key=scores.get)
        confidence = scores[direction]
        if direction == "flat":
            confidence = 0.0
        return StrategySignal(
            symbol,
            direction,
            float(min(confidence, 0.99)),
            "ml_ensemble",
            meta={"xgb": (xgb_dir, xgb_conf), "lstm": (lstm_dir, lstm_conf), "scores": scores},
        )

    def save(self) -> Dict[str, str]:
        paths = {
            "xgb": str(self.xgb.save()),
            "lstm": str(self.lstm.save()),
        }
        # Persist scaler + selected features
        import json
        from config import MODELS_DIR

        meta = {
            "selected_features": self.selected_features,
            "mean": getattr(self, "_mean", np.array([])).tolist(),
            "std": getattr(self, "_std", np.array([])).tolist(),
            "seq_len": self.seq_len,
        }
        meta_path = MODELS_DIR / "ml_meta.json"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        paths["meta"] = str(meta_path)
        return paths

    def load(self) -> bool:
        import json
        from config import MODELS_DIR

        ok_x = self.xgb.load()
        ok_l = self.lstm.load()
        meta_path = MODELS_DIR / "ml_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.selected_features = meta.get("selected_features", self.selected_features)
            self._mean = np.array(meta.get("mean", []), dtype=np.float32)
            self._std = np.array(meta.get("std", []), dtype=np.float32)
            self.seq_len = int(meta.get("seq_len", self.seq_len))
        return ok_x or ok_l
