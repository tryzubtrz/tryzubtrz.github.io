"""XGBoost classifier for trade direction."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import MODELS_DIR

logger = logging.getLogger("ai.xgboost")


class XGBoostDirectionModel:
    def __init__(self, feature_names: Optional[List[str]] = None) -> None:
        self.feature_names = feature_names or []
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_jobs=-1,
            tree_method="hist",
        )
        self.trained = False
        self.metrics: Dict[str, float] = {}
        self.feature_importances_: Dict[str, float] = {}

    @staticmethod
    def _map_y(y: np.ndarray) -> np.ndarray:
        return (y + 1).astype(int)  # -1,0,1 → 0,1,2

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        self.feature_names = list(X.columns)
        y_mapped = self._map_y(y.to_numpy())
        if len(X) < 80:
            logger.warning("Not enough XGBoost samples (%s)", len(X))
            return {"accuracy": 0.0, "f1": 0.0, "samples": float(len(X))}

        X_train, X_val, y_train, y_val = train_test_split(
            X, y_mapped, test_size=0.2, shuffle=False
        )
        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_val)
        acc = float(accuracy_score(y_val, pred))
        f1 = float(f1_score(y_val, pred, average="macro"))
        self.trained = True
        self.metrics = {"accuracy": acc, "f1": f1, "samples": float(len(X))}
        importances = self.model.feature_importances_
        self.feature_importances_ = {
            name: float(score) for name, score in zip(self.feature_names, importances)
        }
        logger.info("XGBoost trained acc=%.3f f1=%.3f", acc, f1)
        return self.metrics

    def predict_proba_direction(self, X_row: pd.DataFrame) -> Tuple[str, float]:
        if not self.trained:
            return "flat", 0.0
        proba = self.model.predict_proba(X_row[self.feature_names])[0]
        idx = int(np.argmax(proba))
        direction = {0: "short", 1: "flat", 2: "long"}[idx]
        return direction, float(proba[idx])

    def save(self, path: Path | None = None) -> Path:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = path or (MODELS_DIR / "xgb_latest.joblib")
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "metrics": self.metrics,
                "importances": self.feature_importances_,
            },
            path,
        )
        return path

    def load(self, path: Path | None = None) -> bool:
        path = path or (MODELS_DIR / "xgb_latest.joblib")
        if not path.exists():
            return False
        blob = joblib.load(path)
        self.model = blob["model"]
        self.feature_names = blob.get("feature_names", [])
        self.metrics = blob.get("metrics", {})
        self.feature_importances_ = blob.get("importances", {})
        self.trained = True
        return True
