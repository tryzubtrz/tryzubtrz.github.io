"""Update important indicators via mutual information + XGBoost importance."""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from ml.features import EXTENDED_FEATURES, make_supervised
from ml.models.xgboost_model import XGBoostDirectionModel

logger = logging.getLogger("ai.features")


def select_features(
    df: pd.DataFrame,
    top_k: int = 12,
    feature_cols: List[str] | None = None,
) -> Tuple[List[str], Dict[str, float]]:
    cols = feature_cols or EXTENDED_FEATURES
    X, y_class, _ = make_supervised(df, feature_cols=cols)
    if len(X) < 50:
        return cols[:top_k], {c: 0.0 for c in cols[:top_k]}

    y_mapped = (y_class.to_numpy() + 1).astype(int)
    mi = mutual_info_classif(X, y_mapped, discrete_features=False, random_state=42)
    mi_scores = {c: float(s) for c, s in zip(X.columns, mi)}

    xgb = XGBoostDirectionModel()
    xgb.fit(X, y_class)
    xgb_scores = xgb.feature_importances_

    combined: Dict[str, float] = {}
    for c in X.columns:
        combined[c] = 0.5 * mi_scores.get(c, 0.0) + 0.5 * xgb_scores.get(c, 0.0)

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    selected = [name for name, _ in ranked[:top_k]]
    logger.info("Selected features: %s", selected)
    return selected, combined
