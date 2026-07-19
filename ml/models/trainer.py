"""High-level training orchestration across symbols."""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import MODELS_DIR, BACKUPS_DIR, get_settings
from core.database import ModelCheckpoint, session_scope
from core.market_data import MarketDataService
from ml.models.ensemble import MLEnsemble

logger = logging.getLogger("ai.trainer")


class ModelTrainer:
    def __init__(self, market: Optional[MarketDataService] = None) -> None:
        self.settings = get_settings()
        self.market = market or MarketDataService()
        self.ensemble = MLEnsemble()
        self.ensemble.load()

    def backup_models(self) -> Path:
        BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dest = BACKUPS_DIR / f"models_{stamp}"
        dest.mkdir(parents=True, exist_ok=True)
        if MODELS_DIR.exists():
            for f in MODELS_DIR.iterdir():
                if f.is_file() and f.name != ".gitkeep":
                    shutil.copy2(f, dest / f.name)
        self._prune_backups()
        return dest

    def _prune_backups(self) -> None:
        dirs = sorted(
            [p for p in BACKUPS_DIR.iterdir() if p.is_dir() and p.name.startswith("models_")],
            reverse=True,
        )
        for old in dirs[self.settings.model_keep_versions :]:
            shutil.rmtree(old, ignore_errors=True)

    def train_all(
        self,
        symbols: Optional[List[str]] = None,
        interval: str = "15m",
        limit: int = 500,
        epochs: int = 12,
    ) -> Dict:
        symbols = symbols or self.settings.pairs
        self.backup_models()
        frames = []
        per_symbol = {}
        for sym in symbols:
            try:
                df = self.market.fetch_ohlcv(sym, interval=interval, limit=limit)
                if len(df) < 100:
                    continue
                frames.append(df)
                # Train also per-symbol lightly for metrics
                ens = MLEnsemble()
                metrics = ens.train_on_df(df, epochs=max(5, epochs // 2))
                per_symbol[sym] = metrics
            except Exception as exc:
                logger.error("Train fetch failed %s: %s", sym, exc)

        if not frames:
            return {"ok": False, "error": "no_data", "per_symbol": per_symbol}

        combined = pd.concat(frames, ignore_index=True)
        metrics = self.ensemble.train_on_df(combined, epochs=epochs)
        paths = self.ensemble.save()
        version = int(datetime.now(timezone.utc).timestamp())
        with session_scope() as s:
            for name, path in paths.items():
                s.add(
                    ModelCheckpoint(
                        model_name=name,
                        version=version,
                        path=path,
                        metrics_json=str(metrics),
                    )
                )
        logger.info("Training complete: %s", metrics)
        return {
            "ok": True,
            "metrics": metrics,
            "per_symbol": per_symbol,
            "paths": paths,
            "version": version,
        }
