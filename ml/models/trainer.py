"""High-level training with lifelong experience memory (continual learning)."""
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
from ml.brain_manager import promote_new_brain, prune_old_brains
from ml.experience_memory import ExperienceMemory
from ml.models.ensemble import MLEnsemble

logger = logging.getLogger("ai.trainer")


class ModelTrainer:
    def __init__(self, market: Optional[MarketDataService] = None) -> None:
        self.settings = get_settings()
        self.market = market or MarketDataService()
        self.ensemble = MLEnsemble()
        # Load yesterday's / last saved brain — never start "empty" if files exist
        self.ensemble.load()
        self.memory = ExperienceMemory()

    def backup_models(self) -> Path:
        """Snapshot current brain file copies before overwriting (disk hygiene only)."""
        BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dest = BACKUPS_DIR / f"models_{stamp}"
        dest.mkdir(parents=True, exist_ok=True)
        if MODELS_DIR.exists():
            for f in MODELS_DIR.iterdir():
                if f.is_file() and f.name != ".gitkeep":
                    shutil.copy2(f, dest / f.name)
        # Only deletes OLD FILE SNAPSHOTS — never experience memory / lessons
        prune_old_brains()
        return dest

    def train_all(
        self,
        symbols: Optional[List[str]] = None,
        interval: str = "15m",
        limit: int = 500,
        epochs: int = 12,
    ) -> Dict:
        """
        Continual learning loop:
        1) ingest today's market + trade mistakes into lifelong memory
        2) train on FULL memory (past + today), warm-starting from previous brain
        3) save improved brain (knowledge carries forward)
        """
        symbols = symbols or self.settings.pairs
        self.backup_models()

        # Ensure previous brain is loaded for warm-start
        self.ensemble.load()

        per_symbol = {}
        ingested = 0
        for sym in symbols:
            try:
                df = self.market.fetch_ohlcv(sym, interval=interval, limit=limit)
                if len(df) < 100:
                    continue
                ingested += self.memory.ingest_market_frame(
                    df, symbol=sym, feature_cols=self.ensemble.selected_features
                )
                per_symbol[sym] = {"rows": len(df)}
            except Exception as exc:
                logger.error("Train fetch failed %s: %s", sym, exc)

        # Permanent lessons from closed trades (losses remembered stronger)
        lessons = self.memory.ingest_trade_lessons(
            feature_cols=self.ensemble.selected_features
        )

        X, y = self.memory.training_xy(feature_cols=self.ensemble.selected_features)
        if X.empty or len(X) < 80:
            # Fallback: today's frames only
            frames = []
            for sym in symbols:
                try:
                    frames.append(self.market.fetch_ohlcv(sym, interval=interval, limit=limit))
                except Exception:
                    continue
            if not frames:
                return {"ok": False, "error": "no_data", "per_symbol": per_symbol}
            combined = pd.concat(frames, ignore_index=True)
            metrics = self.ensemble.train_on_df(combined, epochs=epochs, warm_start=True)
        else:
            metrics = self.ensemble.train_on_xy(X, y, epochs=epochs, warm_start=True)

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
        promote_new_brain(str(version), metrics)
        mem_stats = self.memory.stats()
        logger.info(
            "Continual train done brain=%s memory=%s lessons_new=%s ingested=%s",
            version,
            mem_stats,
            lessons,
            ingested,
        )
        return {
            "ok": True,
            "metrics": metrics,
            "per_symbol": per_symbol,
            "paths": paths,
            "version": version,
            "experience": mem_stats,
            "new_lessons": lessons,
            "continual_learning": True,
        }
