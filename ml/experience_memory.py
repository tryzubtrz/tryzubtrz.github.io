"""
Lifelong experience memory — NEVER wiped by daily brain rotation.

Stores:
1) market supervised samples (features + labels) accumulated over time
2) trade lessons (especially losses) so mistakes are not forgotten

Daily retrain reads this memory + today's data → continues learning forward.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import select

from config import DATA_DIR, get_settings
from core.database import Trade, session_scope
from ml.features import EXTENDED_FEATURES, make_supervised

logger = logging.getLogger("ai.experience")

EXP_DIR = DATA_DIR / "experience"
MARKET_PATH = EXP_DIR / "market_memory.csv"
MISTAKES_PATH = EXP_DIR / "mistakes.jsonl"
META_PATH = EXP_DIR / "memory_meta.json"


class ExperienceMemory:
    def __init__(self, max_market_rows: Optional[int] = None) -> None:
        settings = get_settings()
        self.max_market_rows = max_market_rows or int(
            getattr(settings, "experience_max_samples", 50_000)
        )
        EXP_DIR.mkdir(parents=True, exist_ok=True)

    def _load_market(self) -> pd.DataFrame:
        if not MARKET_PATH.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(MARKET_PATH)
        except Exception as exc:
            logger.error("Failed to load market memory: %s", exc)
            return pd.DataFrame()

    def _save_market(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        # Keep mistake-sourced rows preferentially when trimming
        if len(df) > self.max_market_rows:
            if "source" in df.columns:
                mistakes = df[df["source"] == "trade_lesson"]
                market = df[df["source"] != "trade_lesson"]
                keep_mkt = max(0, self.max_market_rows - len(mistakes))
                market = market.tail(keep_mkt)
                df = pd.concat([market, mistakes], ignore_index=True)
            else:
                df = df.tail(self.max_market_rows)
        df.to_csv(MARKET_PATH, index=False)
        META_PATH.write_text(
            json.dumps(
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "rows": int(len(df)),
                    "mistake_rows": int((df.get("source") == "trade_lesson").sum())
                    if "source" in df.columns
                    else 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def ingest_market_frame(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        feature_cols: Optional[List[str]] = None,
    ) -> int:
        """Append supervised samples from OHLCV into lifelong memory."""
        cols = feature_cols or EXTENDED_FEATURES
        try:
            X, y_class, y_reg = make_supervised(df, feature_cols=cols)
        except Exception as exc:
            logger.warning("ingest_market_frame failed: %s", exc)
            return 0
        if X.empty:
            return 0
        chunk = X.copy()
        chunk["y_class"] = y_class.to_numpy()
        chunk["y_reg"] = y_reg.to_numpy()
        chunk["source"] = "market"
        chunk["symbol"] = symbol
        chunk["ingested_at"] = datetime.now(timezone.utc).isoformat()

        old = self._load_market()
        merged = pd.concat([old, chunk], ignore_index=True) if not old.empty else chunk
        # Drop exact duplicate feature rows (keep last)
        feat_cols = [c for c in cols if c in merged.columns]
        merged = merged.drop_duplicates(subset=feat_cols + ["y_class"], keep="last")
        self._save_market(merged)
        logger.info("Experience memory +%s market rows (total=%s)", len(chunk), len(merged))
        return len(chunk)

    def ingest_trade_lessons(self, feature_cols: Optional[List[str]] = None) -> int:
        """
        Turn closed trades into permanent lessons.
        Losses are up-weighted (duplicated) so mistakes are not forgotten.
        """
        cols = feature_cols or EXTENDED_FEATURES
        with session_scope() as s:
            trades = list(s.execute(select(Trade).where(Trade.status == "closed")).scalars())
            payload = [
                {
                    "id": t.id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "pnl": float(t.pnl or 0),
                    "pnl_pct": float(t.pnl_pct or 0),
                    "exit_reason": t.exit_reason or "",
                    "strategy": t.strategy or "",
                    "confidence": float(t.confidence or 0),
                    "entry_price": float(t.entry_price or 0),
                    "exit_price": float(t.exit_price or 0) if t.exit_price else None,
                }
                for t in trades
            ]

        if not payload:
            return 0

        # Append raw mistake log (append-only, never deleted)
        existing_ids = set()
        if MISTAKES_PATH.exists():
            for line in MISTAKES_PATH.read_text(encoding="utf-8").splitlines():
                try:
                    existing_ids.add(json.loads(line).get("id"))
                except Exception:
                    continue

        added = 0
        lesson_rows = []
        with MISTAKES_PATH.open("a", encoding="utf-8") as fh:
            for t in payload:
                if t["id"] in existing_ids:
                    continue
                # Label: what direction WAS taken vs outcome
                side = (t["side"] or "").lower()
                direction = 1 if side in {"buy", "long"} else -1
                # If trade lost, the correct class was the opposite (or flat)
                if t["pnl"] < 0:
                    y_class = -direction  # should have been opposite
                    weight_dup = 3  # remember losses strongly
                    kind = "loss"
                elif t["pnl"] > 0:
                    y_class = direction
                    weight_dup = 1
                    kind = "win"
                else:
                    y_class = 0
                    weight_dup = 1
                    kind = "flat"

                record = {**t, "kind": kind, "y_class": y_class, "ts": datetime.now(timezone.utc).isoformat()}
                fh.write(json.dumps(record) + "\n")
                added += 1

                # Synthetic feature-ish lesson row (strategy stats as weak features)
                # Real market features are preferred; this keeps a durable error signal.
                row = {c: 0.0 for c in cols}
                row["returns"] = float(t["pnl_pct"] or 0) / 100.0
                row["rsi_14"] = 30.0 if y_class == 1 else (70.0 if y_class == -1 else 50.0)
                row["y_class"] = y_class
                row["y_reg"] = float(t["pnl_pct"] or 0) / 100.0
                row["source"] = "trade_lesson"
                row["symbol"] = t["symbol"]
                row["ingested_at"] = record["ts"]
                for _ in range(weight_dup):
                    lesson_rows.append(row)

        if lesson_rows:
            old = self._load_market()
            chunk = pd.DataFrame(lesson_rows)
            merged = pd.concat([old, chunk], ignore_index=True) if not old.empty else chunk
            self._save_market(merged)

        logger.info("Ingested %s new trade lessons into lifelong memory", added)
        return added

    def training_xy(
        self,
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Full cumulative dataset for continual learning."""
        cols = feature_cols or EXTENDED_FEATURES
        df = self._load_market()
        if df.empty:
            return pd.DataFrame(columns=cols), pd.Series(dtype=int)
        present = [c for c in cols if c in df.columns]
        X = df[present].replace([np.inf, -np.inf], np.nan).dropna()
        y = df.loc[X.index, "y_class"].astype(int)
        # Align columns
        for c in cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[cols]
        return X, y

    def stats(self) -> Dict[str, Any]:
        df = self._load_market()
        mistakes = 0
        if MISTAKES_PATH.exists():
            mistakes = sum(1 for _ in MISTAKES_PATH.open(encoding="utf-8"))
        return {
            "market_rows": int(len(df)),
            "trade_lessons_logged": mistakes,
            "path": str(EXP_DIR),
            "never_deleted_by_brain_rotation": True,
        }
