"""
Nightly self-update cycle (03:00):
1 data collection 2 error analysis 3 retrain 4 GA 5 feature selection
6 A/B 7 shadow 8 correlation 9 seasonality 10 telegram report 11 checkpoints
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytz
from sqlalchemy import select

from automation.backup import backup_database, backup_models
from config import CACHE_DIR, DATA_DIR, get_settings
from core.database import Trade, audit, session_scope
from core.market_data import MarketDataService
from ml.ab_testing import ABTester
from ml.feature_selection import select_features
from ml.genetic import GeneticOptimizer
from ml.models.trainer import ModelTrainer
from ml.shadow_mode import ShadowMode
from strategies.ensemble import EnsembleStrategy
from telegram_bot.notifier import TelegramNotifier

logger = logging.getLogger("ai.daily_update")


class DailyUpdatePipeline:
    def __init__(
        self,
        market: Optional[MarketDataService] = None,
        trainer: Optional[ModelTrainer] = None,
        ensemble: Optional[EnsembleStrategy] = None,
        shadow: Optional[ShadowMode] = None,
        notifier: Optional[TelegramNotifier] = None,
    ) -> None:
        self.settings = get_settings()
        self.tz = pytz.timezone(self.settings.timezone)
        self.market = market or MarketDataService()
        self.trainer = trainer or ModelTrainer(self.market)
        self.ensemble = ensemble or EnsembleStrategy()
        self.shadow = shadow or ShadowMode(self.ensemble)
        self.notifier = notifier or TelegramNotifier()

    def run(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {"ok": False, "started_at": datetime.now(self.tz).isoformat()}
        audit("daily_update_start")
        backup_database()
        backup_models()

        # 1. Collect day data
        market_data = self.market.get_multi_ohlcv(self.settings.pairs, interval="15m", limit=500)
        trades = self._collect_trades()
        news = self._collect_news_proxy(market_data)
        report["trades_count"] = len(trades)
        report["symbols"] = list(market_data.keys())
        report["news"] = news

        # 2. Error analysis — lessons are also written into lifelong experience memory
        report["error_analysis"] = self._analyze_losses(trades)
        try:
            from ml.experience_memory import ExperienceMemory

            mem = ExperienceMemory()
            report["lessons_ingested"] = mem.ingest_trade_lessons(
                feature_cols=self.trainer.ensemble.selected_features
            )
            report["experience_before_train"] = mem.stats()
        except Exception as exc:
            logger.warning("Lesson ingest failed: %s", exc)
            report["lessons_ingested"] = 0

        # Combined frame for ML/GA
        frames = [df for df in market_data.values() if len(df) > 100]
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        # 3. Retrain ML on FULL memory + warm-start (does not forget past mistakes)
        train_result = self.trainer.train_all(epochs=10)
        report.update(
            {
                "ok": bool(train_result.get("ok")),
                "metrics": train_result.get("metrics"),
                "version": train_result.get("version"),
                "paths": train_result.get("paths"),
                "experience": train_result.get("experience"),
                "continual_learning": True,
            }
        )

        # 4. Genetic Algorithm
        ga_fitness = None
        if not combined.empty:
            ga = GeneticOptimizer(population=12, generations=6)
            best = ga.evolve(combined.tail(400))
            ga.apply_to_ensemble(self.ensemble)
            ga_fitness = best.get("fitness")
            report["ga_params"] = best.get("params")
            report["ga_fitness"] = ga_fitness
            self._save_json("ga_best.json", best)

        # 5. Feature selection
        if not combined.empty:
            selected, scores = select_features(combined.tail(500), top_k=12)
            self.trainer.ensemble.selected_features = selected
            self.trainer.ensemble.save()
            report["selected_features"] = selected
            self._save_json("feature_scores.json", scores)

        # 6. A/B test
        if not combined.empty:
            control = {}
            challenger = report.get("ga_params") or {}
            ab = ABTester().compare(
                combined.tail(350),
                {"control": control, "challenger": challenger},
            )
            report["ab_test"] = ab
            if ab.get("winner") == "challenger" and challenger:
                from ml.genetic import _apply_params

                _apply_params(self.ensemble, challenger)

        # 7. Shadow mode analysis
        live_equity = sum(t.get("pnl", 0) or 0 for t in trades)
        # Normalize roughly
        shadow_decision = self.shadow.should_switch(live_equity=live_equity / 1000.0)
        report["shadow"] = shadow_decision
        report["shadow_switch"] = shadow_decision.get("recommend_switch", False)

        # 8. Correlation matrix
        corr = self.market.correlation_matrix(self.settings.pairs)
        if not corr.empty:
            corr_path = CACHE_DIR / "correlation_matrix.csv"
            corr.to_csv(corr_path)
            report["correlation_path"] = str(corr_path)

        # 9. Seasonal patterns
        seasonality = self._seasonal_patterns(market_data)
        report["seasonality"] = seasonality
        self._save_json("seasonality.json", seasonality)

        # 10. Telegram report
        self.notifier.retrain_report(report)
        # Also store full report
        self._save_json(
            f"daily_report_{datetime.now(self.tz).strftime('%Y%m%d')}.json",
            report,
        )

        # 11. Checkpoints already saved by trainer
        report["finished_at"] = datetime.now(self.tz).isoformat()
        audit("daily_update_done", details=json.dumps({"ok": report["ok"], "version": report.get("version")}))
        logger.info("Daily update finished ok=%s", report["ok"])
        return report

    def _collect_trades(self) -> List[Dict[str, Any]]:
        day = (datetime.now(self.tz) - timedelta(days=1)).strftime("%Y-%m-%d")
        with session_scope() as s:
            rows = list(s.execute(select(Trade)).scalars())
            out = []
            for r in rows:
                ts = r.closed_at or r.opened_at
                if ts and ts.astimezone(self.tz).strftime("%Y-%m-%d") in {
                    day,
                    datetime.now(self.tz).strftime("%Y-%m-%d"),
                }:
                    out.append(
                        {
                            "id": r.id,
                            "symbol": r.symbol,
                            "side": r.side,
                            "pnl": r.pnl,
                            "pnl_pct": r.pnl_pct,
                            "exit_reason": r.exit_reason,
                            "strategy": r.strategy,
                            "status": r.status,
                        }
                    )
            return out

    def _analyze_losses(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        losses = [t for t in trades if (t.get("pnl") or 0) < 0]
        by_reason: Dict[str, int] = {}
        by_strategy: Dict[str, int] = {}
        for t in losses:
            by_reason[t.get("exit_reason") or "unknown"] = by_reason.get(t.get("exit_reason") or "unknown", 0) + 1
            by_strategy[t.get("strategy") or "unknown"] = by_strategy.get(t.get("strategy") or "unknown", 0) + 1
        return {
            "loss_count": len(losses),
            "by_reason": by_reason,
            "by_strategy": by_strategy,
            "examples": losses[:10],
        }

    def _collect_news_proxy(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Proxy 'news' via extreme return events when external news API is absent."""
        events = []
        for sym, df in market_data.items():
            if df.empty:
                continue
            rets = df["close"].pct_change().tail(96)
            for ts, r in rets.items():
                if abs(float(r)) >= 0.02:
                    events.append(
                        {
                            "symbol": sym,
                            "timestamp": str(df.loc[ts, "timestamp"]) if "timestamp" in df.columns else str(ts),
                            "return": float(r),
                            "tag": "high_move",
                        }
                    )
        return events[:50]

    def _seasonal_patterns(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        patterns: Dict[str, Any] = {}
        for sym, df in market_data.items():
            if df.empty or "timestamp" not in df.columns:
                continue
            tmp = df.copy()
            tmp["hour"] = pd.to_datetime(tmp["timestamp"], utc=True).dt.hour
            tmp["ret"] = tmp["close"].pct_change()
            by_hour = tmp.groupby("hour")["ret"].mean().to_dict()
            patterns[sym] = {str(k): float(v) for k, v in by_hour.items() if v == v}
        return patterns

    def _save_json(self, name: str, payload: Any) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = DATA_DIR / name

        def default(o):
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            return str(o)

        path.write_text(json.dumps(payload, indent=2, default=default), encoding="utf-8")
