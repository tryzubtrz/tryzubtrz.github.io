"""APScheduler jobs: daily update, morning summary, retrain report, healthcheck."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import select

from config import get_settings
from core.database import DailyStats, Trade, session_scope
from telegram_bot.notifier import TelegramNotifier

logger = logging.getLogger("trading.scheduler")


class BotScheduler:
    def __init__(
        self,
        daily_update_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        healthcheck_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        notifier: Optional[TelegramNotifier] = None,
        last_retrain_report: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        self.settings = get_settings()
        self.tz = pytz.timezone(self.settings.timezone)
        self.scheduler = BackgroundScheduler(timezone=self.tz)
        self.daily_update_fn = daily_update_fn
        self.healthcheck_fn = healthcheck_fn
        self.notifier = notifier or TelegramNotifier()
        self.last_retrain_report = last_retrain_report
        self._last_update_result: Dict[str, Any] = {}

    def start(self) -> None:
        s = self.settings
        self.scheduler.add_job(
            self._run_daily_update,
            "cron",
            hour=s.daily_update_hour,
            minute=s.daily_update_minute,
            id="daily_update",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._morning_summary,
            "cron",
            hour=s.morning_summary_hour,
            minute=s.morning_summary_minute,
            id="morning_summary",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._retrain_report,
            "cron",
            hour=s.retrain_report_hour,
            minute=s.retrain_report_minute,
            id="retrain_report",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._health,
            "interval",
            seconds=s.healthcheck_interval_sec,
            id="healthcheck",
            replace_existing=True,
        )
        self.scheduler.start()
        logger.info("Scheduler started (tz=%s)", self.settings.timezone)

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    def _run_daily_update(self) -> None:
        if not self.daily_update_fn:
            return
        try:
            self._last_update_result = self.daily_update_fn() or {}
        except Exception as exc:
            logger.exception("Daily update failed")
            self.notifier.health_alert(f"Daily update failed: {exc}")

    def _morning_summary(self) -> None:
        yesterday = (datetime.now(self.tz) - timedelta(days=1)).strftime("%Y-%m-%d")
        with session_scope() as s:
            row = s.execute(select(DailyStats).where(DailyStats.day == yesterday)).scalar_one_or_none()
            trades = list(
                s.execute(select(Trade).where(Trade.status == "closed")).scalars()
            )
            day_trades = [
                t
                for t in trades
                if t.closed_at and t.closed_at.astimezone(self.tz).strftime("%Y-%m-%d") == yesterday
            ]
            wins = sum(1 for t in day_trades if (t.pnl or 0) > 0)
            losses = sum(1 for t in day_trades if (t.pnl or 0) <= 0)
            stats = {
                "day": yesterday,
                "pnl": row.pnl if row else sum(t.pnl or 0 for t in day_trades),
                "pnl_pct": row.pnl_pct if row else 0.0,
                "trades": len(day_trades),
                "wins": wins,
                "losses": losses,
                "goals_hit": row.goals_hit if row else "",
                "loss_limit_hit": bool(row.loss_limit_hit) if row else False,
            }
        self.notifier.morning_summary(stats)

    def _retrain_report(self) -> None:
        report = self._last_update_result
        if self.last_retrain_report:
            report = self.last_retrain_report() or report
        if not report:
            # Try load from disk
            from pathlib import Path
            from config import DATA_DIR
            import json

            files = sorted(DATA_DIR.glob("daily_report_*.json"), reverse=True)
            if files:
                report = json.loads(files[0].read_text(encoding="utf-8"))
        if report:
            self.notifier.retrain_report(report)

    def _health(self) -> None:
        if self.healthcheck_fn:
            try:
                self.healthcheck_fn()
            except Exception as exc:
                logger.exception("Healthcheck job error")
                self.notifier.health_alert(str(exc))
