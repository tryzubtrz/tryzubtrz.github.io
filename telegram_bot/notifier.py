"""Telegram notifications for trades, goals, anomalies, daily reports."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from config import get_settings
from core.database import audit

logger = logging.getLogger("trading.telegram")


class TelegramNotifier:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.token = self.settings.telegram_bot_token
        self.chat_id = self.settings.telegram_chat_id
        self.enabled = bool(self.token and self.chat_id)

    @property
    def api_base(self) -> str:
        return f"https://api.telegram.org/bot{self.token}"

    def send(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self.enabled:
            logger.debug("Telegram disabled — message skipped: %s", text[:80])
            return False
        try:
            with httpx.Client(timeout=20) as client:
                resp = client.post(
                    f"{self.api_base}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": parse_mode,
                        "disable_web_page_preview": True,
                    },
                )
                ok = resp.status_code == 200 and resp.json().get("ok")
                if not ok:
                    logger.error("Telegram send failed: %s", resp.text)
                else:
                    audit("telegram_send", details=text[:200])
                return bool(ok)
        except Exception as exc:
            logger.error("Telegram error: %s", exc)
            return False

    def trade_closed(self, trade: Dict[str, Any]) -> None:
        pnl = trade.get("pnl", 0)
        emoji = "✅" if pnl >= 0 else "❌"
        self.send(
            f"{emoji} <b>Угоду закрито</b>\n"
            f"Пара: <code>{trade.get('symbol')}</code>\n"
            f"Сторона: {trade.get('side')}\n"
            f"P&amp;L: <b>{pnl:.4f}</b> ({trade.get('pnl_pct', 0):.2f}%)\n"
            f"Причина виходу: <i>{trade.get('exit_reason')}</i>\n"
            f"Стратегія: {trade.get('strategy')}"
        )

    def daily_goal(self, goal_pct: float, pnl_pct: float) -> None:
        self.send(
            f"🎯 <b>Денну ціль досягнуто: +{goal_pct:g}%</b>\n"
            f"Поточний результат дня: <b>{pnl_pct:.2f}%</b>"
        )

    def loss_limit(self, pnl_pct: float) -> None:
        self.send(
            f"🛑 <b>Денний ліміт збитку спрацював</b>\n"
            f"P&amp;L дня: <b>{pnl_pct:.2f}%</b>\n"
            f"Торгівлю зупинено до наступного дня."
        )

    def anomaly(self, anomaly: Dict[str, Any]) -> None:
        self.send(
            f"⚠️ <b>Аномалія на ринку</b>\n"
            f"Пара: <code>{anomaly.get('symbol')}</code>\n"
            f"Тип: {anomaly.get('anomaly_type')}\n"
            f"Серйозність: {anomaly.get('severity')}\n"
            f"Деталі: {anomaly.get('details')}"
        )

    def morning_summary(self, stats: Dict[str, Any]) -> None:
        self.send(
            f"☀️ <b>Підсумок попереднього дня</b>\n"
            f"День: {stats.get('day')}\n"
            f"P&amp;L: <b>{stats.get('pnl', 0):.4f}</b> ({stats.get('pnl_pct', 0):.2f}%)\n"
            f"Угод: {stats.get('trades', 0)} | Wins: {stats.get('wins', 0)} | "
            f"Losses: {stats.get('losses', 0)}\n"
            f"Цілі: {stats.get('goals_hit') or '—'}\n"
            f"Ліміт збитку: {'так' if stats.get('loss_limit_hit') else 'ні'}"
        )

    def retrain_report(self, report: Dict[str, Any]) -> None:
        metrics = report.get("metrics") or {}
        xgb = metrics.get("xgboost") or {}
        lstm = metrics.get("lstm") or {}
        self.send(
            f"🧠 <b>Результат нічного перенавчання</b>\n"
            f"Статус: {'OK' if report.get('ok') else 'FAIL'}\n"
            f"XGBoost acc: {xgb.get('accuracy', 0):.3f} | f1: {xgb.get('f1', 0):.3f}\n"
            f"LSTM acc: {lstm.get('accuracy', 0):.3f}\n"
            f"Версія: {report.get('version', '—')}\n"
            f"GA fitness: {report.get('ga_fitness', '—')}\n"
            f"Shadow switch: {report.get('shadow_switch', False)}"
        )

    def crash(self, details: str) -> None:
        self.send(f"💥 <b>Краш процесу</b>\n{details}")

    def restart(self, attempt: int, details: str = "") -> None:
        self.send(f"🔄 <b>Автоперезапуск #{attempt}</b>\n{details}")

    def health_alert(self, details: str) -> None:
        self.send(f"🩺 <b>Healthcheck проблема</b>\n{details}")
