#!/usr/bin/env python3
"""
AI Trading Platform — entrypoint.
Run: python main.py
Dashboard: https://localhost:8080
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from typing import Any, Dict, Optional

import uvicorn

from config import get_settings
from core.anomaly import AnomalyDetector
from core.bybit_client import get_bybit_client
from core.database import Signal, audit, init_db, session_scope
from core.market_data import MarketDataService
from core.order_executor import OrderExecutor
from core.persistence import mark_scan, mark_started, mark_stopped, snapshot
from core.position_manager import PositionManager
from core.risk_manager import RiskManager
from ml.brain_manager import current_brain_files
from automation.daily_update import DailyUpdatePipeline
from automation.healthcheck import HealthChecker
from automation.scheduler import BotScheduler
from dashboard.app import create_app
from ml.models.trainer import ModelTrainer
from ml.shadow_mode import ShadowMode
from security.certs import generate_self_signed
from security.env_crypto import ensure_encrypted_env
from strategies.ensemble import EnsembleStrategy
from telegram_bot.notifier import TelegramNotifier
from utils.logging_config import setup_logging

logger = logging.getLogger("trading.main")


class TradingEngine:
    """Central orchestrator for scanning, execution, ML, and alerts."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = get_bybit_client()
        self.market = MarketDataService(self.client)
        self.risk = RiskManager()
        self.positions = PositionManager()
        self.executor = OrderExecutor(
            client=self.client,
            risk=self.risk,
            positions=self.positions,
            market=self.market,
        )
        self.ensemble = EnsembleStrategy()
        self.trainer = ModelTrainer(self.market)
        self.trainer.ensemble.load()
        self.shadow = ShadowMode(EnsembleStrategy())
        self.anomaly = AnomalyDetector()
        self.notifier = TelegramNotifier()
        self.latest_signals: Dict[str, Any] = {}
        self._running = False
        self._loop_thread: Optional[threading.Thread] = None
        self.health = HealthChecker(
            client=self.client,
            positions=self.positions,
            notifier=self.notifier,
            service_flags={"engine": True, "scheduler": True, "dashboard": True},
        )
        self.daily = DailyUpdatePipeline(
            market=self.market,
            trainer=self.trainer,
            ensemble=self.ensemble,
            shadow=self.shadow,
            notifier=self.notifier,
        )
        self.scheduler = BotScheduler(
            daily_update_fn=self.daily.run,
            healthcheck_fn=self.health.run,
            notifier=self.notifier,
            last_retrain_report=lambda: getattr(self, "_last_daily", {}),
        )
        self._last_daily: Dict[str, Any] = {}

    def status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "testnet": self.settings.bybit_testnet,
            "pairs": self.settings.pairs,
            "egress_ip": self.client.egress_ip,
            "open_positions": self.positions.open_count(),
            "risk": self.risk.snapshot(),
            "signals": self.latest_signals,
            "persistence": snapshot(),
            "brain_files": current_brain_files(),
            "ml_loaded": {
                "xgb": self.trainer.ensemble.xgb.trained,
                "lstm": self.trainer.ensemble.lstm.trained,
            },
        }

    def scan_once(self) -> Dict[str, Any]:
        """One market scan cycle across configured pairs."""
        results = []
        for symbol in self.settings.pairs:
            try:
                df = self.market.fetch_ohlcv(symbol, interval="15m", limit=250)
                if df.empty:
                    continue

                # Anomalies
                for a in self.anomaly.analyze(symbol, df):
                    self.notifier.anomaly(a)

                # ML vote
                ml_sig = self.trainer.ensemble.predict(symbol, df)
                self.ensemble.set_ml_vote(ml_sig)

                # Live ensemble
                sig = self.ensemble.generate(symbol, df)
                self.latest_signals[symbol] = sig.to_dict()

                with session_scope() as s:
                    s.add(
                        Signal(
                            symbol=symbol,
                            direction=sig.direction,
                            confidence=sig.confidence,
                            strategy=sig.strategy,
                            features_json=str(sig.meta or {}),
                        )
                    )

                # Shadow parallel
                price = float(df["close"].iloc[-1])
                self.shadow.step(symbol, df, price)

                # Execute
                exec_result = self.executor.execute_signal(sig.to_dict())
                if exec_result.get("executed"):
                    audit("signal_executed", details=str(exec_result))

                results.append(
                    {
                        "symbol": symbol,
                        "signal": sig.to_dict(),
                        "ml": ml_sig.to_dict(),
                        "execution": exec_result,
                    }
                )
            except Exception as exc:
                logger.exception("Scan failed for %s", symbol)
                results.append({"symbol": symbol, "error": str(exc)})

        # Sync closed positions / notify
        closed_events = self.executor.sync_exits_from_exchange()
        for ev in closed_events:
            trade = ev["trade"]
            self.notifier.trade_closed(trade)
            for goal in ev.get("goals_hit") or []:
                self.notifier.daily_goal(goal, self.risk.current_pnl_pct())
            if self.risk.state.loss_limit_hit:
                self.notifier.loss_limit(self.risk.current_pnl_pct())

        mark_scan()
        return {"results": results, "closed": closed_events, "risk": self.risk.snapshot()}

    def _loop(self) -> None:
        while self._running:
            try:
                self.scan_once()
            except Exception:
                logger.exception("Engine loop error")
            time.sleep(60)

    def start(self) -> None:
        self._running = True
        self.health.service_flags["engine"] = True
        # Resume: load is already done in __init__; sync exchange + notify
        resume_info = mark_started(resume=True)
        try:
            self.executor.sync_exits_from_exchange()
        except Exception as exc:
            logger.warning("Resume sync failed: %s", exc)
        self.notifier.resumed(
            {
                **resume_info,
                "brain_files": current_brain_files(),
                "open_positions": self.positions.open_count(),
                "risk": self.risk.snapshot(),
            }
        )
        self.scheduler.start()
        self._loop_thread = threading.Thread(target=self._loop, name="scan-loop", daemon=True)
        self._loop_thread.start()
        audit("engine_start", details=f"pairs={','.join(self.settings.pairs)}")
        logger.info(
            "Trading engine started (resumed brain=%s xgb=%s lstm=%s)",
            resume_info.get("brain_version"),
            self.trainer.ensemble.xgb.trained,
            self.trainer.ensemble.lstm.trained,
        )

    def stop(self) -> None:
        self._running = False
        self.health.service_flags["engine"] = False
        self.scheduler.shutdown()
        mark_stopped()
        audit("engine_stop")
        logger.info("Trading engine stopped")


def run_dashboard(engine: TradingEngine) -> None:
    settings = get_settings()
    app = create_app(engine)
    cert, key = generate_self_signed()
    logger.info(
        "Dashboard listening on https://%s:%s",
        settings.dashboard_host,
        settings.dashboard_port,
    )
    uvicorn.run(
        app,
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        ssl_certfile=str(cert),
        ssl_keyfile=str(key),
        log_level=settings.log_level.lower(),
    )


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="AI Trading Platform")
    parser.add_argument("--child", action="store_true", help="Run as watchdog child")
    parser.add_argument("--watchdog", action="store_true", help="Run under watchdog supervisor")
    parser.add_argument("--once", action="store_true", help="Single scan then exit")
    parser.add_argument("--train", action="store_true", help="Train models then exit")
    parser.add_argument("--daily-update", action="store_true", help="Run nightly pipeline once")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable HTTPS dashboard")
    args = parser.parse_args(argv)

    setup_logging()
    settings = get_settings()
    settings.ensure_directories()
    init_db()

    try:
        ensure_encrypted_env()
    except Exception as exc:
        logger.warning("Env encryption skipped: %s", exc)

    if args.watchdog and not args.child:
        from automation.watchdog import Watchdog

        Watchdog().run_forever()
        return 0

    engine = TradingEngine()

    if args.train:
        print(engine.trainer.train_all())
        return 0
    if args.daily_update:
        print(engine.daily.run())
        return 0
    if args.once:
        print(engine.scan_once())
        return 0

    engine.start()

    def _shutdown(signum, frame):  # noqa: ANN001
        logger.info("Signal %s — shutting down", signum)
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if args.no_dashboard:
        logger.info("Running without dashboard; Ctrl+C to stop")
        while True:
            time.sleep(3600)
    else:
        run_dashboard(engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
