"""
Watchdog process: monitors main bot and auto-restarts on crash.
Max 5 restarts/hour, 30s delay, Telegram alerts.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, List, Optional

from config import BASE_DIR, get_settings
from telegram_bot.notifier import TelegramNotifier

logger = logging.getLogger("trading.watchdog")


class Watchdog:
    def __init__(
        self,
        command: Optional[List[str]] = None,
        notifier: Optional[TelegramNotifier] = None,
    ) -> None:
        self.settings = get_settings()
        self.command = command or [sys.executable, str(BASE_DIR / "main.py"), "--child"]
        self.notifier = notifier or TelegramNotifier()
        self.restart_times: Deque[float] = deque()
        self.proc: Optional[subprocess.Popen] = None

    def _prune_restarts(self) -> None:
        now = time.time()
        while self.restart_times and now - self.restart_times[0] > 3600:
            self.restart_times.popleft()

    def can_restart(self) -> bool:
        self._prune_restarts()
        return len(self.restart_times) < self.settings.watchdog_max_restarts_per_hour

    def start_child(self) -> None:
        logger.info("Starting child: %s", self.command)
        self.proc = subprocess.Popen(
            self.command,
            cwd=str(BASE_DIR),
            env=os.environ.copy(),
        )

    def run_forever(self) -> None:
        attempt = 0
        self.start_child()
        while True:
            assert self.proc is not None
            code = self.proc.wait()
            ts = datetime.now(timezone.utc).isoformat()
            details = f"exit_code={code} at {ts}"
            logger.error("Child crashed/exited: %s", details)
            self.notifier.crash(details)

            if not self.can_restart():
                msg = (
                    f"Max restarts ({self.settings.watchdog_max_restarts_per_hour}/hour) "
                    "reached — waiting 1 hour"
                )
                logger.error(msg)
                self.notifier.health_alert(msg)
                time.sleep(3600)
                self.restart_times.clear()

            delay = self.settings.watchdog_restart_delay_sec
            logger.info("Restarting in %ss...", delay)
            time.sleep(delay)
            attempt += 1
            self.restart_times.append(time.time())
            self.notifier.restart(attempt, details)
            self.start_child()


def main() -> None:
    from utils.logging_config import setup_logging

    setup_logging()
    Watchdog().run_forever()


if __name__ == "__main__":
    main()
