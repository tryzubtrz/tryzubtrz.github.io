"""Periodic health checks: Bybit, balance, positions, services."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.bybit_client import BybitClient
from core.database import audit
from core.position_manager import PositionManager
from telegram_bot.notifier import TelegramNotifier

logger = logging.getLogger("trading.health")


class HealthChecker:
    def __init__(
        self,
        client: Optional[BybitClient] = None,
        positions: Optional[PositionManager] = None,
        notifier: Optional[TelegramNotifier] = None,
        service_flags: Optional[Dict[str, bool]] = None,
    ) -> None:
        self.client = client
        self.positions = positions or PositionManager()
        self.notifier = notifier or TelegramNotifier()
        self.service_flags = service_flags or {}

    def run(self) -> Dict[str, Any]:
        issues: List[str] = []
        details: Dict[str, Any] = {"ok": True, "issues": [], "balance": None, "positions": 0}

        # Bybit connectivity
        try:
            if self.client is None:
                from core.bybit_client import get_bybit_client

                self.client = get_bybit_client()
            st = self.client.get_server_time()
            details["server_time"] = st.get("result", st)
            details["egress_ip"] = self.client.egress_ip
        except Exception as exc:
            issues.append(f"bybit_connection: {exc}")

        # Balance
        try:
            if self.client:
                bal = self.client.get_wallet_balance()
                details["balance"] = bal.get("result")
        except Exception as exc:
            issues.append(f"balance: {exc}")

        # Open positions consistency
        try:
            local = self.positions.list_open()
            details["positions"] = len(local)
            if self.client:
                remote = self.client.get_positions()
                remote_open = [p for p in remote if float(p.get("size") or 0) > 0]
                details["remote_positions"] = len(remote_open)
        except Exception as exc:
            issues.append(f"positions: {exc}")

        # Services
        for name, running in self.service_flags.items():
            if not running:
                issues.append(f"service_down:{name}")

        details["issues"] = issues
        details["ok"] = len(issues) == 0
        if issues:
            msg = "\n".join(f"• {i}" for i in issues)
            logger.error("Healthcheck failed: %s", issues)
            audit("healthcheck_fail", details=msg)
            self.notifier.health_alert(msg)
        else:
            audit("healthcheck_ok", details="all_green")
            logger.info("Healthcheck OK")
        return details
