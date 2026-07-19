"""
Resume state across restarts.

The bot does NOT retrain from zero on boot:
- SQLite keeps trades, daily risk, audits, signals
- data/models/* holds the current "brain"
- BotState keys track last scan / last retrain / session
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.database import get_state, set_state, audit

logger = logging.getLogger("trading.persistence")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mark_started(resume: bool = True) -> Dict[str, Any]:
    prev = get_state("last_shutdown_at", "")
    last_scan = get_state("last_scan_at", "")
    last_retrain = get_state("last_retrain_at", "")
    brain_version = get_state("brain_version", "")
    set_state("last_started_at", utc_now_iso())
    set_state("running", "1")
    info = {
        "resumed": resume,
        "previous_shutdown": prev or None,
        "last_scan_at": last_scan or None,
        "last_retrain_at": last_retrain or None,
        "brain_version": brain_version or None,
    }
    audit("engine_resume" if (prev or last_scan or brain_version) else "engine_fresh_start", details=json.dumps(info))
    logger.info("Persistence boot: %s", info)
    return info


def mark_stopped() -> None:
    set_state("last_shutdown_at", utc_now_iso())
    set_state("running", "0")
    audit("engine_graceful_stop")


def mark_scan() -> None:
    set_state("last_scan_at", utc_now_iso())


def mark_retrain(version: str, metrics: Optional[Dict[str, Any]] = None) -> None:
    set_state("last_retrain_at", utc_now_iso())
    set_state("brain_version", str(version))
    if metrics is not None:
        set_state("last_retrain_metrics", json.dumps(metrics, default=str))


def snapshot() -> Dict[str, str]:
    keys = [
        "last_started_at",
        "last_shutdown_at",
        "last_scan_at",
        "last_retrain_at",
        "brain_version",
        "running",
    ]
    return {k: get_state(k, "") for k in keys}
