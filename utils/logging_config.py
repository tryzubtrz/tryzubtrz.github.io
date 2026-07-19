"""Structured JSON logging with rotation (trading / ai / errors)."""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

try:
    from pythonjsonlogger.json import JsonFormatter
except ImportError:  # pragma: no cover - older package layouts
    from pythonjsonlogger import jsonlogger

    JsonFormatter = jsonlogger.JsonFormatter

from config import LOGS_DIR, get_settings


class TimestampActionFilter(logging.Filter):
    """Ensure every record has an ISO-like timestamp field for audits."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "action"):
            record.action = record.getMessage()[:120]
        return True


def _make_handler(
    path: Path,
    level: int,
    max_bytes: int,
    backup_count: int,
) -> logging.Handler:
    handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    formatter = JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s %(action)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )
    handler.setFormatter(formatter)
    handler.addFilter(TimestampActionFilter())
    return handler


def setup_logging(level: Optional[str] = None) -> None:
    """Configure root + specialized loggers."""
    settings = get_settings()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_level = getattr(logging, (level or settings.log_level).upper(), logging.INFO)

    # ~30 days retention assuming ~1 rotated file/day at max size bound
    backup_count = max(settings.log_retention_days, 1)
    max_bytes = settings.log_max_bytes

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    root.addHandler(console)

    # Dedicated files
    trading_logger = logging.getLogger("trading")
    trading_logger.setLevel(log_level)
    trading_logger.addHandler(
        _make_handler(LOGS_DIR / "trading.log", log_level, max_bytes, backup_count)
    )
    trading_logger.propagate = True

    ai_logger = logging.getLogger("ai")
    ai_logger.setLevel(log_level)
    ai_logger.addHandler(
        _make_handler(LOGS_DIR / "ai.log", log_level, max_bytes, backup_count)
    )
    ai_logger.propagate = True

    error_logger = logging.getLogger("errors")
    error_logger.setLevel(logging.WARNING)
    error_logger.addHandler(
        _make_handler(LOGS_DIR / "errors.log", logging.WARNING, max_bytes, backup_count)
    )
    error_logger.propagate = True

    # Capture warnings into errors log
    logging.captureWarnings(True)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_action(logger: logging.Logger, action: str, level: int = logging.INFO, **extra) -> None:
    """Log a user/system action with timestamp (via formatter) and structured extras."""
    logger.log(level, action, extra={"action": action, **{k: str(v) for k, v in extra.items()}})
