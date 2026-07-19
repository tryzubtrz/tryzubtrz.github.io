"""Database and model backup utilities."""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from config import BACKUPS_DIR, DB_PATH, MODELS_DIR, get_settings

logger = logging.getLogger("trading.backup")


def backup_database() -> Path:
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = BACKUPS_DIR / f"trading_{stamp}.db"
    if DB_PATH.exists():
        shutil.copy2(DB_PATH, dest)
        logger.info("DB backup → %s", dest)
    else:
        dest.write_bytes(b"")
    _prune_db_backups()
    return dest


def backup_models() -> Path:
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = BACKUPS_DIR / f"models_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    if MODELS_DIR.exists():
        for f in MODELS_DIR.iterdir():
            if f.is_file() and f.name != ".gitkeep":
                shutil.copy2(f, dest / f.name)
    logger.info("Models backup → %s", dest)
    return dest


def _prune_db_backups(keep: int = 30) -> None:
    files = sorted(BACKUPS_DIR.glob("trading_*.db"), reverse=True)
    for old in files[keep:]:
        old.unlink(missing_ok=True)


def prune_model_versions() -> None:
    settings = get_settings()
    dirs = sorted(
        [p for p in BACKUPS_DIR.iterdir() if p.is_dir() and p.name.startswith("models_")],
        reverse=True,
    )
    for old in dirs[settings.model_keep_versions :]:
        shutil.rmtree(old, ignore_errors=True)
