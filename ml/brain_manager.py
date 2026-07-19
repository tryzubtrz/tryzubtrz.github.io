"""
Daily "brain" file snapshots (disk hygiene ONLY).

IMPORTANT:
- Deleting old backups does NOT erase knowledge / mistakes.
- Lifelong learning lives in data/experience/ (never pruned here)
  and in the evolving live weights under data/models/.
- prune_old_brains() only removes duplicate snapshot folders to save disk.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from sqlalchemy import delete, desc, select

from config import BACKUPS_DIR, MODELS_DIR, get_settings
from core.database import ModelCheckpoint, session_scope
from core.persistence import mark_retrain

logger = logging.getLogger("ai.brain")


def list_brain_backups() -> List[Path]:
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        [p for p in BACKUPS_DIR.iterdir() if p.is_dir() and p.name.startswith("models_")],
        reverse=True,
    )


def prune_old_brains(keep: int | None = None) -> List[str]:
    """Delete old brain backups; keep only the newest `keep` folders."""
    settings = get_settings()
    keep = settings.model_keep_versions if keep is None else keep
    keep = max(0, int(keep))
    deleted = []
    backups = list_brain_backups()
    for old in backups[keep:]:
        shutil.rmtree(old, ignore_errors=True)
        deleted.append(old.name)
        logger.info("Deleted old brain backup: %s", old.name)
    # Also prune DB checkpoint rows beyond keep*3 artifacts
    with session_scope() as s:
        rows = list(
            s.execute(select(ModelCheckpoint).order_by(desc(ModelCheckpoint.created_at))).scalars()
        )
        # Keep metadata for current live + recent backups
        max_rows = max(keep * 3, 3)
        for row in rows[max_rows:]:
            s.delete(row)
    return deleted


def promote_new_brain(version: str, metrics: dict) -> None:
    """Record that today's brain is live and wipe surplus history."""
    mark_retrain(version, metrics)
    deleted = prune_old_brains()
    logger.info(
        "New brain active version=%s deleted_old=%s",
        version,
        deleted,
    )


def current_brain_files() -> List[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted(p.name for p in MODELS_DIR.iterdir() if p.is_file() and p.name != ".gitkeep")
