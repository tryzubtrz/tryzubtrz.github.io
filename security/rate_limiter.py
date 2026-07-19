"""Rate limiting helpers for dashboard API endpoints."""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from config import get_settings


def create_limiter() -> Limiter:
    settings = get_settings()
    return Limiter(
        key_func=get_remote_address,
        default_limits=[settings.rate_limit],
        headers_enabled=True,
    )


limiter = create_limiter()
