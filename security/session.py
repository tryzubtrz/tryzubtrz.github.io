"""JWT session tokens with 24h TTL by default."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt

from config import get_settings


def create_session_token(subject: str, extra: Optional[Dict[str, Any]] = None) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(hours=settings.session_ttl_hours),
        "type": "session",
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_session_token(token: str) -> Dict[str, Any]:
    settings = get_settings()
    return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])


def is_token_valid(token: str) -> bool:
    try:
        decode_session_token(token)
        return True
    except jwt.PyJWTError:
        return False
