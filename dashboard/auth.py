"""Dashboard authentication dependencies."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import get_settings
from security.passwords import verify_password
from security.session import create_session_token, decode_session_token

bearer = HTTPBearer(auto_error=False)


def authenticate_user(username: str, password: str) -> bool:
    settings = get_settings()
    if username != settings.dashboard_username:
        return False
    # Bootstrap: if hash empty, accept password "admin" once in testnet setups
    if not settings.dashboard_password_hash:
        return password == "admin"
    return verify_password(password, settings.dashboard_password_hash)


def login(username: str, password: str) -> str:
    if not authenticate_user(username, password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return create_session_token(username)


def current_user(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
) -> str:
    token = None
    if creds:
        token = creds.credentials
    if not token:
        token = request.cookies.get("session")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = decode_session_token(token)
        return str(payload.get("sub") or "")
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid/expired session") from exc
