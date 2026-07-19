"""API routes for the trading dashboard."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel

from config import get_settings
from core.database import audit, recent_audits
from dashboard.auth import current_user, login
from security.rate_limiter import limiter

logger = logging.getLogger("trading.dashboard")
router = APIRouter(prefix="/api")


class LoginBody(BaseModel):
    username: str
    password: str


class CloseBody(BaseModel):
    symbol: str
    reason: str = "manual"


def attach_engine(request: Request):
    return request.app.state.engine


@router.post("/login")
@limiter.limit("10/minute")
async def api_login(request: Request, response: Response, body: LoginBody):
    token = login(body.username, body.password)
    audit("dashboard_login", actor=body.username)
    return {"token": token, "expires_hours": get_settings().session_ttl_hours}


@router.get("/status")
@limiter.limit("60/minute")
async def status(request: Request, response: Response, user: str = Depends(current_user)):
    engine = attach_engine(request)
    return engine.status()


@router.get("/trades")
@limiter.limit("60/minute")
async def trades(
    request: Request,
    response: Response,
    user: str = Depends(current_user),
    limit: int = 50,
):
    engine = attach_engine(request)
    return {
        "open": engine.positions.list_open(),
        "closed": engine.positions.list_closed(limit=limit),
    }


@router.get("/risk")
@limiter.limit("60/minute")
async def risk(request: Request, response: Response, user: str = Depends(current_user)):
    engine = attach_engine(request)
    return engine.risk.snapshot()


@router.get("/signals")
@limiter.limit("60/minute")
async def signals(request: Request, response: Response, user: str = Depends(current_user)):
    engine = attach_engine(request)
    return {"latest": engine.latest_signals}


@router.get("/market/{symbol}")
@limiter.limit("60/minute")
async def market(
    symbol: str,
    request: Request,
    response: Response,
    user: str = Depends(current_user),
    interval: str = "15m",
):
    engine = attach_engine(request)
    df = engine.market.fetch_ohlcv(symbol.upper(), interval=interval, limit=120)
    rows = df.tail(120).to_dict(orient="records")
    for r in rows:
        if hasattr(r.get("timestamp"), "isoformat"):
            r["timestamp"] = r["timestamp"].isoformat()
        else:
            r["timestamp"] = str(r.get("timestamp"))
    return {"symbol": symbol.upper(), "candles": rows}


@router.post("/close")
@limiter.limit("20/minute")
async def close_position(
    body: CloseBody,
    request: Request,
    response: Response,
    user: str = Depends(current_user),
):
    engine = attach_engine(request)
    result = engine.executor.close_position(body.symbol.upper(), body.reason)
    audit("manual_close", actor=user, details=str(result))
    return result


@router.post("/scan")
@limiter.limit("10/minute")
async def scan(request: Request, response: Response, user: str = Depends(current_user)):
    engine = attach_engine(request)
    result = engine.scan_once()
    audit("manual_scan", actor=user)
    return result


@router.get("/audits")
@limiter.limit("30/minute")
async def audits(
    request: Request,
    response: Response,
    user: str = Depends(current_user),
    limit: int = 100,
):
    rows = recent_audits(limit=limit)
    return {
        "items": [
            {
                "action": r.action,
                "actor": r.actor,
                "details": r.details,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    }


@router.get("/health")
@limiter.limit("30/minute")
async def health(request: Request, response: Response, user: str = Depends(current_user)):
    engine = attach_engine(request)
    return engine.health.run()


@router.get("/ml")
@limiter.limit("30/minute")
async def ml_info(request: Request, response: Response, user: str = Depends(current_user)):
    engine = attach_engine(request)
    ens = engine.trainer.ensemble
    return {
        "xgb_trained": ens.xgb.trained,
        "lstm_trained": ens.lstm.trained,
        "xgb_metrics": ens.xgb.metrics,
        "lstm_metrics": ens.lstm.metrics,
        "features": ens.selected_features,
        "shadow": {
            "equity": engine.shadow.state.equity,
            "trades": engine.shadow.state.trades,
            "wins": engine.shadow.state.wins,
        },
    }
