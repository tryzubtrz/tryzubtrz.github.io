"""FastAPI dashboard — HTTPS on :8080."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from security.rate_limiter import limiter

logger = logging.getLogger("trading.dashboard")
STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(engine) -> FastAPI:
    app = FastAPI(title="AI Trading Platform", version="1.0.0")
    app.state.engine = engine
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    from dashboard.routes import router

    app.include_router(router)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index():
        index_path = STATIC_DIR / "index.html"
        return FileResponse(index_path)

    @app.get("/api/ping")
    async def ping():
        return {"ok": True, "service": "ai-trading-platform"}

    @app.middleware("http")
    async def force_https_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.exception("Dashboard error on %s", request.url.path)
        return JSONResponse({"detail": str(exc)}, status_code=500)

    return app
