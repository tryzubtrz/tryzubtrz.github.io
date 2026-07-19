"""
Central configuration for the AI Trading Platform.
All secrets come from environment / encrypted .env — never hardcode keys.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
BACKUPS_DIR = DATA_DIR / "backups"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = BASE_DIR / "logs"
CERTS_DIR = BASE_DIR / "certs"
DB_PATH = DATA_DIR / "trading.db"


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Bybit
    bybit_api_key: str = ""
    bybit_api_secret: str = ""
    bybit_testnet: bool = True
    bybit_base_url: str = "https://api-testnet.bybit.com"
    bybit_ws_public: str = "wss://stream-testnet.bybit.com/v5/public/linear"
    bybit_ws_private: str = "wss://stream-testnet.bybit.com/v5/private"

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Dashboard / auth
    dashboard_username: str = "admin"
    dashboard_password_hash: str = ""
    jwt_secret: str = Field(default="change-me-to-a-long-random-string")
    session_ttl_hours: int = 24
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8080
    force_https: bool = True
    rate_limit: str = "60/minute"

    # Trading risk
    initial_balance: float = 10_000.0
    max_daily_loss_pct: float = 3.0
    max_position_pct: float = 5.0
    default_leverage: int = 5
    trading_pairs: str = "BTCUSDT,ETHUSDT,SOLUSDT"
    min_confidence: float = 0.62
    take_profit_pct: float = 1.5
    stop_loss_pct: float = 0.8
    trailing_stop_pct: float = 0.4
    max_open_positions: int = 5
    daily_goal_pcts: str = "1.0,3.0,5.0"

    # Security
    env_encryption_key: str = ""

    # Runtime
    log_level: str = "INFO"
    timezone: str = "Europe/Kyiv"
    healthcheck_interval_sec: int = 300
    watchdog_max_restarts_per_hour: int = 5
    watchdog_restart_delay_sec: int = 30
    model_keep_versions: int = 30
    log_retention_days: int = 30
    log_max_bytes: int = 100 * 1024 * 1024  # 100MB

    # Scheduler (local timezone)
    daily_update_hour: int = 3
    daily_update_minute: int = 0
    morning_summary_hour: int = 8
    morning_summary_minute: int = 0
    retrain_report_hour: int = 8
    retrain_report_minute: int = 5

    @field_validator("bybit_testnet", mode="before")
    @classmethod
    def parse_bool(cls, v):  # noqa: ANN001
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "on"}
        return bool(v)

    @property
    def pairs(self) -> List[str]:
        return [p.strip().upper() for p in self.trading_pairs.split(",") if p.strip()]

    @property
    def daily_goals(self) -> List[float]:
        return [float(x.strip()) for x in self.daily_goal_pcts.split(",") if x.strip()]

    @property
    def rest_base(self) -> str:
        if self.bybit_testnet:
            return "https://api-testnet.bybit.com"
        return self.bybit_base_url or "https://api.bybit.com"

    @property
    def ws_public(self) -> str:
        if self.bybit_testnet:
            return "wss://stream-testnet.bybit.com/v5/public/linear"
        return "wss://stream.bybit.com/v5/public/linear"

    @property
    def ws_private(self) -> str:
        if self.bybit_testnet:
            return "wss://stream-testnet.bybit.com/v5/private"
        return "wss://stream.bybit.com/v5/private"

    def ensure_directories(self) -> None:
        for path in (DATA_DIR, MODELS_DIR, BACKUPS_DIR, CACHE_DIR, LOGS_DIR, CERTS_DIR):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    # Sync public/private WS defaults when testnet flag flips
    if settings.bybit_testnet:
        os.environ.setdefault("BYBIT_BASE_URL", settings.rest_base)
    return settings


# Convenience module-level aliases used across the codebase
settings = get_settings()
