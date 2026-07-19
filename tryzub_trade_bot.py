#!/usr/bin/env python3
"""
================================================================================
TRYZUB TRADE — SINGLE-FILE AI TRADING BOT (Bybit Testnet)
================================================================================
One complete file. Save it, install deps once, run:

  pip install pybit pandas numpy scikit-learn xgboost torch requests
  pip install fastapi uvicorn cryptography bcrypt PyJWT slowapi APScheduler
  pip install python-dotenv sqlalchemy pydantic pydantic-settings httpx pytz rich

  # create .env next to this file (or it will create a template)
  python tryzub_trade_bot.py

Dashboard: https://localhost:8080
Continual learning: data/experience/ is NEVER wiped.
Old brain FILE snapshots only are pruned.
================================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import logging.handlers
import os
import secrets
import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

# ── optional / required third-party ──────────────────────────────────────────
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = DATA / "models"
BACKUPS = DATA / "backups"
CACHE = DATA / "cache"
LOGS = BASE / "logs"
CERTS = BASE / "certs"
EXP = DATA / "experience"
DB_PATH = DATA / "trading.db"
ENV_PATH = BASE / ".env"

for p in (DATA, MODELS, BACKUPS, CACHE, LOGS, CERTS, EXP):
    p.mkdir(parents=True, exist_ok=True)

load_dotenv(ENV_PATH)


# =============================================================================
# CONFIG
# =============================================================================
def env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def env_bool(key: str, default: bool = False) -> bool:
    v = env(key, str(default)).lower()
    return v in {"1", "true", "yes", "on"}


def env_float(key: str, default: float) -> float:
    try:
        return float(env(key, str(default)))
    except ValueError:
        return default


def env_int(key: str, default: int) -> int:
    try:
        return int(env(key, str(default)))
    except ValueError:
        return default


CFG = {
    "bybit_key": env("BYBIT_API_KEY"),
    "bybit_secret": env("BYBIT_API_SECRET"),
    "testnet": env_bool("BYBIT_TESTNET", True),
    "tg_token": env("TELEGRAM_BOT_TOKEN"),
    "tg_chat": env("TELEGRAM_CHAT_ID"),
    "dash_user": env("DASHBOARD_USERNAME", "admin"),
    "dash_pass_hash": env("DASHBOARD_PASSWORD_HASH"),
    "jwt_secret": env("JWT_SECRET", secrets.token_urlsafe(32)),
    "session_hours": env_int("SESSION_TTL_HOURS", 24),
    "initial_balance": env_float("INITIAL_BALANCE", 10000),
    "max_daily_loss_pct": env_float("MAX_DAILY_LOSS_PCT", 3.0),
    "max_position_pct": env_float("MAX_POSITION_PCT", 5.0),
    "leverage": env_int("DEFAULT_LEVERAGE", 5),
    "pairs": [x.strip().upper() for x in env("TRADING_PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",") if x.strip()],
    "min_confidence": env_float("MIN_CONFIDENCE", 0.62),
    "tp_pct": env_float("TAKE_PROFIT_PCT", 1.5),
    "sl_pct": env_float("STOP_LOSS_PCT", 0.8),
    "max_open": env_int("MAX_OPEN_POSITIONS", 5),
    "goals": [1.0, 3.0, 5.0],
    "host": env("DASHBOARD_HOST", "0.0.0.0"),
    "port": env_int("DASHBOARD_PORT", 8080),
    "tz": env("TIMEZONE", "Europe/Kyiv"),
    "model_keep": env_int("MODEL_KEEP_VERSIONS", 1),
    "exp_max": env_int("EXPERIENCE_MAX_SAMPLES", 50000),
    "log_level": env("LOG_LEVEL", "INFO"),
}


def ensure_env_template() -> None:
    if ENV_PATH.exists():
        return
    ENV_PATH.write_text(
        """BYBIT_API_KEY=
BYBIT_API_SECRET=
BYBIT_TESTNET=true
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD_HASH=
JWT_SECRET={jwt}
SESSION_TTL_HOURS=24
INITIAL_BALANCE=10000
MAX_DAILY_LOSS_PCT=3.0
MAX_POSITION_PCT=5.0
DEFAULT_LEVERAGE=5
TRADING_PAIRS=BTCUSDT,ETHUSDT,SOLUSDT
MODEL_KEEP_VERSIONS=1
EXPERIENCE_MAX_SAMPLES=50000
LOG_LEVEL=INFO
TIMEZONE=Europe/Kyiv
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
""".format(jwt=secrets.token_urlsafe(32)),
        encoding="utf-8",
    )
    print(f"Created {ENV_PATH} — fill API keys then rerun.")


# =============================================================================
# LOGGING
# =============================================================================
def setup_logging() -> None:
    level = getattr(logging, CFG["log_level"].upper(), logging.INFO)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.handlers.RotatingFileHandler(LOGS / "trading.log", maxBytes=50_000_000, backupCount=10)
    fh.setFormatter(fmt)
    root.addHandler(fh)


log = logging.getLogger("tryzub")


# =============================================================================
# SECURITY HELPERS
# =============================================================================
def hash_password(password: str) -> str:
    import bcrypt

    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(12)).decode()


def verify_password(password: str, password_hash: str) -> bool:
    import bcrypt

    if not password_hash:
        return password == "admin"
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except Exception:
        return False


def make_token(sub: str) -> str:
    import jwt

    now = datetime.now(timezone.utc)
    return jwt.encode(
        {"sub": sub, "iat": now, "exp": now + timedelta(hours=CFG["session_hours"])},
        CFG["jwt_secret"],
        algorithm="HS256",
    )


def decode_token(token: str) -> Dict[str, Any]:
    import jwt

    return jwt.decode(token, CFG["jwt_secret"], algorithms=["HS256"])


def gen_certs() -> Tuple[Path, Path]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    import ipaddress

    cert_p, key_p = CERTS / "localhost.crt", CERTS / "localhost.key"
    if cert_p.exists() and key_p.exists():
        return cert_p, key_p
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Tryzub Trade"),
        ]
    )
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_p.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_p.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    return cert_p, key_p


# =============================================================================
# DATABASE
# =============================================================================
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine, select, desc
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False)
    side = Column(String(8), nullable=False)
    qty = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    leverage = Column(Integer, default=1)
    pnl = Column(Float, default=0.0)
    pnl_pct = Column(Float, default=0.0)
    status = Column(String(16), default="open")
    strategy = Column(String(64), default="")
    exit_reason = Column(String(128), default="")
    confidence = Column(Float, default=0.0)
    order_id = Column(String(64), default="")
    opened_at = Column(DateTime(timezone=True), default=utcnow)
    closed_at = Column(DateTime(timezone=True))


class DailyStats(Base):
    __tablename__ = "daily_stats"
    id = Column(Integer, primary_key=True, autoincrement=True)
    day = Column(String(10), unique=True)
    start_balance = Column(Float, default=0.0)
    end_balance = Column(Float, default=0.0)
    pnl = Column(Float, default=0.0)
    pnl_pct = Column(Float, default=0.0)
    goals_hit = Column(String(64), default="")
    loss_limit_hit = Column(Boolean, default=False)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(256))
    actor = Column(String(64), default="system")
    details = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=utcnow)


class BotState(Base):
    __tablename__ = "bot_state"
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(64), unique=True)
    value = Column(Text, default="")
    updated_at = Column(DateTime(timezone=True), default=utcnow)


_engine = None
_Session = None


def init_db() -> None:
    global _engine, _Session
    _engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
    _Session = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(_engine)


@contextmanager
def db() -> Generator[Session, None, None]:
    assert _Session is not None
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def audit(action: str, details: str = "", actor: str = "system") -> None:
    with db() as s:
        s.add(AuditLog(action=action, details=details, actor=actor))


def set_state(key: str, value: str) -> None:
    with db() as s:
        row = s.execute(select(BotState).where(BotState.key == key)).scalar_one_or_none()
        if row is None:
            s.add(BotState(key=key, value=value, updated_at=utcnow()))
        else:
            row.value = value
            row.updated_at = utcnow()


def get_state(key: str, default: str = "") -> str:
    with db() as s:
        row = s.execute(select(BotState).where(BotState.key == key)).scalar_one_or_none()
        return row.value if row else default


# =============================================================================
# TELEGRAM
# =============================================================================
class Telegram:
    def __init__(self) -> None:
        self.token = CFG["tg_token"]
        self.chat = CFG["tg_chat"]
        self.on = bool(self.token and self.chat)

    def send(self, text: str) -> None:
        if not self.on:
            log.debug("TG skip: %s", text[:80])
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat, "text": text, "parse_mode": "HTML"},
                timeout=20,
            )
            audit("telegram", text[:200])
        except Exception as exc:
            log.error("Telegram: %s", exc)

    def trade_closed(self, t: dict) -> None:
        e = "✅" if t.get("pnl", 0) >= 0 else "❌"
        self.send(
            f"{e} <b>Угоду закрито</b>\n"
            f"{t.get('symbol')} {t.get('side')}\n"
            f"P&L: <b>{t.get('pnl', 0):.4f}</b> ({t.get('pnl_pct', 0):.2f}%)\n"
            f"Причина: {t.get('exit_reason')}"
        )

    def goal(self, g: float, pct: float) -> None:
        self.send(f"🎯 Денна ціль +{g:g}% досягнута (зараз {pct:.2f}%)")

    def loss_limit(self, pct: float) -> None:
        self.send(f"🛑 Денний ліміт збитку: {pct:.2f}% — торгівлю зупинено")

    def anomaly(self, a: dict) -> None:
        self.send(f"⚠️ Аномалія {a.get('symbol')}: {a.get('type')} — {a.get('details')}")

    def resumed(self, info: dict) -> None:
        self.send(
            "▶️ <b>Бот продовжив роботу</b>\n"
            "Не з нуля — завантажено стан і мозок.\n"
            f"Brain: {info.get('brain_version') or '—'}\n"
            f"Останній скан: {info.get('last_scan_at') or '—'}"
        )

    def retrain(self, report: dict) -> None:
        m = report.get("metrics") or {}
        exp = report.get("experience") or {}
        self.send(
            "🧠 <b>Нічне донавчання</b>\n"
            f"OK={report.get('ok')} ver={report.get('version')}\n"
            f"XGB={((m.get('xgboost') or {}).get('accuracy') or 0):.3f} "
            f"LSTM={((m.get('lstm') or {}).get('accuracy') or 0):.3f}\n"
            f"Пам’ять: {exp.get('market_rows', 0)} | уроки: {exp.get('trade_lessons_logged', 0)}"
        )


# =============================================================================
# BYBIT + MARKET DATA
# =============================================================================
class Bybit:
    def __init__(self) -> None:
        from pybit.unified_trading import HTTP

        self.testnet = CFG["testnet"]
        self.http = HTTP(
            testnet=self.testnet,
            api_key=CFG["bybit_key"] or None,
            api_secret=CFG["bybit_secret"] or None,
            timeout=30,
        )
        self.pub = requests.Session()
        self.ip = self._ip()
        log.info("Bybit ready testnet=%s ip=%s", self.testnet, self.ip)

    def _ip(self) -> str:
        for u in ("https://api.ipify.org", "https://ifconfig.me/ip"):
            try:
                r = self.pub.get(u, timeout=5)
                if r.ok:
                    return r.text.strip()
            except Exception:
                pass
        return "?"

    def _market(self, method: str, **kw):
        from pybit.unified_trading import HTTP

        order = [True, False] if self.testnet else [False, True]
        last = None
        for tn in order:
            try:
                c = HTTP(testnet=tn, timeout=30)
                return getattr(c, method)(**kw)
            except Exception as exc:
                last = exc
                if "403" in str(exc) or "usa" in str(exc).lower():
                    continue
                raise
        raise last  # type: ignore

    def kline(self, symbol: str, interval: str = "15", limit: int = 200) -> list:
        r = self._market("get_kline", category="linear", symbol=symbol, interval=interval, limit=limit)
        return r.get("result", {}).get("list", []) or []

    def ticker(self, symbol: str) -> float:
        r = self._market("get_tickers", category="linear", symbol=symbol)
        lst = r.get("result", {}).get("list", []) or []
        return float(lst[0]["lastPrice"]) if lst else 0.0

    def balance(self) -> float:
        try:
            r = self.http.get_wallet_balance(accountType="UNIFIED")
            for c in (r.get("result", {}).get("list", []) or [{}])[0].get("coin", []) or []:
                if c.get("coin") == "USDT":
                    return float(c.get("walletBalance") or 0)
        except Exception as exc:
            log.warning("balance: %s", exc)
        return CFG["initial_balance"]

    def set_leverage(self, symbol: str, lev: int) -> None:
        try:
            self.http.set_leverage(
                category="linear", symbol=symbol, buyLeverage=str(lev), sellLeverage=str(lev)
            )
        except Exception as exc:
            if "110043" not in str(exc) and "not modified" not in str(exc).lower():
                log.warning("leverage: %s", exc)

    def place(self, symbol: str, side: str, qty: float, tp: float, sl: float) -> dict:
        audit("place_order", f"{side} {qty} {symbol}")
        return self.http.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=str(qty),
            takeProfit=str(tp),
            stopLoss=str(sl),
            timeInForce="IOC",
        )

    def positions(self, symbol: Optional[str] = None) -> list:
        kw = {"category": "linear", "settleCoin": "USDT"}
        if symbol:
            kw["symbol"] = symbol
        try:
            return self.http.get_positions(**kw).get("result", {}).get("list", []) or []
        except Exception as exc:
            log.warning("positions: %s", exc)
            return []

    def close(self, symbol: str) -> Optional[dict]:
        for p in self.positions(symbol):
            size = float(p.get("size") or 0)
            if size <= 0:
                continue
            side = "Sell" if p.get("side") == "Buy" else "Buy"
            return self.http.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(size),
                reduceOnly=True,
                timeInForce="IOC",
            )
        return None


def synthetic_ohlcv(symbol: str, limit: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(sum(ord(c) for c in symbol) % 10000)
    price = {"BTCUSDT": 65000.0, "ETHUSDT": 3400.0, "SOLUSDT": 145.0}.get(symbol, 100.0)
    rows, ts = [], pd.Timestamp.utcnow().floor("min") - pd.Timedelta(minutes=15 * limit)
    for _ in range(limit):
        ret = float(rng.normal(0, 0.0025))
        o, c = price, max(0.01, price * (1 + ret))
        h, l = max(o, c) * 1.001, min(o, c) * 0.999
        rows.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": abs(float(rng.normal(100, 25)))})
        price, ts = c, ts + pd.Timedelta(minutes=15)
    return pd.DataFrame(rows)


class Market:
    def __init__(self, client: Bybit) -> None:
        self.c = client

    def ohlcv(self, symbol: str, limit: int = 300) -> pd.DataFrame:
        cache = CACHE / f"{symbol}_15.csv"
        try:
            raw = self.c.kline(symbol, "15", limit)
            rows = [
                {
                    "timestamp": pd.to_datetime(int(i[0]), unit="ms", utc=True),
                    "open": float(i[1]),
                    "high": float(i[2]),
                    "low": float(i[3]),
                    "close": float(i[4]),
                    "volume": float(i[5]),
                }
                for i in raw
            ]
            df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
            if df.empty:
                raise ValueError("empty")
            df.to_csv(cache, index=False)
            return df
        except Exception as exc:
            log.warning("ohlcv %s failed (%s) — cache/synthetic", symbol, exc)
            if cache.exists():
                return pd.read_csv(cache, parse_dates=["timestamp"])
            df = synthetic_ohlcv(symbol, limit)
            df.to_csv(cache, index=False)
            return df

    def price(self, symbol: str) -> float:
        try:
            p = self.c.ticker(symbol)
            if p > 0:
                return p
        except Exception:
            pass
        return float(self.ohlcv(symbol, 5)["close"].iloc[-1])


# =============================================================================
# INDICATORS + STRATEGIES
# =============================================================================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def add_ind(df: pd.DataFrame) -> pd.DataFrame:
    o = df.copy()
    c = o["close"]
    o["ema_9"], o["ema_21"], o["ema_55"] = ema(c, 9), ema(c, 21), ema(c, 55)
    o["rsi_14"] = rsi(c)
    macd = ema(c, 12) - ema(c, 26)
    o["macd"], o["macd_signal"] = macd, ema(macd, 9)
    o["macd_hist"] = o["macd"] - o["macd_signal"]
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    o["bb_upper"], o["bb_lower"] = mid + 2 * std, mid - 2 * std
    o["bb_width"] = (o["bb_upper"] - o["bb_lower"]) / (mid + 1e-12)
    o["returns"] = c.pct_change()
    o["volatility_20"] = o["returns"].rolling(20).std()
    o["volume_sma_20"] = o["volume"].rolling(20).mean()
    o["volume_ratio"] = o["volume"] / (o["volume_sma_20"] + 1e-12)
    o["sma_50"] = c.rolling(50).mean()
    o["ema9_rel"] = o["ema_9"] / c - 1
    o["ema21_rel"] = o["ema_21"] / c - 1
    o["ema55_rel"] = o["ema_55"] / c - 1
    o["sma50_rel"] = o["sma_50"] / c - 1
    o["bb_pos"] = (c - o["bb_lower"]) / (o["bb_upper"] - o["bb_lower"] + 1e-12)
    tr = pd.concat([(o["high"] - o["low"]), (o["high"] - c.shift()).abs(), (o["low"] - c.shift()).abs()], axis=1).max(axis=1)
    o["atr_14"] = tr.rolling(14).mean()
    o["adx_14"] = o["atr_14"]  # lightweight proxy in single-file build
    low_min, high_max = o["low"].rolling(14).min(), o["high"].rolling(14).max()
    o["stoch_k"] = 100 * (c - low_min) / (high_max - low_min + 1e-12)
    o["stoch_d"] = o["stoch_k"].rolling(3).mean()
    o["log_returns"] = np.log(c / c.shift(1))
    return o


FEATS = [
    "returns", "log_returns", "volatility_20", "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_width", "atr_14", "adx_14", "stoch_k", "stoch_d", "volume_ratio",
    "ema_9", "ema_21", "ema_55", "sma_50", "ema9_rel", "ema21_rel", "ema55_rel", "sma50_rel", "bb_pos",
]


@dataclass
class Signal:
    symbol: str
    direction: str
    confidence: float
    strategy: str

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "direction": self.direction, "confidence": self.confidence, "strategy": self.strategy}


def strat_trend(symbol: str, df: pd.DataFrame) -> Signal:
    d = add_ind(df).iloc[-1]
    if d["ema_9"] > d["ema_21"] > d["ema_55"] and d["macd_hist"] > 0:
        return Signal(symbol, "long", min(0.9, 0.6 + float(d["adx_14"] or 0) / 200), "trend")
    if d["ema_9"] < d["ema_21"] < d["ema_55"] and d["macd_hist"] < 0:
        return Signal(symbol, "short", min(0.9, 0.6 + float(d["adx_14"] or 0) / 200), "trend")
    return Signal(symbol, "flat", 0.0, "trend")


def strat_mr(symbol: str, df: pd.DataFrame) -> Signal:
    d = add_ind(df).iloc[-1]
    if d["close"] <= d["bb_lower"] and d["rsi_14"] <= 30:
        return Signal(symbol, "long", 0.7, "mean_reversion")
    if d["close"] >= d["bb_upper"] and d["rsi_14"] >= 70:
        return Signal(symbol, "short", 0.7, "mean_reversion")
    return Signal(symbol, "flat", 0.0, "mean_reversion")


def strat_breakout(symbol: str, df: pd.DataFrame) -> Signal:
    d = add_ind(df)
    if len(d) < 25:
        return Signal(symbol, "flat", 0.0, "breakout")
    w, row = d.iloc[-21:-1], d.iloc[-1]
    if row["close"] > w["high"].max() and row["volume_ratio"] >= 1.5:
        return Signal(symbol, "long", 0.72, "breakout")
    if row["close"] < w["low"].min() and row["volume_ratio"] >= 1.5:
        return Signal(symbol, "short", 0.72, "breakout")
    return Signal(symbol, "flat", 0.0, "breakout")


def ensemble_signal(symbol: str, df: pd.DataFrame, ml: Optional[Signal] = None) -> Signal:
    votes = {"long": 0.0, "short": 0.0, "flat": 0.0}
    for s, w in ((strat_trend(symbol, df), 0.35), (strat_mr(symbol, df), 0.25), (strat_breakout(symbol, df), 0.25)):
        votes[s.direction] += w * s.confidence
    if ml:
        votes[ml.direction] += 0.4 * ml.confidence
    direction = max(votes, key=votes.get)
    conf = votes[direction]
    if direction == "flat" or conf < 0.35:
        return Signal(symbol, "flat", 0.0, "ensemble")
    return Signal(symbol, direction, float(min(conf, 0.99)), "ensemble")


# =============================================================================
# EXPERIENCE MEMORY + ML (continual learning)
# =============================================================================
class ExperienceMemory:
    def __init__(self) -> None:
        self.market_path = EXP / "market_memory.csv"
        self.mistakes_path = EXP / "mistakes.jsonl"

    def _load(self) -> pd.DataFrame:
        if self.market_path.exists():
            try:
                return pd.read_csv(self.market_path)
            except Exception:
                pass
        return pd.DataFrame()

    def _save(self, df: pd.DataFrame) -> None:
        if len(df) > CFG["exp_max"]:
            if "source" in df.columns:
                m = df[df["source"] == "trade_lesson"]
                o = df[df["source"] != "trade_lesson"].tail(max(0, CFG["exp_max"] - len(m)))
                df = pd.concat([o, m], ignore_index=True)
            else:
                df = df.tail(CFG["exp_max"])
        df.to_csv(self.market_path, index=False)

    def make_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        data = add_ind(df)
        fut = data["close"].shift(-1) / data["close"] - 1
        y = np.where(fut > 0.0015, 1, np.where(fut < -0.0015, -1, 0))
        cols = [c for c in FEATS if c in data.columns]
        frame = data[cols].copy()
        frame["y_class"] = y
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        return frame[cols], frame["y_class"].astype(int)

    def ingest_market(self, df: pd.DataFrame, symbol: str) -> int:
        X, y = self.make_xy(df)
        if X.empty:
            return 0
        chunk = X.copy()
        chunk["y_class"] = y.to_numpy()
        chunk["source"] = "market"
        chunk["symbol"] = symbol
        old = self._load()
        merged = pd.concat([old, chunk], ignore_index=True) if not old.empty else chunk
        merged = merged.drop_duplicates(subset=[c for c in FEATS if c in merged.columns] + ["y_class"], keep="last")
        self._save(merged)
        return len(chunk)

    def ingest_trades(self) -> int:
        with db() as s:
            trades = list(s.execute(select(Trade).where(Trade.status == "closed")).scalars())
            rows = [
                {
                    "id": t.id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "pnl": float(t.pnl or 0),
                    "pnl_pct": float(t.pnl_pct or 0),
                }
                for t in trades
            ]
        seen = set()
        if self.mistakes_path.exists():
            for line in self.mistakes_path.read_text(encoding="utf-8").splitlines():
                try:
                    seen.add(json.loads(line).get("id"))
                except Exception:
                    pass
        added, lessons = 0, []
        with self.mistakes_path.open("a", encoding="utf-8") as fh:
            for t in rows:
                if t["id"] in seen:
                    continue
                side_dir = 1 if (t["side"] or "").lower() in {"buy", "long"} else -1
                if t["pnl"] < 0:
                    y, dup, kind = -side_dir, 3, "loss"
                elif t["pnl"] > 0:
                    y, dup, kind = side_dir, 1, "win"
                else:
                    y, dup, kind = 0, 1, "flat"
                fh.write(json.dumps({**t, "kind": kind, "y_class": y}) + "\n")
                added += 1
                row = {c: 0.0 for c in FEATS}
                row["returns"] = t["pnl_pct"] / 100.0
                row["rsi_14"] = 30.0 if y == 1 else (70.0 if y == -1 else 50.0)
                row["y_class"] = y
                row["source"] = "trade_lesson"
                row["symbol"] = t["symbol"]
                lessons.extend([row] * dup)
        if lessons:
            old = self._load()
            merged = pd.concat([old, pd.DataFrame(lessons)], ignore_index=True) if not old.empty else pd.DataFrame(lessons)
            self._save(merged)
        return added

    def training_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        df = self._load()
        if df.empty:
            return pd.DataFrame(columns=FEATS), pd.Series(dtype=int)
        cols = [c for c in FEATS if c in df.columns]
        X = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
        y = df.loc[X.index, "y_class"].astype(int)
        for c in FEATS:
            if c not in X.columns:
                X[c] = 0.0
        return X[FEATS], y

    def stats(self) -> dict:
        df = self._load()
        n = sum(1 for _ in self.mistakes_path.open()) if self.mistakes_path.exists() else 0
        return {"market_rows": len(df), "trade_lessons_logged": n, "never_deleted": True}


class Brain:
    """XGBoost + LSTM with warm-start continual learning."""

    def __init__(self) -> None:
        import torch
        import torch.nn as nn
        from xgboost import XGBClassifier

        self.torch = torch
        self.nn = nn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = 32
        self.xgb = XGBClassifier(
            n_estimators=120, max_depth=5, learning_rate=0.05, objective="multi:softprob",
            num_class=3, n_jobs=-1, tree_method="hist",
        )
        self.xgb_trained = False
        self.lstm = None
        self.lstm_trained = False
        self.mean = None
        self.std = None
        self.metrics: Dict[str, Any] = {}
        self._build_lstm(len(FEATS))

    def _build_lstm(self, n: int) -> None:
        nn, torch = self.nn, self.torch

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(n, 64, 2, batch_first=True, dropout=0.2)
                self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))

            def forward(self, x):
                o, _ = self.lstm(x)
                return self.head(o[:, -1, :])

        self.lstm = Net().to(self.device)
        self.n_features = n

    def load(self) -> bool:
        import joblib

        ok = False
        # Dedicated filenames so this single-file bot never clashes with multi-module checkpoints
        xp = MODELS / "sf_xgb.joblib"
        if xp.exists():
            try:
                blob = joblib.load(xp)
                self.xgb = blob["model"]
                self.xgb_trained = True
                self.metrics["xgboost"] = blob.get("metrics", {})
                ok = True
            except Exception as exc:
                log.warning("XGB load skip: %s", exc)
        lp = MODELS / "sf_lstm.pt"
        if lp.exists():
            try:
                blob = self.torch.load(lp, map_location=self.device, weights_only=False)
                self._build_lstm(blob.get("n_features", len(FEATS)))
                self.lstm.load_state_dict(blob["state_dict"])
                self.lstm_trained = True
                self.metrics["lstm"] = blob.get("metrics", {})
                ok = True
            except Exception as exc:
                log.warning("LSTM checkpoint skip (%s) — will train new", exc)
                self.lstm_trained = False
        meta = MODELS / "sf_meta.json"
        if meta.exists():
            m = json.loads(meta.read_text(encoding="utf-8"))
            self.mean = np.array(m.get("mean", []), dtype=np.float32)
            self.std = np.array(m.get("std", []), dtype=np.float32)
        return ok

    def save(self) -> None:
        import joblib

        joblib.dump({"model": self.xgb, "metrics": self.metrics.get("xgboost", {})}, MODELS / "sf_xgb.joblib")
        self.torch.save(
            {"state_dict": self.lstm.state_dict(), "n_features": self.n_features, "metrics": self.metrics.get("lstm", {})},
            MODELS / "sf_lstm.pt",
        )
        (MODELS / "sf_meta.json").write_text(
            json.dumps({
                "mean": self.mean.tolist() if self.mean is not None else [],
                "std": self.std.tolist() if self.std is not None else [],
                "features": FEATS,
            }),
            encoding="utf-8",
        )

    def train(self, X: pd.DataFrame, y: pd.Series, epochs: int = 8) -> dict:
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, TensorDataset

        if len(X) < 80:
            return {"ok": False}
        y_m = (y.to_numpy() + 1).astype(int)
        Xtr, Xva, ytr, yva = train_test_split(X, y_m, test_size=0.2, shuffle=False)
        fit_kw = {}
        if self.xgb_trained:
            try:
                fit_kw["xgb_model"] = self.xgb.get_booster()
                log.info("XGB warm-start")
            except Exception:
                fit_kw = {}
        self.xgb.fit(Xtr, ytr, **fit_kw)
        acc_x = float(accuracy_score(yva, self.xgb.predict(Xva)))
        self.xgb_trained = True
        self.metrics["xgboost"] = {"accuracy": acc_x, "warm_start": float(bool(fit_kw)), "samples": float(len(X))}

        Xn = X.to_numpy(dtype=np.float32)
        mean, std = Xn.mean(0), Xn.std(0) + 1e-8
        if self.mean is not None and len(self.mean) == len(mean):
            mean = 0.8 * self.mean + 0.2 * mean
            std = 0.8 * self.std + 0.2 * std
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
        Xnorm = (Xn - self.mean) / self.std
        xs, ys = [], []
        for i in range(self.seq_len, len(Xnorm)):
            xs.append(Xnorm[i - self.seq_len : i])
            ys.append(y_m[i])
        if len(xs) < 50:
            self.save()
            return {"xgboost": self.metrics["xgboost"], "lstm": {"accuracy": 0.0}}
        X_seq = np.asarray(xs, dtype=np.float32)
        y_seq = np.asarray(ys)
        warm = self.lstm_trained and self.n_features == X_seq.shape[-1]
        lr = 3e-4 if warm else 1e-3
        if not warm:
            self._build_lstm(X_seq.shape[-1])
        else:
            log.info("LSTM warm-start")
        split = int(len(X_seq) * 0.85)
        ds = TensorDataset(self.torch.tensor(X_seq[:split]), self.torch.tensor(y_seq[:split], dtype=self.torch.long))
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        opt = self.torch.optim.Adam(self.lstm.parameters(), lr=lr)
        crit = self.nn.CrossEntropyLoss()
        self.lstm.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = crit(self.lstm(xb), yb)
                loss.backward()
                opt.step()
        self.lstm.eval()
        with self.torch.no_grad():
            pred = self.lstm(self.torch.tensor(X_seq[split:], dtype=self.torch.float32).to(self.device)).argmax(1).cpu().numpy()
        acc_l = float((pred == y_seq[split:]).mean()) if len(pred) else 0.0
        self.lstm_trained = True
        self.metrics["lstm"] = {"accuracy": acc_l, "warm_start": float(warm), "samples": float(len(X_seq))}
        self.save()
        # prune old file snapshots only
        self._prune_backups()
        return {"xgboost": self.metrics["xgboost"], "lstm": self.metrics["lstm"]}

    def _prune_backups(self) -> None:
        BACKUPS.mkdir(exist_ok=True)
        # copy current as snapshot
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dest = BACKUPS / f"models_{stamp}"
        dest.mkdir(exist_ok=True)
        for f in MODELS.iterdir():
            if f.is_file():
                (dest / f.name).write_bytes(f.read_bytes())
        dirs = sorted([p for p in BACKUPS.iterdir() if p.is_dir() and p.name.startswith("models_")], reverse=True)
        for old in dirs[CFG["model_keep"] :]:
            import shutil

            shutil.rmtree(old, ignore_errors=True)

    def predict(self, symbol: str, df: pd.DataFrame) -> Signal:
        data = add_ind(df)
        cols = [c for c in FEATS if c in data.columns]
        row = data[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if row.empty:
            return Signal(symbol, "flat", 0.0, "ml")
        scores = {"long": 0.0, "short": 0.0, "flat": 0.0}
        try:
            if self.xgb_trained:
                names = list(getattr(self.xgb, "feature_names_in_", cols))
                use = [c for c in names if c in row.columns]
                if len(use) == len(names):
                    proba = self.xgb.predict_proba(row.iloc[[-1]][use])[0]
                    idx = int(np.argmax(proba))
                    d = {0: "short", 1: "flat", 2: "long"}[idx]
                    scores[d] += 0.55 * float(proba[idx])
            if self.lstm_trained and self.mean is not None:
                Xn = row[cols].to_numpy(dtype=np.float32)
                n = min(Xn.shape[1], len(self.mean))
                Xnorm = (Xn[:, :n] - self.mean[:n]) / self.std[:n]
                if len(Xnorm) >= self.seq_len:
                    import torch

                    self.lstm.eval()
                    with torch.no_grad():
                        x = torch.tensor(Xnorm[-self.seq_len :][None, ...], dtype=torch.float32).to(self.device)
                        proba = torch.softmax(self.lstm(x), 1).cpu().numpy()[0]
                    idx = int(np.argmax(proba))
                    d = {0: "short", 1: "flat", 2: "long"}[idx]
                    scores[d] += 0.45 * float(proba[idx])
        except Exception as exc:
            log.warning("ml predict fallback: %s", exc)
            return Signal(symbol, "flat", 0.0, "ml")
        direction = max(scores, key=scores.get)
        conf = scores[direction]
        if direction == "flat":
            conf = 0.0
        return Signal(symbol, direction, float(min(conf, 0.99)), "ml")


# =============================================================================
# RISK / POSITIONS / EXECUTOR
# =============================================================================
@dataclass
class RiskState:
    day: str
    start_balance: float
    realized_pnl: float = 0.0
    goals_hit: Set[float] = field(default_factory=set)
    loss_limit_hit: bool = False
    halted: bool = False


class Risk:
    def __init__(self) -> None:
        import pytz

        self.tz = pytz.timezone(CFG["tz"])
        self.state = RiskState(day=datetime.now(self.tz).strftime("%Y-%m-%d"), start_balance=CFG["initial_balance"])
        self._load()

    def _load(self) -> None:
        with db() as s:
            row = s.execute(select(DailyStats).where(DailyStats.day == self.state.day)).scalar_one_or_none()
            if row is None:
                s.add(DailyStats(day=self.state.day, start_balance=self.state.start_balance, end_balance=self.state.start_balance))
            else:
                self.state.realized_pnl = row.pnl or 0.0
                self.state.loss_limit_hit = bool(row.loss_limit_hit)
                self.state.halted = self.state.loss_limit_hit
                if row.goals_hit:
                    self.state.goals_hit = {float(x) for x in row.goals_hit.split(",") if x}

    def pnl_pct(self) -> float:
        return (self.state.realized_pnl / self.state.start_balance * 100) if self.state.start_balance else 0.0

    def can_open(self, n_open: int) -> Tuple[bool, str]:
        if self.state.halted or self.state.loss_limit_hit:
            return False, "daily_loss_limit"
        if n_open >= CFG["max_open"]:
            return False, "max_open"
        if self.pnl_pct() <= -abs(CFG["max_daily_loss_pct"]):
            self.trigger_loss()
            return False, "daily_loss_limit"
        return True, "ok"

    def size_usd(self, bal: float, conf: float) -> float:
        return round(bal * (CFG["max_position_pct"] / 100) * max(0.4, min(1.0, conf)), 4)

    def qty(self, usd: float, price: float) -> float:
        q = (usd * CFG["leverage"]) / max(price, 1e-9)
        return round(q, 3 if price >= 1000 else 2)

    def levels(self, side: str, entry: float) -> dict:
        sl, tp = CFG["sl_pct"] / 100, CFG["tp_pct"] / 100
        if side == "Buy":
            return {"stop_loss": round(entry * (1 - sl), 6), "take_profit": round(entry * (1 + tp), 6)}
        return {"stop_loss": round(entry * (1 + sl), 6), "take_profit": round(entry * (1 - tp), 6)}

    def register(self, pnl: float, bal: float) -> List[float]:
        self.state.realized_pnl += pnl
        pct = self.pnl_pct()
        newly = []
        for g in CFG["goals"]:
            if pct >= g and g not in self.state.goals_hit:
                self.state.goals_hit.add(g)
                newly.append(g)
        if pct <= -abs(CFG["max_daily_loss_pct"]):
            self.trigger_loss()
        self._persist(bal)
        return newly

    def trigger_loss(self) -> None:
        self.state.loss_limit_hit = True
        self.state.halted = True
        self._persist(self.state.start_balance + self.state.realized_pnl)

    def _persist(self, end: float) -> None:
        with db() as s:
            row = s.execute(select(DailyStats).where(DailyStats.day == self.state.day)).scalar_one_or_none()
            goals = ",".join(str(g) for g in sorted(self.state.goals_hit))
            if row is None:
                s.add(DailyStats(day=self.state.day, start_balance=self.state.start_balance, end_balance=end, pnl=self.state.realized_pnl, pnl_pct=self.pnl_pct(), goals_hit=goals, loss_limit_hit=self.state.loss_limit_hit))
            else:
                row.end_balance, row.pnl, row.pnl_pct = end, self.state.realized_pnl, self.pnl_pct()
                row.goals_hit, row.loss_limit_hit = goals, self.state.loss_limit_hit

    def snap(self) -> dict:
        return {"day": self.state.day, "pnl": self.state.realized_pnl, "pnl_pct": self.pnl_pct(), "goals_hit": sorted(self.state.goals_hit), "loss_limit_hit": self.state.loss_limit_hit}


class Positions:
    def open(self, **kw) -> int:
        with db() as s:
            t = Trade(**kw, status="open", opened_at=utcnow())
            s.add(t)
            s.flush()
            return int(t.id)

    def close(self, trade_id: int, exit_price: float, reason: str) -> dict:
        with db() as s:
            t = s.get(Trade, trade_id)
            direction = 1.0 if t.side.lower() in {"buy", "long"} else -1.0
            pnl = (exit_price - t.entry_price) * t.qty * direction
            margin = (t.entry_price * t.qty) / max(t.leverage or 1, 1)
            t.exit_price, t.pnl, t.pnl_pct = exit_price, pnl, (pnl / margin * 100 if margin else 0)
            t.exit_reason, t.status, t.closed_at = reason, "closed", utcnow()
            return {"id": t.id, "symbol": t.symbol, "side": t.side, "pnl": t.pnl, "pnl_pct": t.pnl_pct, "exit_reason": reason, "strategy": t.strategy}

    def open_list(self) -> List[dict]:
        with db() as s:
            rows = list(s.execute(select(Trade).where(Trade.status == "open")).scalars())
            return [{"id": r.id, "symbol": r.symbol, "side": r.side, "qty": r.qty, "entry_price": r.entry_price, "strategy": r.strategy} for r in rows]

    def closed_list(self, limit: int = 50) -> List[dict]:
        with db() as s:
            rows = list(s.execute(select(Trade).where(Trade.status == "closed").order_by(desc(Trade.closed_at)).limit(limit)).scalars())
            return [{"id": r.id, "symbol": r.symbol, "side": r.side, "pnl": r.pnl, "pnl_pct": r.pnl_pct, "exit_reason": r.exit_reason} for r in rows]


class Executor:
    def __init__(self, client: Bybit, market: Market, risk: Risk, positions: Positions, tg: Telegram) -> None:
        self.c, self.m, self.r, self.p, self.tg = client, market, risk, positions, tg

    def execute(self, sig: Signal) -> dict:
        if sig.direction == "flat" or sig.confidence < CFG["min_confidence"]:
            return {"executed": False, "reason": "low_confidence_or_flat"}
        ok, why = self.r.can_open(len(self.p.open_list()))
        if not ok:
            return {"executed": False, "reason": why}
        if any(x["symbol"] == sig.symbol for x in self.p.open_list()):
            return {"executed": False, "reason": "already_open"}
        price = self.m.price(sig.symbol)
        bal = self.c.balance()
        qty = self.r.qty(self.r.size_usd(bal, sig.confidence), price)
        if qty <= 0:
            return {"executed": False, "reason": "qty_zero"}
        side = "Buy" if sig.direction == "long" else "Sell"
        lv = self.r.levels(side, price)
        try:
            self.c.set_leverage(sig.symbol, CFG["leverage"])
            order = self.c.place(sig.symbol, side, qty, lv["take_profit"], lv["stop_loss"])
            oid = (order.get("result") or {}).get("orderId", "")
            tid = self.p.open(symbol=sig.symbol, side=side, qty=qty, entry_price=price, leverage=CFG["leverage"], strategy=sig.strategy, confidence=sig.confidence, order_id=oid)
            return {"executed": True, "trade": {"id": tid, "symbol": sig.symbol, "side": side, "qty": qty, "entry_price": price}}
        except Exception as exc:
            log.exception("execute")
            return {"executed": False, "reason": str(exc)}

    def sync(self) -> list:
        events = []
        for pos in self.p.open_list():
            remote = self.c.positions(pos["symbol"])
            size = sum(float(r.get("size") or 0) for r in remote)
            if size > 0:
                continue
            price = self.m.price(pos["symbol"])
            result = self.p.close(pos["id"], price, "exchange_flat")
            goals = self.r.register(result["pnl"], self.c.balance())
            self.tg.trade_closed(result)
            for g in goals:
                self.tg.goal(g, self.r.pnl_pct())
            if self.r.state.loss_limit_hit:
                self.tg.loss_limit(self.r.pnl_pct())
            events.append(result)
        return events


# =============================================================================
# ANOMALY
# =============================================================================
def detect_anomaly(symbol: str, df: pd.DataFrame) -> List[dict]:
    if len(df) < 40:
        return []
    rets = df["close"].pct_change()
    vol = rets.rolling(20).std()
    z = float(((vol.iloc[-1] - vol.mean()) / (vol.std() + 1e-12)))
    out = []
    if abs(z) >= 3.5:
        out.append({"symbol": symbol, "type": "volatility_spike", "details": f"z={z:.2f}"})
    return out


# =============================================================================
# ENGINE
# =============================================================================
class Engine:
    def __init__(self) -> None:
        self.client = Bybit()
        self.market = Market(self.client)
        self.risk = Risk()
        self.positions = Positions()
        self.tg = Telegram()
        self.exec = Executor(self.client, self.market, self.risk, self.positions, self.tg)
        self.memory = ExperienceMemory()
        self.brain = Brain()
        self.brain.load()
        self.signals: Dict[str, dict] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def status(self) -> dict:
        return {
            "running": self._running,
            "testnet": CFG["testnet"],
            "pairs": CFG["pairs"],
            "ip": self.client.ip,
            "risk": self.risk.snap(),
            "open": self.positions.open_list(),
            "signals": self.signals,
            "ml": self.brain.metrics,
            "experience": self.memory.stats(),
            "brain_version": get_state("brain_version"),
            "persistence": {
                "last_scan_at": get_state("last_scan_at"),
                "last_retrain_at": get_state("last_retrain_at"),
                "last_started_at": get_state("last_started_at"),
            },
        }

    def scan_once(self) -> dict:
        results = []
        for symbol in CFG["pairs"]:
            try:
                df = self.market.ohlcv(symbol, 250)
                for a in detect_anomaly(symbol, df):
                    self.tg.anomaly(a)
                ml = self.brain.predict(symbol, df)
                sig = ensemble_signal(symbol, df, ml)
                self.signals[symbol] = sig.to_dict()
                ex = self.exec.execute(sig)
                results.append({"symbol": symbol, "signal": sig.to_dict(), "ml": ml.to_dict(), "execution": ex})
            except Exception as exc:
                log.exception("scan %s", symbol)
                results.append({"symbol": symbol, "error": str(exc)})
        closed = self.exec.sync()
        set_state("last_scan_at", utcnow().isoformat())
        return {"results": results, "closed": closed, "risk": self.risk.snap()}

    def train(self, epochs: int = 8) -> dict:
        for sym in CFG["pairs"]:
            df = self.market.ohlcv(sym, 500)
            self.memory.ingest_market(df, sym)
        lessons = self.memory.ingest_trades()
        X, y = self.memory.training_xy()
        if X.empty:
            df = self.market.ohlcv(CFG["pairs"][0], 500)
            X, y = self.memory.make_xy(df)
        self.brain.load()
        metrics = self.brain.train(X, y, epochs=epochs)
        ver = str(int(time.time()))
        set_state("brain_version", ver)
        set_state("last_retrain_at", utcnow().isoformat())
        report = {"ok": True, "metrics": metrics, "version": ver, "experience": self.memory.stats(), "new_lessons": lessons}
        self.tg.retrain(report)
        return report

    def start(self) -> None:
        self._running = True
        info = {
            "brain_version": get_state("brain_version"),
            "last_scan_at": get_state("last_scan_at"),
            "last_retrain_at": get_state("last_retrain_at"),
        }
        set_state("last_started_at", utcnow().isoformat())
        self.tg.resumed(info)
        self.exec.sync()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        audit("engine_start", str(CFG["pairs"]))
        # simple daily retrain scheduler thread
        threading.Thread(target=self._nightly, daemon=True).start()
        log.info("Engine started (continual learning ON)")

    def _loop(self) -> None:
        while self._running:
            try:
                self.scan_once()
            except Exception:
                log.exception("loop")
            time.sleep(60)

    def _nightly(self) -> None:
        import pytz

        tz = pytz.timezone(CFG["tz"])
        while self._running:
            now = datetime.now(tz)
            if now.hour == 3 and now.minute < 2:
                try:
                    self.train(epochs=10)
                except Exception:
                    log.exception("nightly")
                time.sleep(120)
            time.sleep(30)

    def stop(self) -> None:
        self._running = False
        set_state("last_shutdown_at", utcnow().isoformat())


# =============================================================================
# DASHBOARD (minimal HTTPS)
# =============================================================================
DASH_HTML = """<!doctype html><html lang="uk"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Tryzub Trade</title>
<style>
body{margin:0;font-family:system-ui,sans-serif;background:#0e1412;color:#e7f0ea;padding:20px}
h1{margin:0 0 8px} .muted{color:#8fa399} .panel{border:1px solid #2a3d34;padding:14px;margin:12px 0;background:#15201c}
.metric{font-size:28px;font-weight:700;color:#3dcea7} button{background:#259f7d;color:#fff;border:0;padding:10px 14px;margin:4px;cursor:pointer}
input{padding:10px;margin:6px 0;width:100%;max-width:320px;background:#0e1412;border:1px solid #2a3d34;color:#fff}
table{width:100%;border-collapse:collapse;font-size:13px} td,th{border-bottom:1px solid #2a3d34;padding:6px;text-align:left}
</style></head><body>
<div id="login"><h1>Tryzub Trade</h1><p class="muted">Single-file bot dashboard</p>
<input id="u" value="admin"/><input id="p" type="password" value="admin"/>
<br/><button onclick="login()">Увійти</button><pre id="err"></pre></div>
<div id="app" style="display:none"><h1>Tryzub Trade</h1>
<div class="panel"><div class="muted">P&L дня</div><div class="metric" id="pnl">—</div></div>
<div class="panel"><div class="muted">Статус</div><pre id="st"></pre>
<button onclick="scan()">Скан</button><button onclick="refresh()">Оновити</button></div>
<div class="panel"><div class="muted">Відкриті</div><table><tbody id="op"></tbody></table></div>
<div class="panel"><div class="muted">Закриті</div><table><tbody id="cl"></tbody></table></div>
</div>
<script>
let token='';
async function api(path, opt={}){
  const h={'Content-Type':'application/json',...(opt.headers||{})};
  if(token) h.Authorization='Bearer '+token;
  const r=await fetch(path,{...opt,headers:h});
  if(!r.ok) throw new Error((await r.json().catch(()=>({}))).detail||r.status);
  return r.json();
}
async function login(){
  try{
    const d=await api('/api/login',{method:'POST',body:JSON.stringify({username:u.value,password:p.value})});
    token=d.token; login.style.display='none'; app.style.display='block'; refresh();
  }catch(e){err.textContent=e.message}
}
async function refresh(){
  const s=await api('/api/status'); const t=await api('/api/trades');
  pnl.textContent=(s.risk.pnl_pct||0).toFixed(2)+'%';
  st.textContent=JSON.stringify({testnet:s.testnet,pairs:s.pairs,experience:s.experience,ml:s.ml,persistence:s.persistence},null,2);
  op.innerHTML=(t.open||[]).map(x=>`<tr><td>${x.symbol}</td><td>${x.side}</td><td>${x.entry_price}</td></tr>`).join('');
  cl.innerHTML=(t.closed||[]).map(x=>`<tr><td>${x.symbol}</td><td>${x.pnl}</td><td>${x.exit_reason||''}</td></tr>`).join('');
}
async function scan(){ await api('/api/scan',{method:'POST'}); refresh(); }
setInterval(()=>{ if(token) refresh().catch(()=>{}); }, 15000);
</script></body></html>"""


def create_app(engine: Engine):
    from fastapi import Depends, FastAPI, HTTPException, Request, Response
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
    app = FastAPI(title="Tryzub Trade Single-File")
    app.state.engine = engine
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, lambda r, e: Response("rate limit", 429))
    app.add_middleware(SlowAPIMiddleware)

    class Login(BaseModel):
        username: str
        password: str

    def user(request: Request):
        auth = request.headers.get("Authorization", "")
        token = auth.replace("Bearer ", "") if auth.startswith("Bearer ") else ""
        if not token:
            raise HTTPException(401, "Not authenticated")
        try:
            return decode_token(token)["sub"]
        except Exception as exc:
            raise HTTPException(401, "bad token") from exc

    @app.get("/", response_class=HTMLResponse)
    def index():
        return DASH_HTML

    @app.post("/api/login")
    @limiter.limit("10/minute")
    def login(request: Request, response: Response, body: Login):
        if body.username != CFG["dash_user"] or not verify_password(body.password, CFG["dash_pass_hash"]):
            raise HTTPException(401, "Invalid credentials")
        return {"token": make_token(body.username)}

    @app.get("/api/status")
    @limiter.limit("60/minute")
    def status(request: Request, response: Response, u: str = Depends(user)):
        return engine.status()

    @app.get("/api/trades")
    @limiter.limit("60/minute")
    def trades(request: Request, response: Response, u: str = Depends(user)):
        return {"open": engine.positions.open_list(), "closed": engine.positions.closed_list()}

    @app.post("/api/scan")
    @limiter.limit("10/minute")
    def scan(request: Request, response: Response, u: str = Depends(user)):
        return engine.scan_once()

    @app.get("/api/ping")
    def ping():
        return {"ok": True, "single_file": True}

    return app


# =============================================================================
# MAIN
# =============================================================================
def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Tryzub Trade single-file bot")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--no-dashboard", action="store_true")
    args = parser.parse_args(argv)

    ensure_env_template()
    setup_logging()
    init_db()

    if not CFG["bybit_key"]:
        log.warning("BYBIT_API_KEY empty — public/synthetic mode; fill .env for real orders")

    engine = Engine()

    if args.train:
        print(json.dumps(engine.train(), indent=2, default=str))
        return 0
    if args.once:
        print(json.dumps(engine.scan_once(), indent=2, default=str))
        return 0

    engine.start()

    def stop(*_):
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    if args.no_dashboard:
        while True:
            time.sleep(3600)

    import uvicorn

    cert, key = gen_certs()
    log.info("Dashboard https://%s:%s", CFG["host"], CFG["port"])
    app = create_app(engine)
    uvicorn.run(app, host=CFG["host"], port=CFG["port"], ssl_certfile=str(cert), ssl_keyfile=str(key), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
