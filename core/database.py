"""SQLAlchemy models and DB helpers for trades, signals, models, audits."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator, Iterable, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
    desc,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import DB_PATH


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False, index=True)
    side = Column(String(8), nullable=False)  # Buy / Sell
    qty = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    leverage = Column(Integer, default=1)
    pnl = Column(Float, default=0.0)
    pnl_pct = Column(Float, default=0.0)
    fee = Column(Float, default=0.0)
    status = Column(String(16), default="open")  # open / closed / cancelled
    strategy = Column(String(64), default="")
    exit_reason = Column(String(128), default="")
    confidence = Column(Float, default=0.0)
    order_id = Column(String(64), default="")
    opened_at = Column(DateTime(timezone=True), default=utcnow)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, default="")


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False, index=True)
    direction = Column(String(8), nullable=False)  # long / short / flat
    confidence = Column(Float, default=0.0)
    strategy = Column(String(64), default="")
    features_json = Column(Text, default="{}")
    created_at = Column(DateTime(timezone=True), default=utcnow)
    executed = Column(Boolean, default=False)


class DailyStats(Base):
    __tablename__ = "daily_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    day = Column(String(10), unique=True, nullable=False)  # YYYY-MM-DD
    start_balance = Column(Float, default=0.0)
    end_balance = Column(Float, default=0.0)
    pnl = Column(Float, default=0.0)
    pnl_pct = Column(Float, default=0.0)
    trades = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    goals_hit = Column(String(64), default="")  # e.g. "1,3"
    loss_limit_hit = Column(Boolean, default=False)
    notes = Column(Text, default="")


class ModelCheckpoint(Base):
    __tablename__ = "model_checkpoints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(64), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    path = Column(String(512), nullable=False)
    metrics_json = Column(Text, default="{}")
    created_at = Column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(256), nullable=False)
    actor = Column(String(64), default="system")
    details = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=utcnow, index=True)


class MarketAnomaly(Base):
    __tablename__ = "market_anomalies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False)
    anomaly_type = Column(String(64), nullable=False)
    severity = Column(String(16), default="medium")
    details = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=utcnow)


class BotState(Base):
    __tablename__ = "bot_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(64), unique=True, nullable=False)
    value = Column(Text, default="")
    updated_at = Column(DateTime(timezone=True), default=utcnow)


_engine = None
_SessionLocal = None


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{DB_PATH}",
            connect_args={"check_same_thread": False},
            future=True,
        )
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    get_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def audit(action: str, actor: str = "system", details: str = "") -> None:
    with session_scope() as s:
        s.add(AuditLog(action=action, actor=actor, details=details))


def set_state(key: str, value: str) -> None:
    with session_scope() as s:
        row = s.execute(select(BotState).where(BotState.key == key)).scalar_one_or_none()
        if row is None:
            s.add(BotState(key=key, value=value, updated_at=utcnow()))
        else:
            row.value = value
            row.updated_at = utcnow()


def get_state(key: str, default: str = "") -> str:
    with session_scope() as s:
        row = s.execute(select(BotState).where(BotState.key == key)).scalar_one_or_none()
        return row.value if row else default


def list_trades(limit: int = 100, status: Optional[str] = None) -> List[Trade]:
    with session_scope() as s:
        q = select(Trade).order_by(desc(Trade.opened_at)).limit(limit)
        if status:
            q = select(Trade).where(Trade.status == status).order_by(desc(Trade.opened_at)).limit(limit)
        rows = list(s.execute(q).scalars().all())
        # Detach
        s.expunge_all()
        return rows


def recent_audits(limit: int = 100) -> List[AuditLog]:
    with session_scope() as s:
        rows = list(
            s.execute(select(AuditLog).order_by(desc(AuditLog.created_at)).limit(limit)).scalars()
        )
        s.expunge_all()
        return rows
