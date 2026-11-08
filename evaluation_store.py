"""SQLite persistence for live model evaluation runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.engine import Engine, URL, make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


class EvaluationRecord(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )


def default_database_url() -> URL:
    database_path = Path.home() / ".evalsphere" / "evalsphere.db"
    return URL.create("sqlite", database=str(database_path))


def create_database_engine(database_url: str | URL) -> Engine:
    url = make_url(database_url) if isinstance(database_url, str) else database_url
    backend = url.get_backend_name()
    if backend == "sqlite":
        options: dict[str, Any] = {"connect_args": {"check_same_thread": False}}
        if url.database in (None, "", ":memory:"):
            options["poolclass"] = StaticPool
        return create_engine(url, **options)
    return create_engine(url)


def initialize_database(engine: Engine) -> None:
    if engine.dialect.name == "sqlite":
        database_path = engine.url.database
        if database_path not in (None, "", ":memory:"):
            Path(database_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def save_evaluation(
    session: Session,
    *,
    model: str,
    prompt: str,
    response: str,
    latency_ms: float,
) -> EvaluationRecord:
    record = EvaluationRecord(
        model=model,
        prompt=prompt,
        response=response,
        latency_ms=latency_ms,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def recent_evaluations(session: Session, *, limit: int) -> list[EvaluationRecord]:
    statement = (
        select(EvaluationRecord)
        .order_by(EvaluationRecord.created_at.desc(), EvaluationRecord.id.desc())
        .limit(limit)
    )
    return list(session.scalars(statement))


def evaluation_to_dict(record: EvaluationRecord) -> dict[str, Any]:
    created_at = record.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return {
        "id": record.id,
        "model": record.model,
        "prompt": record.prompt,
        "response": record.response,
        "latency_ms": record.latency_ms,
        "created_at": created_at.isoformat(),
    }
