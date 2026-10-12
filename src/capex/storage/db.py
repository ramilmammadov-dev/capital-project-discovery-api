"""Relational state: jobs, the per-request usage ledger, the document fetch index and the
stable project-ID registry.

SQLite by default (zero setup); any SQLAlchemy URL works (``CAPEX_DATABASE_URL=postgresql+psycopg://…``).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    event,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from capex.models import UsageRecord


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    request_key: Mapped[str] = mapped_column(String(64), index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(200), unique=True, nullable=True)
    organization_name: Mapped[str] = mapped_column(String(300))
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), index=True, default="queued")
    stage: Mapped[str] = mapped_column(String(40), default="queued")
    progress: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    warnings: Mapped[list[str]] = mapped_column(JSON, default=list)
    errors: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    result_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class UsageRow(Base):
    __tablename__ = "usage_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str | None] = mapped_column(String(40), index=True, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    stage: Mapped[str] = mapped_column(String(40))
    task: Mapped[str] = mapped_column(String(60))
    provider: Mapped[str] = mapped_column(String(40))
    model: Mapped[str] = mapped_column(String(80))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0)
    search_requests: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    retries: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    estimated: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class DocumentRow(Base):
    """Fetch index: canonical URL -> last downloaded content (for conditional re-fetch)."""

    __tablename__ = "documents"

    url: Mapped[str] = mapped_column(String(1000), primary_key=True)
    final_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    etag: Mapped[str | None] = mapped_column(String(300), nullable=True)
    last_modified: Mapped[str | None] = mapped_column(String(100), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ProjectRegistryRow(Base):
    """Alias key -> stable project UID, so repeat runs never mint duplicate project records."""

    __tablename__ = "project_registry"

    org_key: Mapped[str] = mapped_column(String(200), primary_key=True)
    alias_key: Mapped[str] = mapped_column(String(400), primary_key=True)
    project_uid: Mapped[str] = mapped_column(String(40), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Database:
    def __init__(self, url: str) -> None:
        if url.startswith("sqlite:///"):
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        connect_args = {"check_same_thread": False, "timeout": 30} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, connect_args=connect_args, future=True)
        if url.startswith("sqlite"):

            @event.listens_for(self.engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - trivial
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.close()

        Base.metadata.create_all(self.engine)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._sessions()
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    # -- jobs -----------------------------------------------------------------------------------

    def create_job(self, row: JobRow) -> JobRow:
        with self.session() as s:
            s.add(row)
        return row

    def get_job(self, job_id: str) -> JobRow | None:
        with self.session() as s:
            return s.get(JobRow, job_id)

    def find_job_by_idempotency_key(self, key: str) -> JobRow | None:
        with self.session() as s:
            return s.scalars(select(JobRow).where(JobRow.idempotency_key == key)).first()

    def find_reusable_job(self, request_key: str, max_age_hours: int) -> JobRow | None:
        """Latest job for the same request that is still running or completed recently."""
        cutoff = _now() - timedelta(hours=max_age_hours)
        with self.session() as s:
            rows = s.scalars(
                select(JobRow)
                .where(JobRow.request_key == request_key)
                .where(JobRow.status.in_(["queued", "running", "completed"]))
                .order_by(JobRow.created_at.desc())
            ).all()
        for row in rows:
            created = row.created_at if row.created_at.tzinfo else row.created_at.replace(tzinfo=UTC)
            if row.status in ("queued", "running") or created >= cutoff:
                return row
        return None

    def update_job(self, job_id: str, **fields: Any) -> JobRow | None:
        with self.session() as s:
            row = s.get(JobRow, job_id)
            if row is None:
                return None
            for key, value in fields.items():
                setattr(row, key, value)
            row.updated_at = _now()
            return row

    def list_jobs(self, limit: int = 50, status: str | None = None) -> list[JobRow]:
        with self.session() as s:
            query = select(JobRow).order_by(JobRow.created_at.desc()).limit(limit)
            if status:
                query = query.where(JobRow.status == status)
            return list(s.scalars(query).all())

    def jobs_with_status(self, *statuses: str) -> list[JobRow]:
        with self.session() as s:
            return list(s.scalars(select(JobRow).where(JobRow.status.in_(statuses))).all())

    # -- usage ----------------------------------------------------------------------------------

    def add_usage(self, job_id: str | None, record: UsageRecord) -> None:
        with self.session() as s:
            s.add(
                UsageRow(
                    job_id=job_id,
                    ts=record.ts,
                    stage=record.stage,
                    task=record.task,
                    provider=record.provider,
                    model=record.model,
                    input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens,
                    cache_read_tokens=record.cache_read_tokens,
                    cache_write_tokens=record.cache_write_tokens,
                    search_requests=record.search_requests,
                    cost_usd=record.cost_usd,
                    duration_ms=record.duration_ms,
                    retries=record.retries,
                    success=record.success,
                    estimated=record.estimated,
                    error=record.error,
                )
            )

    # -- documents ------------------------------------------------------------------------------

    def get_document(self, url: str) -> DocumentRow | None:
        with self.session() as s:
            return s.get(DocumentRow, url)

    def upsert_document(self, url: str, **fields: Any) -> None:
        with self.session() as s:
            row = s.get(DocumentRow, url)
            if row is None:
                row = DocumentRow(url=url)
                s.add(row)
            for key, value in fields.items():
                setattr(row, key, value)
            row.fetched_at = _now()

    # -- project registry -----------------------------------------------------------------------

    def registry_lookup(self, org_key: str, alias_keys: list[str]) -> str | None:
        if not alias_keys:
            return None
        with self.session() as s:
            row = s.scalars(
                select(ProjectRegistryRow)
                .where(ProjectRegistryRow.org_key == org_key)
                .where(ProjectRegistryRow.alias_key.in_(alias_keys))
                .order_by(ProjectRegistryRow.created_at)
            ).first()
            return row.project_uid if row else None

    def registry_register(self, org_key: str, alias_keys: list[str], project_uid: str) -> None:
        with self.session() as s:
            for alias in alias_keys:
                if s.get(ProjectRegistryRow, (org_key, alias)) is None:
                    s.add(ProjectRegistryRow(org_key=org_key, alias_key=alias, project_uid=project_uid))
