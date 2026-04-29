"""Synchronous DB access for the Streamlit dashboard.

Streamlit is sync-only, so this module uses a separate sync engine via
psycopg + DatabaseConfig.sync_url. ORM models are reused from
`inference_agent.db.models`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import streamlit as st
from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from inference_agent.db.models import ExperimentRow
from inference_agent.models_pkg.config import DatabaseConfig


@dataclass(frozen=True)
class HardwareKey:
    gpu_name: str
    gpu_count: int
    gpu_vram_mb: int
    nvlink_available: bool

    def label(self) -> str:
        nvlink = " NVLink" if self.nvlink_available else ""
        return f"{self.gpu_name} x{self.gpu_count} ({self.gpu_vram_mb}MB){nvlink}"


@dataclass(frozen=True)
class Filters:
    hardware: HardwareKey | None = None
    models: tuple[str, ...] = ()
    engines: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    date_from: datetime | None = None
    date_to: datetime | None = None


def _load_db_config() -> DatabaseConfig:
    """Build DatabaseConfig from DATABASE_* env vars (same names as the agent)."""
    raw: dict[str, Any] = {}
    for field_name in (
        "host", "port", "database", "user",
        "password", "password_env",
        "pool_size", "pool_max_overflow", "pool_timeout_sec", "echo",
    ):
        env_name = f"DATABASE_{field_name.upper()}"
        if env_name in os.environ:
            raw[field_name] = os.environ[env_name]
    return DatabaseConfig(**raw)


@st.cache_resource
def get_engine() -> Engine:
    cfg = _load_db_config()
    return create_engine(cfg.sync_url, pool_pre_ping=True)


def _get_sessionmaker():
    return sessionmaker(get_engine(), expire_on_commit=False)


@st.cache_data(ttl=300)
def list_distinct_hardware() -> list[HardwareKey]:
    Session = _get_sessionmaker()
    with Session() as session:
        rows = session.execute(
            select(
                ExperimentRow.gpu_name,
                ExperimentRow.gpu_count,
                ExperimentRow.gpu_vram_mb,
                ExperimentRow.nvlink_available,
            ).distinct()
        ).all()
    return [HardwareKey(*row) for row in rows]


@st.cache_data(ttl=300)
def list_distinct_models() -> list[str]:
    Session = _get_sessionmaker()
    with Session() as session:
        rows = session.execute(
            select(ExperimentRow.model_name).distinct()
        ).all()
    return sorted({row[0] for row in rows})


@st.cache_data(ttl=300)
def list_distinct_engines() -> list[str]:
    Session = _get_sessionmaker()
    with Session() as session:
        rows = session.execute(
            select(ExperimentRow.engine).distinct()
        ).all()
    return sorted({row[0] for row in rows})


@st.cache_data(ttl=30)
def list_experiments(filters: Filters) -> list[dict]:
    """Fetch experiments matching filters, returning the JSONB `data` payloads.

    Each item is `ExperimentResult.model_dump(mode="json")` — the same shape
    that the rest of the dashboard already expects.
    """
    Session = _get_sessionmaker()
    stmt = select(ExperimentRow).order_by(ExperimentRow.created_at.desc())

    if filters.hardware is not None:
        hw = filters.hardware
        stmt = stmt.where(
            ExperimentRow.gpu_name == hw.gpu_name,
            ExperimentRow.gpu_count == hw.gpu_count,
            ExperimentRow.gpu_vram_mb == hw.gpu_vram_mb,
            ExperimentRow.nvlink_available == hw.nvlink_available,
        )
    if filters.models:
        stmt = stmt.where(ExperimentRow.model_name.in_(filters.models))
    if filters.engines:
        stmt = stmt.where(ExperimentRow.engine.in_(filters.engines))
    if filters.statuses:
        stmt = stmt.where(ExperimentRow.status.in_(filters.statuses))
    if filters.date_from is not None:
        stmt = stmt.where(ExperimentRow.created_at >= filters.date_from)
    if filters.date_to is not None:
        stmt = stmt.where(ExperimentRow.created_at <= filters.date_to)

    with Session() as session:
        rows = session.execute(stmt).scalars().all()
    return [row.data for row in rows]
