"""Async engine bootstrap and schema initialization via Alembic."""

from __future__ import annotations

import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncEngine

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _async_url_to_sync(url: str) -> str:
    """Strip the async driver from a SQLAlchemy URL so alembic can use it.

    Alembic's `engine_from_config` runs synchronously and can't drive asyncpg.
    We swap to psycopg (declared as a runtime dep alongside asyncpg).
    """
    return url.replace("+asyncpg", "+psycopg")


def _alembic_config(database_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.set_main_option("version_locations", str(_MIGRATIONS_DIR / "versions"))
    # env.py reads this via cfg.attributes["database_url"]; we avoid putting
    # the URL in `sqlalchemy.url` because alembic interprets `%` in URLs as
    # config interpolation tokens.
    cfg.attributes["database_url"] = _async_url_to_sync(database_url)
    return cfg


def _run_upgrade(database_url: str) -> None:
    """Synchronous: bring the DB schema to head."""
    cfg = _alembic_config(database_url)
    command.upgrade(cfg, "head")


async def init_schema(engine: AsyncEngine) -> None:
    """Bring the schema to the latest alembic revision. Idempotent.

    First real round-trip to Postgres — if the DB is unreachable, this is
    where startup fails fast. The async engine already holds the URL we
    need; we extract it, swap the driver, and run alembic in a worker
    thread so the event loop isn't blocked by sync DB I/O.
    """
    url = str(engine.url.render_as_string(hide_password=False))
    await asyncio.to_thread(_run_upgrade, url)
