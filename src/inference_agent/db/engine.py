"""Async engine bootstrap and schema initialization."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine

from inference_agent.db.models import Base


async def init_schema(engine: AsyncEngine) -> None:
    """Create all tables if they don't exist. Idempotent.

    Performs the first real round-trip to Postgres — if the DB is unreachable,
    this is where startup fails fast.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
