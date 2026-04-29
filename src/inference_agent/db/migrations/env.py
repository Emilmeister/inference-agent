"""Alembic environment.

Resolves the database URL in this order:
  1. `cfg.attributes["database_url"]` — set programmatically when migrations
     are run from `init_schema` (production startup, tests).
  2. `sqlalchemy.url` from alembic.ini — only used when explicitly set; we
     intentionally leave it blank in the committed alembic.ini.
  3. `DATABASE_*` env vars + `DB_PASSWORD` — same vars the agent reads at
     startup. This is the path used by the `alembic` CLI during development:
     `export DB_PASSWORD=...; alembic upgrade head`.

Target metadata is `Base.metadata`, so `alembic revision --autogenerate`
diffs the ORM models against the DB.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from inference_agent.db.models import Base
from inference_agent.models_pkg.config import DatabaseConfig

# Alembic Config object.
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _resolve_url() -> str:
    url = config.attributes.get("database_url")
    if url:
        return url
    url = config.get_main_option("sqlalchemy.url") or ""
    if url:
        return url
    # Last resort: build from DATABASE_* env vars the same way cli does.
    raw = {}
    for field_name in ("host", "port", "database", "user", "password", "password_env"):
        env_name = f"DATABASE_{field_name.upper()}"
        if env_name in os.environ:
            raw[field_name] = os.environ[env_name]
    db_cfg = DatabaseConfig(**raw)
    return db_cfg.sync_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL to stdout, no DB connection)."""
    context.configure(
        url=_resolve_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live DB."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _resolve_url()

    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
