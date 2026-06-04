"""Configuration for the REST service.

Two layers:

  * `DatabaseConfig` — Postgres connection params. Moved here from the agent
    package since the agent no longer talks to Postgres directly.
  * `ApiServiceConfig` — bind address, auth token, nested DatabaseConfig.

Loaded from a YAML file by `inference_api.cli`. Any field of `database` /
`server` / `auth` may be overridden via env vars (`DATABASE_*`,
`INFERENCE_API_*`).
"""

from __future__ import annotations

import os
from urllib.parse import quote_plus

from pydantic import BaseModel, Field, model_validator


class DatabaseConfig(BaseModel):
    """Postgres connection parameters.

    Password may be set directly, via `password_env` (env var name), or via
    the `DATABASE_PASSWORD` env override applied by `inference_api.cli`.

    `http_proxy` / `https_proxy` route DB traffic through an HTTP CONNECT
    proxy (Postgres is TCP, not HTTP, so this is implemented via a local TCP
    forwarder — see `inference_api.db_proxy.DBProxyTunnel`). If unset,
    standard `HTTP_PROXY` / `HTTPS_PROXY` env vars are honored.
    """

    host: str = "localhost"
    port: int = 5432
    database: str = "inference_agent"
    user: str = "inference_agent"
    password: str | None = None
    password_env: str = "DB_PASSWORD"

    pool_size: int = 5
    pool_max_overflow: int = 10
    pool_timeout_sec: int = 30
    echo: bool = False

    http_proxy: str | None = None
    https_proxy: str | None = None

    @model_validator(mode="after")
    def _resolve_password(self) -> "DatabaseConfig":
        if not self.password and self.password_env:
            self.password = os.environ.get(self.password_env)
        return self

    @model_validator(mode="after")
    def _resolve_proxy(self) -> "DatabaseConfig":
        if not self.http_proxy:
            self.http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
        if not self.https_proxy:
            self.https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        return self

    @property
    def effective_proxy_url(self) -> str | None:
        return self.https_proxy or self.http_proxy

    def with_endpoint(self, host: str, port: int) -> "DatabaseConfig":
        return self.model_copy(update={"host": host, "port": port})

    @property
    def url(self) -> str:
        """Async URL for SQLAlchemy + asyncpg."""
        pwd = quote_plus(self.password or "")
        usr = quote_plus(self.user)
        return f"postgresql+asyncpg://{usr}:{pwd}@{self.host}:{self.port}/{self.database}"

    @property
    def sync_url(self) -> str:
        """Sync URL for SQLAlchemy + psycopg (used by alembic CLI)."""
        pwd = quote_plus(self.password or "")
        usr = quote_plus(self.user)
        return f"postgresql+psycopg://{usr}:{pwd}@{self.host}:{self.port}/{self.database}"


class ServerConfig(BaseModel):
    """uvicorn bind settings."""

    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"


class AuthConfig(BaseModel):
    """Static Bearer token for clients.

    Provide either `token` directly or `token_env` (env var name). If neither
    is set after env override, the service refuses to start — running without
    auth is too easy to do accidentally.
    """

    token: str | None = None
    token_env: str = "INFERENCE_API_TOKEN"

    @model_validator(mode="after")
    def _resolve_token(self) -> "AuthConfig":
        if not self.token and self.token_env:
            self.token = os.environ.get(self.token_env)
        return self


class ApiServiceConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
