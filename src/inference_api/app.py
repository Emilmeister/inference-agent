"""FastAPI application factory.

The factory wires the lifespan (DB engine + optional proxy tunnel + alembic
migrations), attaches the repository and auth token to `app.state`, and
registers the route modules. Tests build their own app via this factory with
a pre-baked `ApiServiceConfig`.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from inference_api.config import ApiServiceConfig
from inference_api.db import ExperimentRepository, init_schema
from inference_api.db_proxy import DBProxyTunnel
from inference_api.routes import experiments, health, meta

logger = logging.getLogger(__name__)


def create_app(config: ApiServiceConfig) -> FastAPI:
    if not config.auth.token:
        raise RuntimeError(
            "auth.token is not set — refuse to start without bearer auth. "
            "Set INFERENCE_API_TOKEN or auth.token in config."
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db_cfg = config.database

        tunnel: DBProxyTunnel | None = None
        proxy_url = db_cfg.effective_proxy_url
        if proxy_url:
            tunnel = DBProxyTunnel(proxy_url, db_cfg.host, db_cfg.port)
            local_host, local_port = await tunnel.start_async()
            effective_db_cfg = db_cfg.with_endpoint(local_host, local_port)
        else:
            effective_db_cfg = db_cfg

        engine = create_async_engine(
            effective_db_cfg.url,
            pool_size=effective_db_cfg.pool_size,
            max_overflow=effective_db_cfg.pool_max_overflow,
            pool_timeout=effective_db_cfg.pool_timeout_sec,
            echo=effective_db_cfg.echo,
        )
        try:
            await init_schema(engine)
            sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
            repo = ExperimentRepository(sessionmaker)

            app.state.repository = repo
            app.state.auth_token = config.auth.token
            logger.info(
                "inference-api ready — DB %s:%d/%s",
                db_cfg.host, db_cfg.port, db_cfg.database,
            )
            yield
        finally:
            await engine.dispose()
            if tunnel is not None:
                await tunnel.stop_async()

    app = FastAPI(
        title="Inference Benchmark API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(experiments.router)
    app.include_router(meta.router)

    return app
