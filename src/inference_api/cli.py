"""CLI entrypoint — loads config, applies env overrides, runs uvicorn."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import uvicorn
import yaml

from inference_api.app import create_app
from inference_api.config import ApiServiceConfig

logger = logging.getLogger(__name__)


_DATABASE_ENV_FIELDS = (
    "host",
    "port",
    "database",
    "user",
    "password",
    "password_env",
    "pool_size",
    "pool_max_overflow",
    "pool_timeout_sec",
    "echo",
    "http_proxy",
    "https_proxy",
)


_SERVER_ENV_FIELDS = ("host", "port", "log_level")


_AUTH_ENV_FIELDS = ("token", "token_env")


def _apply_env_overrides(raw: dict) -> None:
    db_section = raw.setdefault("database", {})
    for field in _DATABASE_ENV_FIELDS:
        env_name = f"DATABASE_{field.upper()}"
        if env_name in os.environ:
            db_section[field] = os.environ[env_name]

    server_section = raw.setdefault("server", {})
    for field in _SERVER_ENV_FIELDS:
        env_name = f"INFERENCE_API_SERVER_{field.upper()}"
        if env_name in os.environ:
            server_section[field] = os.environ[env_name]

    auth_section = raw.setdefault("auth", {})
    for field in _AUTH_ENV_FIELDS:
        env_name = f"INFERENCE_API_{field.upper()}"
        if env_name in os.environ:
            auth_section[field] = os.environ[env_name]


def _load_config(path: str | None) -> ApiServiceConfig:
    raw: dict = {}
    if path:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    _apply_env_overrides(raw)
    return ApiServiceConfig(**raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference Benchmark REST API")
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to YAML config (optional — defaults + env vars are enough)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.config and not os.path.exists(args.config):
        print(f"Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = _load_config(args.config)
    app = create_app(config)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
    )


if __name__ == "__main__":
    main()
