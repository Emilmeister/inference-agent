"""HTTP client for the inference-api REST service.

Replaces the previous direct-DB `ExperimentRepository` used by the reporter
and history_loader nodes. The agent never opens a Postgres connection itself
— all storage and history queries go through this client.

Failure mode is fail-fast: any non-2xx response, malformed body, network
error, or timeout raises `APIClientError`. Reporter and history_loader
surface that error to the graph runner, which lets the agent shut down
loudly instead of silently dropping a result.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from inference_agent.models_pkg.domain import (
    ExperimentResult,
    ExperimentSummary,
    HardwareProfile,
)

logger = logging.getLogger(__name__)


class APIClientError(RuntimeError):
    """Raised when a call to the inference-api service fails.

    Wraps HTTP non-2xx responses, connection errors, and decode failures so
    callers can treat any API problem uniformly.
    """


class ExperimentApiClient:
    """Async client for inference-api.

    Mirrors the surface of the old `ExperimentRepository` so the agent nodes
    can substitute one for the other (same method names and arguments).
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_sec: float = 30.0,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not token:
            raise ValueError("token is required (agent cannot run without API auth)")
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_sec)
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "ExperimentApiClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def start(self) -> None:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                headers=self._headers,
            )

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _require_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise APIClientError(
                "ExperimentApiClient used before start() — call `async with client:` "
                "or `await client.start()` first"
            )
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        session = self._require_session()
        try:
            async with session.request(
                method, self._url(path), json=json, params=params
            ) as resp:
                body_text = await resp.text()
                if resp.status >= 400:
                    raise APIClientError(
                        f"{method} {path} failed with HTTP {resp.status}: {body_text[:512]}"
                    )
                if not body_text:
                    return None
                try:
                    return await resp.json(content_type=None)
                except aiohttp.ContentTypeError as e:
                    raise APIClientError(
                        f"{method} {path}: response is not JSON ({e})"
                    ) from e
        except aiohttp.ClientError as e:
            raise APIClientError(f"{method} {path}: HTTP error {e}") from e

    # ── Agent-facing surface ───────────────────────────────────────────────

    async def insert_experiment(self, result: ExperimentResult) -> None:
        """POST /experiments — persist a single experiment result.

        Mirrors `ExperimentRepository.insert_experiment`.
        """
        await self._request(
            "POST",
            "/experiments",
            json=result.model_dump(mode="json"),
        )
        logger.info(
            "Posted experiment %s to inference-api (engine=%s, status=%s)",
            result.experiment_id,
            result.engine.value,
            result.status.value,
        )

    async def find_top_for_hardware(
        self,
        hardware: HardwareProfile,
        model_name: str,
        latency_threshold_ms: float,
        limit: int = 2,
    ) -> list[ExperimentSummary]:
        """GET /experiments/top — top-N per category for this hardware+model.

        Server enforces homogeneous-cluster invariants; we send the primary GPU.
        """
        if not hardware.gpus:
            return []
        primary = hardware.gpus[0]
        params = {
            "gpu_name": primary.name,
            "gpu_count": str(hardware.gpu_count),
            "gpu_vram_mb": str(primary.vram_total_mb),
            "nvlink_available": "true" if hardware.nvlink_available else "false",
            "model_name": model_name,
            "latency_threshold_ms": str(latency_threshold_ms),
            "limit": str(limit),
        }
        payload = await self._request("GET", "/experiments/top", params=params)
        if not isinstance(payload, dict) or "summaries" not in payload:
            raise APIClientError(
                f"GET /experiments/top: unexpected payload shape {type(payload).__name__}"
            )
        return [ExperimentSummary.model_validate(s) for s in payload["summaries"]]
