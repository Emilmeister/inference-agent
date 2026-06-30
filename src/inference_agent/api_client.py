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
    callers can treat any API problem uniformly. `status_code` is set for HTTP
    error responses (None for transport/decode failures) so callers can treat
    specific codes — e.g. a 404 on an optional endpoint — as a soft outcome.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


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
            # trust_env=True makes aiohttp read HTTP_PROXY / HTTPS_PROXY /
            # NO_PROXY from the environment the way requests does. Without
            # this the client tries to dial the API host directly and hangs
            # on networks where outbound traffic must go through a corporate
            # HTTP proxy.
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                headers=self._headers,
                trust_env=True,
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
                        f"{method} {path} failed with HTTP {resp.status}: {body_text[:512]}",
                        status_code=resp.status,
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

    async def get_experiment(self, experiment_id: str) -> ExperimentResult:
        """GET /experiments/{id} — full ExperimentResult for a finalist.

        Used by quality_finalize to recover the finalist's launch config so it
        can relaunch the container. Raises APIClientError (incl. 404) on miss.
        """
        payload = await self._request("GET", f"/experiments/{experiment_id}")
        if not isinstance(payload, dict) or "data" not in payload:
            raise APIClientError(
                f"GET /experiments/{experiment_id}: unexpected payload shape "
                f"{type(payload).__name__}"
            )
        return ExperimentResult.model_validate(payload["data"])

    async def get_quality_run(self, run_id: str) -> dict[str, Any] | None:
        """GET /quality/runs/{id} — a stored quality run, or None if absent.

        Used for idempotency: skip a (fingerprint, suite) whose run is done.
        """
        try:
            return await self._request("GET", f"/quality/runs/{run_id}")
        except APIClientError as e:
            if e.status_code == 404:
                return None
            raise

    async def upsert_quality_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /quality/runs — insert or replace a quality run by id."""
        return await self._request("POST", "/quality/runs", json=payload)

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

    async def find_baseline(
        self,
        hardware: HardwareProfile,
        model_name: str,
    ) -> ExperimentSummary | None:
        """GET /experiments/baseline — the latest baseline for this hw+model.

        Returns None when no baseline has been measured yet for this exact
        hardware + model, so the agent knows it still needs to run one.
        """
        if not hardware.gpus:
            return None
        primary = hardware.gpus[0]
        params = {
            "gpu_name": primary.name,
            "gpu_count": str(hardware.gpu_count),
            "gpu_vram_mb": str(primary.vram_total_mb),
            "nvlink_available": "true" if hardware.nvlink_available else "false",
            "model_name": model_name,
        }
        try:
            payload = await self._request("GET", "/experiments/baseline", params=params)
        except APIClientError as e:
            # Tolerate an older inference-api that predates the /baseline route:
            # FastAPI then matches "baseline" against /experiments/{id} and 404s.
            # A missing optional endpoint means "no baseline", not a fatal run
            # failure — degrade gracefully instead of crashing the agent.
            if e.status_code == 404:
                logger.warning(
                    "GET /experiments/baseline returned 404 — inference-api "
                    "likely predates the baseline endpoint; treating as 'no "
                    "baseline'. Update the API to enable baseline anchoring."
                )
                return None
            raise
        if not payload:
            return None
        if not isinstance(payload, dict) or "summary" not in payload:
            raise APIClientError(
                f"GET /experiments/baseline: unexpected payload shape "
                f"{type(payload).__name__}"
            )
        summary = payload["summary"]
        if summary is None:
            return None
        return ExperimentSummary.model_validate(summary)
