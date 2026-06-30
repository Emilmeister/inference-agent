"""End-to-end test: real FastAPI service + real Postgres + agent HTTP client.

Verifies the full contract:
  * agent POSTs an ExperimentResult through `ExperimentApiClient`
  * REST service persists it through `ExperimentRepository`
  * agent GETs /experiments/top and gets the same result back as a summary

Run with `pytest -m integration` (requires Docker + testcontainers).
"""

from __future__ import annotations

import asyncio
import socket
import threading

import pytest
import uvicorn

from inference_agent.api_client import ExperimentApiClient
from inference_agent.models import (
    BenchmarkResult,
    EngineType,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    GPUInfo,
    HardwareProfile,
)
from inference_api.app import create_app
from inference_api.config import (
    ApiServiceConfig,
    AuthConfig,
    DatabaseConfig,
    ServerConfig,
)

pytestmark = pytest.mark.integration

try:
    from testcontainers.postgres import PostgresContainer
except ImportError:  # pragma: no cover
    pytest.skip("testcontainers[postgres] not installed", allow_module_level=True)


TOKEN = "e2e-token"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def pg_container():
    with PostgresContainer("postgres:16") as pg:
        yield pg


@pytest.fixture(scope="session")
def api_service(pg_container):
    """Boot the real FastAPI app on a background thread with its own loop."""
    raw_url = pg_container.get_connection_url()
    # testcontainers gives "postgresql+psycopg2://user:pass@host:port/db"
    # — extract pieces for DatabaseConfig.
    from urllib.parse import urlparse

    parsed = urlparse(raw_url.replace("postgresql+psycopg2", "postgresql"))
    db_cfg = DatabaseConfig(
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        database=(parsed.path or "/test").lstrip("/"),
        user=parsed.username or "test",
        password=parsed.password or "",
        password_env="__UNUSED_E2E_PASSWORD_ENV__",
    )
    port = _free_port()
    config = ApiServiceConfig(
        server=ServerConfig(host="127.0.0.1", port=port, log_level="warning"),
        auth=AuthConfig(token=TOKEN, token_env="__UNUSED_E2E_TOKEN_ENV__"),
        database=db_cfg,
    )

    app = create_app(config)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for service to come up.
    deadline = 30.0
    while deadline > 0 and not server.started:
        thread.join(timeout=0.1)
        deadline -= 0.1
    if not server.started:
        raise RuntimeError("inference-api did not start within 30s")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=10)


def _make_result(exp_id: str, throughput: float, ttft: float) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=exp_id,
        engine=EngineType.VLLM,
        model="Qwen/Qwen2.5-7B-Instruct",
        hardware=HardwareProfile(
            gpus=[GPUInfo(index=0, name="H100", vram_total_mb=81920, vram_free_mb=80000)],
            gpu_count=1,
            nvlink_available=False,
            model_name="Qwen/Qwen2.5-7B-Instruct",
        ),
        config=ExperimentConfig(engine=EngineType.VLLM),
        status=ExperimentStatus.SUCCESS,
        correctness_gate_passed=True,
        benchmark=BenchmarkResult(
            peak_output_tokens_per_sec=throughput,
            low_concurrency_ttft_p95_ms=ttft,
        ),
    )


@pytest.mark.asyncio
async def test_roundtrip_insert_and_top(api_service):
    async with ExperimentApiClient(base_url=api_service, token=TOKEN) as client:
        await client.insert_experiment(_make_result("e2e_a", 500.0, 80.0))
        await client.insert_experiment(_make_result("e2e_b", 900.0, 200.0))

        hardware = HardwareProfile(
            gpus=[GPUInfo(index=0, name="H100", vram_total_mb=81920, vram_free_mb=80000)],
            gpu_count=1,
            nvlink_available=False,
            model_name="Qwen/Qwen2.5-7B-Instruct",
        )
        summaries = await client.find_top_for_hardware(
            hardware=hardware,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            latency_threshold_ms=500.0,
            limit=2,
        )

    ids = {s.experiment_id for s in summaries}
    assert ids == {"e2e_a", "e2e_b"}


@pytest.mark.asyncio
async def test_unauthorized_without_token(api_service):
    from inference_agent.api_client import APIClientError

    async with ExperimentApiClient(base_url=api_service, token="wrong") as client:
        with pytest.raises(APIClientError, match="HTTP 401"):
            await client.insert_experiment(_make_result("denied", 1.0, 1.0))


@pytest.mark.asyncio
async def test_get_experiment_returns_full_result(api_service):
    async with ExperimentApiClient(base_url=api_service, token=TOKEN) as client:
        await client.insert_experiment(_make_result("e2e_full", 700.0, 90.0))
        result = await client.get_experiment("e2e_full")
    assert result.experiment_id == "e2e_full"
    assert result.benchmark.peak_output_tokens_per_sec == 700.0


@pytest.mark.asyncio
async def test_quality_run_upsert_idempotency_and_list(api_service):
    async with ExperimentApiClient(base_url=api_service, token=TOKEN) as client:
        run_id = "fp_quality-so_testing"
        base = {
            "id": run_id,
            "fingerprint": "fp_quality",
            "suite": "so_testing",
            "suite_version": "1.0",
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "gpu_name": "H100",
            "gpu_count": 1,
            "gpu_vram_mb": 81920,
            "nvlink_available": False,
            "experiment_ids": ["e2e_a", "e2e_b"],
            "categories": ["agentic", "latency"],
        }
        # First upsert: running, no score.
        await client.upsert_quality_run({**base, "status": "running", "score": None, "data": {}})
        running = await client.get_quality_run(run_id)
        assert running["status"] == "running"

        # Second upsert (same id): done with a score → replaces, not duplicates.
        await client.upsert_quality_run({
            **base, "status": "done", "score": 88.5, "data": {"suites": {}},
        })
        done = await client.get_quality_run(run_id)
        assert done["status"] == "done"
        assert done["score"] == 88.5

        listed = await client._request(
            "GET", "/quality/runs", params={"fingerprint": "fp_quality"},
        )
        assert len(listed["runs"]) == 1
        assert listed["runs"][0]["id"] == run_id

        missing = await client.get_quality_run("does-not-exist")
        assert missing is None
