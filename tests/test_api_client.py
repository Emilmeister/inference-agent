"""Unit tests for `ExperimentApiClient` (the agent's HTTP client).

Spins up a tiny aiohttp server in-process to exercise the wire contract — no
mocking of HTTP internals, just real round-trips against a fixture app.
"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

from inference_agent.api_client import APIClientError, ExperimentApiClient
from inference_agent.models import (
    BenchmarkResult,
    EngineType,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    ExperimentSummary,
    GPUInfo,
    HardwareProfile,
)


TOKEN = "test-token"


def _make_result(exp_id: str = "exp1") -> ExperimentResult:
    return ExperimentResult(
        experiment_id=exp_id,
        engine=EngineType.VLLM,
        model="m/x",
        hardware=HardwareProfile(
            gpus=[GPUInfo(index=0, name="H100", vram_total_mb=80000, vram_free_mb=70000)],
            gpu_count=1,
            nvlink_available=False,
            model_name="m/x",
        ),
        config=ExperimentConfig(engine=EngineType.VLLM),
        status=ExperimentStatus.SUCCESS,
        correctness_gate_passed=True,
        benchmark=BenchmarkResult(
            peak_output_tokens_per_sec=100.0,
            low_concurrency_ttft_p95_ms=50.0,
        ),
    )


@pytest_asyncio.fixture
async def fake_server():
    """Tiny stand-in for inference-api that captures requests for assertion."""
    state: dict[str, Any] = {"posts": [], "top_calls": []}

    async def check_auth(request: web.Request) -> None:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {TOKEN}":
            raise web.HTTPUnauthorized()

    async def post_experiments(request: web.Request) -> web.Response:
        await check_auth(request)
        body = await request.json()
        state["posts"].append(body)
        return web.json_response({"experiment_id": body["experiment_id"]}, status=201)

    async def get_top(request: web.Request) -> web.Response:
        await check_auth(request)
        state["top_calls"].append(dict(request.query))
        summary = ExperimentSummary(
            experiment_id="hist1",
            engine=EngineType.VLLM,
            status=ExperimentStatus.SUCCESS,
            peak_throughput=500.0,
            low_concurrency_ttft_p95=50.0,
            correctness_gate_passed=True,
        )
        return web.json_response({"summaries": [summary.model_dump(mode="json")]})

    async def error_endpoint(request: web.Request) -> web.Response:
        await check_auth(request)
        return web.Response(status=500, text="boom")

    app = web.Application()
    app.router.add_post("/experiments", post_experiments)
    app.router.add_get("/experiments/top", get_top)
    app.router.add_post("/error", error_endpoint)
    server = TestServer(app)
    await server.start_server()
    server.state = state
    try:
        yield server
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_insert_experiment_posts_payload(fake_server):
    base_url = f"http://{fake_server.host}:{fake_server.port}"
    async with ExperimentApiClient(base_url=base_url, token=TOKEN) as client:
        await client.insert_experiment(_make_result("exp1"))

    assert len(fake_server.state["posts"]) == 1
    assert fake_server.state["posts"][0]["experiment_id"] == "exp1"


@pytest.mark.asyncio
async def test_find_top_for_hardware_sends_params(fake_server):
    base_url = f"http://{fake_server.host}:{fake_server.port}"
    hw = HardwareProfile(
        gpus=[GPUInfo(index=0, name="H100", vram_total_mb=80000, vram_free_mb=70000)],
        gpu_count=2,
        nvlink_available=True,
        model_name="m/x",
    )
    async with ExperimentApiClient(base_url=base_url, token=TOKEN) as client:
        summaries = await client.find_top_for_hardware(
            hardware=hw,
            model_name="m/x",
            latency_threshold_ms=500.0,
            limit=2,
        )

    assert [s.experiment_id for s in summaries] == ["hist1"]
    call = fake_server.state["top_calls"][0]
    assert call["gpu_name"] == "H100"
    assert call["gpu_count"] == "2"
    assert call["gpu_vram_mb"] == "80000"
    assert call["nvlink_available"] == "true"
    assert call["model_name"] == "m/x"
    assert call["latency_threshold_ms"] == "500.0"
    assert call["limit"] == "2"


@pytest.mark.asyncio
async def test_http_error_raises_api_client_error(fake_server):
    base_url = f"http://{fake_server.host}:{fake_server.port}"
    async with ExperimentApiClient(base_url=base_url, token=TOKEN) as client:
        with pytest.raises(APIClientError, match="HTTP 500"):
            await client._request("POST", "/error", json={})


@pytest.mark.asyncio
async def test_auth_failure_raises(fake_server):
    base_url = f"http://{fake_server.host}:{fake_server.port}"
    async with ExperimentApiClient(base_url=base_url, token="wrong") as client:
        with pytest.raises(APIClientError, match="HTTP 401"):
            await client.insert_experiment(_make_result())


def test_client_requires_token():
    with pytest.raises(ValueError, match="token is required"):
        ExperimentApiClient(base_url="http://x", token="")


def test_client_requires_base_url():
    with pytest.raises(ValueError, match="base_url is required"):
        ExperimentApiClient(base_url="", token="t")
