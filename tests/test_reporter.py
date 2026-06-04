"""Tests for reporter — persistence via the REST API client."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from inference_agent.api_client import ExperimentApiClient
from inference_agent.models import (
    AgentConfig,
    EngineType,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    GPUInfo,
    HardwareProfile,
)
from inference_agent.nodes.reporter import make_reporter_node


def _make_result(exp_id: str = "test123") -> ExperimentResult:
    return ExperimentResult(
        experiment_id=exp_id,
        engine=EngineType.VLLM,
        model="test/model",
        hardware=HardwareProfile(
            gpus=[GPUInfo(index=0, name="A100", vram_total_mb=81920, vram_free_mb=80000)],
            gpu_count=1,
            nvlink_available=False,
            model_name="test/model",
        ),
        config=ExperimentConfig(engine=EngineType.VLLM, max_model_len=4096),
        status=ExperimentStatus.SUCCESS,
    )


@pytest.mark.asyncio
async def test_reporter_posts_current_result():
    client = AsyncMock(spec=ExperimentApiClient)
    node = make_reporter_node(client)
    result = _make_result()

    out = await node({"config": AgentConfig(), "current_result": result})

    client.insert_experiment.assert_awaited_once_with(result)
    assert out == {}


@pytest.mark.asyncio
async def test_reporter_skips_when_no_result():
    client = AsyncMock(spec=ExperimentApiClient)
    node = make_reporter_node(client)

    out = await node({"config": AgentConfig(), "current_result": None})

    client.insert_experiment.assert_not_awaited()
    assert out == {}


@pytest.mark.asyncio
async def test_reporter_propagates_api_errors():
    client = AsyncMock(spec=ExperimentApiClient)
    client.insert_experiment.side_effect = RuntimeError("API down")
    node = make_reporter_node(client)

    with pytest.raises(RuntimeError, match="API down"):
        await node({"config": AgentConfig(), "current_result": _make_result()})
