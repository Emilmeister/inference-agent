"""Unit tests for the history_loader node."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from inference_agent.db.repository import ExperimentRepository
from inference_agent.models import (
    AgentConfig,
    EngineType,
    ExperimentStatus,
    ExperimentSummary,
    GPUInfo,
    HardwareProfile,
)
from inference_agent.nodes.history_loader import make_history_loader_node


def _hw(gpu_name: str = "NVIDIA H100") -> HardwareProfile:
    return HardwareProfile(
        gpus=[GPUInfo(index=0, name=gpu_name, vram_total_mb=81920, vram_free_mb=80000)],
        gpu_count=1,
        nvlink_available=False,
        model_name="Qwen/Qwen2.5-7B-Instruct",
    )


def _summary(exp_id: str) -> ExperimentSummary:
    return ExperimentSummary(
        experiment_id=exp_id,
        engine=EngineType.VLLM,
        status=ExperimentStatus.SUCCESS,
        peak_throughput=500.0,
        low_concurrency_ttft_p95=50.0,
        correctness_gate_passed=True,
    )


@pytest.mark.asyncio
async def test_history_loader_populates_state():
    repo = AsyncMock(spec=ExperimentRepository)
    summaries = [_summary("a"), _summary("b")]
    repo.find_top_for_hardware.return_value = summaries

    node = make_history_loader_node(repo)
    config = AgentConfig(model_name="Qwen/Qwen2.5-7B-Instruct")
    out = await node({"config": config, "hardware": _hw()})

    assert out == {"loaded_top_history": summaries}
    repo.find_top_for_hardware.assert_awaited_once()
    kwargs = repo.find_top_for_hardware.call_args.kwargs
    assert kwargs["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert kwargs["limit"] == 2
    assert kwargs["latency_threshold_ms"] == config.benchmark.latency_threshold_ms


@pytest.mark.asyncio
async def test_history_loader_handles_missing_hardware():
    repo = AsyncMock(spec=ExperimentRepository)
    node = make_history_loader_node(repo)
    out = await node({"config": AgentConfig()})

    assert out == {"loaded_top_history": []}
    repo.find_top_for_hardware.assert_not_awaited()
