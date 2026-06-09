"""Unit tests for the baseline_injector node."""

from __future__ import annotations

import pytest

from inference_agent.models import AgentConfig, EngineType, ExperimentConfig
from inference_agent.nodes.baseline_injector import baseline_injector_node


@pytest.mark.asyncio
async def test_injects_baseline_as_current_config():
    baseline = ExperimentConfig(
        engine=EngineType.VLLM,
        tensor_parallel_size=8,
        max_model_len=202752,
        enable_prefix_caching=True,
    )
    config = AgentConfig(model_name="zai-org/GLM-4.7-FP8", baseline=baseline)

    out = await baseline_injector_node({"config": config, "baseline_pending": True})

    assert out["baseline_pending"] is False
    injected = out["current_config"]
    assert injected.is_baseline is True
    assert injected.engine == EngineType.VLLM
    assert injected.tensor_parallel_size == 8
    # Rationale auto-filled so the dashboard / history show why it ran.
    assert "baseline" in injected.rationale.lower()


@pytest.mark.asyncio
async def test_does_not_mutate_shared_baseline():
    baseline = ExperimentConfig(engine=EngineType.VLLM)
    config = AgentConfig(baseline=baseline)

    out = await baseline_injector_node({"config": config, "baseline_pending": True})

    # The shared AgentConfig.baseline must stay un-flagged; only the injected
    # copy carries is_baseline=True.
    assert config.baseline.is_baseline is False
    assert out["current_config"].is_baseline is True


@pytest.mark.asyncio
async def test_no_baseline_clears_flag_gracefully():
    config = AgentConfig(baseline=None)
    out = await baseline_injector_node({"config": config, "baseline_pending": True})
    assert out["baseline_pending"] is False
    assert "current_config" not in out
