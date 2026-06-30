"""Tests for the quality_finalize node (mocked client, engine, runners)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from inference_agent.models import (
    AgentConfig,
    EngineType,
    ExperimentConfig,
    ExperimentResult,
    GPUInfo,
    HardwareProfile,
)
from inference_agent.nodes import quality_finalize
from inference_agent.nodes.quality_finalize import make_quality_finalize_node
from inference_agent.quality.runner import SuiteResult


def _hardware() -> HardwareProfile:
    return HardwareProfile(
        gpus=[GPUInfo(index=0, name="H100", vram_total_mb=81920, vram_free_mb=80000)],
        gpu_count=1,
        nvlink_available=False,
        model_name="test/model",
    )


def _result(exp_id: str, **cfg_kw) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=exp_id,
        engine=EngineType.VLLM,
        model="test/model",
        hardware=_hardware(),
        config=ExperimentConfig(engine=EngineType.VLLM, experiment_id=exp_id, **cfg_kw),
    )


def _config(**quality_kw) -> AgentConfig:
    cfg = AgentConfig(model_name="test/model")
    cfg.quality.enabled = True
    cfg.quality.so_testing.enabled = True
    cfg.quality.terminal_bench.enabled = False
    for k, v in quality_kw.items():
        setattr(cfg.quality, k, v)
    return cfg


def _state(config: AgentConfig, **best) -> dict:
    base = {
        "config": config,
        "hardware": _hardware(),
        "best_agentic_config_id": "exp_a",
        "best_latency_config_id": "exp_l",
        "best_balanced_config_id": "exp_b",
    }
    base.update(best)
    return base


@pytest.fixture(autouse=True)
def _patch_infra(monkeypatch):
    """Stub container relaunch + teardown so no nerdctl is invoked."""
    monkeypatch.setattr(
        quality_finalize, "_start_engine",
        AsyncMock(return_value=("container-id", [], 1.0)),
    )
    monkeypatch.setattr(quality_finalize, "stop_container", AsyncMock())


@pytest.mark.asyncio
async def test_disabled_quality_is_noop():
    cfg = AgentConfig()  # quality disabled by default
    client = AsyncMock()
    node = make_quality_finalize_node(client)
    assert await node({"config": cfg}) == {}
    client.get_experiment.assert_not_called()


@pytest.mark.asyncio
async def test_runs_so_testing_for_each_distinct_finalist(monkeypatch):
    # Three finalists, three distinct quantizations → three fingerprints.
    results = {
        "exp_a": _result("exp_a", quantization="fp8"),
        "exp_l": _result("exp_l", quantization=None),
        "exp_b": _result("exp_b", quantization="awq"),
    }
    client = AsyncMock()
    client.get_experiment.side_effect = lambda eid: results[eid]
    client.get_quality_run.return_value = None  # nothing done yet

    so_calls = []

    async def fake_so(cfg, base_url, model):
        so_calls.append(model)
        return SuiteResult(suite="so_testing", status="done", score=90.0, data={"ok": True})

    monkeypatch.setattr(quality_finalize, "run_so_testing", fake_so)

    node = make_quality_finalize_node(client)
    await node(_state(_config(fingerprint_dedup=True)))

    # 3 distinct fingerprints → engine relaunched 3×, so-testing run 3×.
    assert quality_finalize._start_engine.await_count == 3
    assert len(so_calls) == 3
    # Each suite persisted running + done = 2 upserts per group.
    done = [c for c in client.upsert_quality_run.await_args_list
            if c.args[0]["status"] == "done"]
    assert len(done) == 3
    assert all(c.args[0]["suite"] == "so_testing" for c in done)


@pytest.mark.asyncio
async def test_fingerprint_dedup_collapses_identical_finalists(monkeypatch):
    # All three finalists share the same config → one fingerprint → one run.
    same = _result("exp_a", quantization="fp8")
    results = {"exp_a": same, "exp_l": same, "exp_b": same}
    client = AsyncMock()
    client.get_experiment.side_effect = lambda eid: results[eid]
    client.get_quality_run.return_value = None

    monkeypatch.setattr(
        quality_finalize, "run_so_testing",
        AsyncMock(return_value=SuiteResult(suite="so_testing", status="done", score=1.0)),
    )

    node = make_quality_finalize_node(client)
    await node(_state(_config(fingerprint_dedup=True)))

    assert quality_finalize._start_engine.await_count == 1
    done = [c for c in client.upsert_quality_run.await_args_list
            if c.args[0]["status"] == "done"]
    assert len(done) == 1
    # The single run is attributed to all three finalist experiments that
    # share the fingerprint, with their leaderboard categories.
    assert set(done[0].args[0]["experiment_ids"]) == {"exp_a", "exp_l", "exp_b"}
    assert set(done[0].args[0]["categories"]) == {"agentic", "latency", "balanced"}


@pytest.mark.asyncio
async def test_idempotency_skips_done_suite(monkeypatch):
    results = {"exp_a": _result("exp_a"), "exp_l": _result("exp_a"), "exp_b": _result("exp_a")}
    client = AsyncMock()
    client.get_experiment.side_effect = lambda eid: results.get(eid, _result(eid))
    client.get_quality_run.return_value = {"status": "done"}  # already validated

    so = AsyncMock()
    monkeypatch.setattr(quality_finalize, "run_so_testing", so)

    node = make_quality_finalize_node(client)
    await node(_state(_config()))

    so.assert_not_called()
    quality_finalize._start_engine.assert_not_awaited()


@pytest.mark.asyncio
async def test_relaunch_failure_records_failed(monkeypatch):
    from inference_agent.models import ExperimentError

    monkeypatch.setattr(
        quality_finalize, "_start_engine",
        AsyncMock(return_value=(None, [ExperimentError(stage="startup", message="oom")], 0.0)),
    )
    client = AsyncMock()
    client.get_experiment.side_effect = lambda eid: _result(eid)
    client.get_quality_run.return_value = None
    so = AsyncMock()
    monkeypatch.setattr(quality_finalize, "run_so_testing", so)

    node = make_quality_finalize_node(client)
    await node(_state(_config(finalists=["agentic"])))

    so.assert_not_called()
    failed = [c for c in client.upsert_quality_run.await_args_list
              if c.args[0]["status"] == "failed"]
    assert len(failed) == 1
    assert "oom" in failed[0].args[0]["error"]
