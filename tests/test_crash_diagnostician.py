"""Tests for the crash_diagnostician node.

On a container crash it makes ONE dedicated LLM call over the full logs and
rewrites the planner/analyzer-facing error with a distilled root cause + fix.
Everything else passes through, and an LLM failure never gates the run.
"""

from __future__ import annotations

import pytest

import inference_agent.nodes.crash_diagnostician as cd
from inference_agent.models import (
    AgentConfig,
    CrashDiagnosis,
    EngineType,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    ExperimentStatus,
    GPUInfo,
    HardwareProfile,
)
from inference_agent.nodes.crash_diagnostician import _load_logs, crash_diagnostician_node


def _hw() -> HardwareProfile:
    return HardwareProfile(
        gpus=[GPUInfo(index=0, name="H100", vram_total_mb=80000, vram_free_mb=79000)],
        gpu_count=8, nvlink_available=True,
        model_name="m", model_max_context=8192,
    )


def _result(status: ExperimentStatus, errors: list[ExperimentError]) -> ExperimentResult:
    return ExperimentResult(
        experiment_id="x", engine=EngineType.VLLM, model="m", hardware=_hw(),
        config=ExperimentConfig(engine=EngineType.VLLM),
        status=status, error="; ".join(e.message for e in errors), errors=errors,
        container_command="vllm serve --gpu-memory-utilization 0.95 ...",
        failure_classification="healthcheck_timeout",
    )


def _crash_result() -> ExperimentResult:
    return _result(
        ExperimentStatus.FAILED,
        [ExperimentError(
            stage="healthcheck",
            message="Engine did not become healthy (reason=fatal_in_logs)",
            details={"logs": "CUDA out of memory. Tried to allocate 2GB", "log_path": None},
        )],
    )


def _state(result: ExperimentResult) -> dict:
    return {"current_result": result, "config": AgentConfig()}


@pytest.fixture
def fake_llm(monkeypatch):
    """Patch structured_output to return a canned CrashDiagnosis; record the prompt."""
    calls = {}

    async def _fake(prompt, output_model, llm_config):
        calls["prompt"] = prompt
        calls["model"] = output_model
        return CrashDiagnosis(
            summary="CUDA OOM during KV cache allocation",
            root_cause="gpu_memory_utilization 0.95 left no room for the KV cache",
            fix="Lower gpu_memory_utilization to ~0.85",
            config_fixable=True,
        )

    monkeypatch.setattr(cd, "structured_output", _fake)
    return calls


@pytest.mark.asyncio
async def test_diagnoses_container_crash(fake_llm):
    result = _crash_result()
    out = await crash_diagnostician_node(_state(result))
    assert "current_result" in out
    r = out["current_result"]
    assert r.crash_diagnosis is not None
    assert r.crash_diagnosis.config_fixable is True
    # planner/analyzer-facing error is now the distilled diagnosis, not raw stderr
    assert "CONTAINER CRASH" in r.error
    assert "Lower gpu_memory_utilization" in r.error
    assert "CUDA out of memory" not in r.error  # raw log not pushed to planner
    # the LLM did receive the full logs + launch command
    assert "CUDA out of memory" in fake_llm["prompt"]
    assert "gpu-memory-utilization 0.95" in fake_llm["prompt"]
    assert fake_llm["model"] is CrashDiagnosis


@pytest.mark.asyncio
async def test_success_result_passes_through(fake_llm):
    out = await crash_diagnostician_node(_state(_result(ExperimentStatus.SUCCESS, [])))
    assert out == {}
    assert "prompt" not in fake_llm  # LLM not called


@pytest.mark.asyncio
async def test_correctness_failure_not_diagnosed(fake_llm):
    # FAILED_CORRECTNESS: container is healthy, smoke tests failed — not a crash.
    out = await crash_diagnostician_node(
        _state(_result(ExperimentStatus.FAILED_CORRECTNESS, []))
    )
    assert out == {}
    assert "prompt" not in fake_llm


@pytest.mark.asyncio
async def test_crash_without_logs_skipped(fake_llm):
    # startup error with no captured logs (e.g. image pull) → nothing to diagnose
    result = _result(
        ExperimentStatus.FAILED,
        [ExperimentError(stage="startup", message="image pull failed",
                         details={"classification": "image_pull_failed"})],
    )
    out = await crash_diagnostician_node(_state(result))
    assert out == {}
    assert "prompt" not in fake_llm


@pytest.mark.asyncio
async def test_llm_failure_keeps_raw_error(monkeypatch):
    async def _boom(*a, **k):
        raise RuntimeError("LLM endpoint down")

    monkeypatch.setattr(cd, "structured_output", _boom)
    result = _crash_result()
    original = result.error
    out = await crash_diagnostician_node(_state(result))
    assert out == {}            # no state change
    assert result.error == original  # raw error untouched
    assert result.crash_diagnosis is None


def test_load_logs_prefers_file_and_truncates(tmp_path):
    big = "HEAD-MARKER\n" + ("x" * 100_000) + "\nTAIL-MARKER"
    p = tmp_path / "crash.log"
    p.write_text(big, encoding="utf-8")
    err = ExperimentError(stage="healthcheck", message="m",
                          details={"logs": "excerpt", "log_path": str(p)})
    logs = _load_logs(err)
    assert "HEAD-MARKER" in logs and "TAIL-MARKER" in logs  # both ends kept
    assert "truncated" in logs
    assert len(logs) < 60_000


def test_load_logs_falls_back_to_excerpt():
    err = ExperimentError(stage="healthcheck", message="m",
                          details={"logs": "just-the-excerpt", "log_path": "/no/such/file"})
    assert _load_logs(err) == "just-the-excerpt"
