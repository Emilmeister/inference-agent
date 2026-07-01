"""Tests for the quality preflight (fast-fail when suites aren't runnable)."""

from __future__ import annotations

import sys

import pytest

from inference_agent.models_pkg.config import (
    QualityConfig,
    SoTestingConfig,
    TerminalBenchConfig,
)
from inference_agent.quality import preflight
from inference_agent.quality.preflight import QualityPreflightError, preflight_quality


def _qcfg(**kw) -> QualityConfig:
    base = dict(
        enabled=True,
        so_testing=SoTestingConfig(enabled=False),
        terminal_bench=TerminalBenchConfig(enabled=False),
    )
    base.update(kw)
    return QualityConfig(**base)


@pytest.mark.asyncio
async def test_disabled_quality_skips_preflight(monkeypatch):
    monkeypatch.setattr(preflight, "_run_subprocess", _should_not_run)
    await preflight_quality(QualityConfig(enabled=False))  # no raise, no launch


@pytest.mark.asyncio
async def test_missing_interpreter_path_fails(monkeypatch):
    monkeypatch.setattr(preflight, "_run_subprocess", _should_not_run)
    qcfg = _qcfg(so_testing=SoTestingConfig(enabled=True, interpreter="/no/such/python3"))
    with pytest.raises(QualityPreflightError, match="not found"):
        await preflight_quality(qcfg)


@pytest.mark.asyncio
async def test_missing_harbor_on_path_fails(monkeypatch):
    monkeypatch.setattr(preflight, "_run_subprocess", _should_not_run)
    qcfg = _qcfg(terminal_bench=TerminalBenchConfig(
        enabled=True, harbor_bin="definitely-not-a-real-binary-xyzzy",
    ))
    with pytest.raises(QualityPreflightError, match="not found on PATH"):
        await preflight_quality(qcfg)


@pytest.mark.asyncio
async def test_missing_cwd_fails(monkeypatch):
    monkeypatch.setattr(preflight, "_run_subprocess", _should_not_run)
    qcfg = _qcfg(so_testing=SoTestingConfig(
        enabled=True, interpreter=sys.executable, cwd="/no/such/dir",
    ))
    with pytest.raises(QualityPreflightError, match="cwd does not exist"):
        await preflight_quality(qcfg)


@pytest.mark.asyncio
async def test_passes_when_launchable(monkeypatch):
    async def fake_ok(cmd, *, cwd, timeout_sec, env=None):
        return 0, "", ""

    monkeypatch.setattr(preflight, "_run_subprocess", fake_ok)
    qcfg = _qcfg(so_testing=SoTestingConfig(enabled=True, interpreter=sys.executable))
    await preflight_quality(qcfg)  # no raise


@pytest.mark.asyncio
async def test_terminal_bench_missing_docker_fails(monkeypatch):
    async def ok(cmd, *, cwd, timeout_sec, env=None):
        return 0, "", ""

    monkeypatch.setenv("PATH", "/nonexistent-xyz")   # no docker anywhere
    monkeypatch.setattr(preflight, "_run_subprocess", ok)
    qcfg = _qcfg(terminal_bench=TerminalBenchConfig(
        enabled=True, harbor_bin=sys.executable,     # real file → passes exe check
    ))
    with pytest.raises(QualityPreflightError, match="needs `docker`"):
        await preflight_quality(qcfg)


@pytest.mark.asyncio
async def test_terminal_bench_docker_found_via_path_prepend(monkeypatch, tmp_path):
    async def ok(cmd, *, cwd, timeout_sec, env=None):
        return 0, "", ""

    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\n")
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", "/nonexistent-xyz")
    monkeypatch.setattr(preflight, "_run_subprocess", ok)
    qcfg = _qcfg(terminal_bench=TerminalBenchConfig(
        enabled=True, harbor_bin=sys.executable, path_prepend=[str(tmp_path)],
    ))
    await preflight_quality(qcfg)   # docker resolvable via path_prepend → no raise


@pytest.mark.asyncio
async def test_launch_failure_fails(monkeypatch):
    async def fake_fail(cmd, *, cwd, timeout_sec, env=None):
        return 1, "", "ModuleNotFoundError: llm_provider_benchmark"

    monkeypatch.setattr(preflight, "_run_subprocess", fake_fail)
    qcfg = _qcfg(so_testing=SoTestingConfig(enabled=True, interpreter=sys.executable))
    with pytest.raises(QualityPreflightError, match="cannot launch"):
        await preflight_quality(qcfg)


async def _should_not_run(*args, **kwargs):  # pragma: no cover - guard
    raise AssertionError("_run_subprocess should not be called in this path")


# ── discovery degrade-vs-abort behavior ─────────────────────────────────────


@pytest.mark.asyncio
async def test_discovery_degrades_disables_quality_and_continues(monkeypatch):
    from inference_agent.models import AgentConfig
    from inference_agent.nodes import discovery

    cfg = AgentConfig()
    cfg.quality.enabled = True
    cfg.quality.preflight_strict = False

    async def boom(_qcfg):
        raise QualityPreflightError("not runnable")

    monkeypatch.setattr(discovery, "preflight_quality", boom)
    await discovery._apply_quality_preflight(cfg)  # must NOT raise
    assert cfg.quality.enabled is False  # disabled, optimization continues


@pytest.mark.asyncio
async def test_discovery_strict_aborts(monkeypatch):
    from inference_agent.models import AgentConfig
    from inference_agent.nodes import discovery

    cfg = AgentConfig()
    cfg.quality.enabled = True
    cfg.quality.preflight_strict = True

    async def boom(_qcfg):
        raise QualityPreflightError("not runnable")

    monkeypatch.setattr(discovery, "preflight_quality", boom)
    with pytest.raises(QualityPreflightError):
        await discovery._apply_quality_preflight(cfg)
    assert cfg.quality.enabled is True  # left intact for the abort


@pytest.mark.asyncio
async def test_discovery_noop_when_quality_disabled(monkeypatch):
    from inference_agent.models import AgentConfig
    from inference_agent.nodes import discovery

    cfg = AgentConfig()  # quality disabled by default
    monkeypatch.setattr(discovery, "preflight_quality", _should_not_run)
    await discovery._apply_quality_preflight(cfg)
