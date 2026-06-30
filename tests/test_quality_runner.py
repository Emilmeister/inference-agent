"""Tests for the quality suite subprocess runners (no real subprocess)."""

from __future__ import annotations

import json
import os

import pytest

from inference_agent.models_pkg.config import SoTestingConfig, TerminalBenchConfig
from inference_agent.quality import runner
from inference_agent.quality.runner import (
    _parse_harbor_results,
    _resolve_executable,
    _so_headline_score,
    run_so_testing,
    run_terminal_bench,
)


def test_resolve_executable_bare_name_uses_path():
    # No separator → returned as-is for PATH (system) lookup.
    assert _resolve_executable("python", "/repo") == "python"
    assert _resolve_executable("harbor", None) == "harbor"


def test_resolve_executable_relative_resolves_against_cwd():
    assert _resolve_executable(".venv/bin/python", "/repo") == "/repo/.venv/bin/python"
    assert _resolve_executable("./bin/harbor", "/repo") == "/repo/bin/harbor"


def test_resolve_executable_absolute_kept():
    assert _resolve_executable("/usr/bin/python3", "/repo") == "/usr/bin/python3"


def test_so_headline_score_means_available_composites():
    report = {
        "suites": {
            "structured_output": {"composite_score": 80.0},
            "tool_calling": {"composite_score": 90.0},
            "streaming": {"composite_score": None},  # excluded
        }
    }
    assert _so_headline_score(report) == 85.0


def test_so_headline_score_none_when_no_scores():
    assert _so_headline_score({"suites": {}}) is None


@pytest.mark.asyncio
async def test_run_so_testing_parses_report(monkeypatch):
    report = {
        "report_version": "1.0",
        "suites": {"structured_output": {"composite_score": 88.0}},
    }

    async def fake_subprocess(cmd, *, cwd, timeout_sec, env=None):
        out_path = cmd[cmd.index("--json") + 1]
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh)
        return 0, "", ""

    monkeypatch.setattr(runner, "_run_subprocess", fake_subprocess)
    result = await run_so_testing(
        SoTestingConfig(), base_url="http://x/v1", model="m"
    )
    assert result.suite == "so_testing"
    assert result.status == "done"
    assert result.score == 88.0
    assert result.suite_version == "1.0"


@pytest.mark.asyncio
async def test_run_so_testing_failure_returns_failed(monkeypatch):
    async def fake_subprocess(cmd, *, cwd, timeout_sec, env=None):
        return 1, "", "boom"

    monkeypatch.setattr(runner, "_run_subprocess", fake_subprocess)
    result = await run_so_testing(SoTestingConfig(), base_url="http://x/v1", model="m")
    assert result.status == "failed"
    assert "boom" in result.error


def test_parse_harbor_results_finds_accuracy(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps({"accuracy": 0.42, "n_resolved": 21, "n_tasks": 50}),
        encoding="utf-8",
    )
    accuracy, summary = _parse_harbor_results(str(tmp_path))
    assert accuracy == 0.42
    assert summary["parsed"] is True
    assert summary["n_resolved"] == 21


def test_parse_harbor_results_missing(tmp_path):
    accuracy, summary = _parse_harbor_results(str(tmp_path))
    assert accuracy is None
    assert summary["parsed"] is False


@pytest.mark.asyncio
async def test_run_terminal_bench_parses_jobs_dir(monkeypatch, tmp_path):
    cfg = TerminalBenchConfig(enabled=True, cwd=str(tmp_path), jobs_dir="jobs")
    fp = "abc123"
    jobs = tmp_path / "jobs" / fp
    jobs.mkdir(parents=True)
    (jobs / "results.json").write_text(json.dumps({"accuracy": 0.5}), encoding="utf-8")

    async def fake_subprocess(cmd, *, cwd, timeout_sec, env=None):
        assert f"openai/m" in cmd
        return 0, "harbor done", ""

    monkeypatch.setattr(runner, "_run_subprocess", fake_subprocess)
    result = await run_terminal_bench(cfg, base_url="http://x/v1", model="m", fingerprint=fp)
    assert result.suite == "terminal_bench"
    assert result.status == "done"
    assert result.score == 0.5
