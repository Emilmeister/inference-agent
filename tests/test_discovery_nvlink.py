"""Tests for _detect_nvlink retry/robustness.

NVLink presence is part of the exact HardwareProfile match key used for
history/baseline lookups, so a flaky `nvidia-smi topo -m` probe must NOT
silently flip nvlink_available to False on a transient failure.
"""

import subprocess

from inference_agent.nodes import discovery


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["nvidia-smi", "topo", "-m"], returncode=returncode, stdout=stdout, stderr=""
    )


def _patch_run(monkeypatch, side_effects):
    """side_effects: list of CompletedProcess or Exception, one per call."""
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        i = calls["n"]
        calls["n"] += 1
        eff = side_effects[i]
        if isinstance(eff, Exception):
            raise eff
        return eff

    monkeypatch.setattr(discovery.subprocess, "run", fake_run)
    monkeypatch.setattr(discovery.time, "sleep", lambda *_: None)
    return calls


def test_clean_success_with_nvlink(monkeypatch):
    calls = _patch_run(monkeypatch, [_completed("GPU0 NV12 ...")])
    assert discovery._detect_nvlink() is True
    assert calls["n"] == 1  # no retries on a successful probe


def test_clean_success_without_nvlink_is_legit_false(monkeypatch):
    # A probe that succeeds and shows no NV links is a real "no NVLink".
    calls = _patch_run(monkeypatch, [_completed("GPU0 X SYS PHB ...")])
    assert discovery._detect_nvlink() is False
    assert calls["n"] == 1


def test_transient_timeout_then_success_recovers(monkeypatch):
    calls = _patch_run(
        monkeypatch,
        [subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=20), _completed("NV12")],
    )
    assert discovery._detect_nvlink() is True
    assert calls["n"] == 2  # retried after the timeout


def test_nonzero_exit_treated_as_failed_not_no_nvlink(monkeypatch):
    # Non-zero exit (e.g. driver busy) must be retried, not read as "no NVLink".
    calls = _patch_run(
        monkeypatch,
        [_completed("", returncode=9), _completed("NV12")],
    )
    assert discovery._detect_nvlink() is True
    assert calls["n"] == 2


def test_all_attempts_fail_falls_back_to_false(monkeypatch):
    calls = _patch_run(
        monkeypatch,
        [subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=20)] * 3,
    )
    assert discovery._detect_nvlink(attempts=3) is False
    assert calls["n"] == 3  # exhausted all attempts, did not raise


def test_missing_nvidia_smi_short_circuits(monkeypatch):
    calls = _patch_run(monkeypatch, [FileNotFoundError()])
    assert discovery._detect_nvlink() is False
    assert calls["n"] == 1  # not transient — no retries wasted
