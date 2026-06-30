"""Subprocess runners for the external quality suites.

so-testing and terminal-bench live in their own repos/venvs; the agent shells
out to them against a finalist's OpenAI-compatible endpoint and parses the
result. Both return a uniform `SuiteResult` so the node persists them the same
way.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

from inference_agent.models_pkg.config import SoTestingConfig, TerminalBenchConfig

logger = logging.getLogger(__name__)


def _resolve_executable(exe: str, cwd: str | None) -> str:
    """Resolve an executable spec to something exec can find deterministically.

    - A bare command name (no path separator), e.g. ``python`` / ``harbor``,
      is left as-is → looked up on ``PATH`` (the system install).
    - A path (``~/...``, ``./...``, ``.venv/bin/python``, ``/abs/...``) has
      ``~`` expanded; a relative path is resolved against ``cwd`` (the suite's
      repo dir) so it works regardless of where the agent was launched, rather
      than relying on subprocess's ambiguous relative-exe behavior.
    """
    exe = os.path.expanduser(exe)
    has_sep = os.sep in exe or (os.altsep is not None and os.altsep in exe)
    if not has_sep:
        return exe  # bare name → PATH lookup (system)
    if os.path.isabs(exe):
        return exe
    base = os.path.expanduser(cwd) if cwd else os.getcwd()
    return os.path.abspath(os.path.join(base, exe))


@dataclass
class SuiteResult:
    suite: str                       # "so_testing" | "terminal_bench"
    status: str                      # "done" | "failed"
    suite_version: str = ""
    score: float | None = None       # headline (composite / accuracy)
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""


async def _run_subprocess(
    cmd: list[str],
    *,
    cwd: str | None,
    timeout_sec: int,
    env: dict[str, str] | None = None,
) -> tuple[int | None, str, str]:
    """Run a command, return (returncode, stdout_tail, stderr_tail).

    returncode is None on timeout (process killed). Output is tail-truncated so
    a chatty suite can't blow memory/logs.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        return None, "", f"timed out after {timeout_sec}s"
    tail = lambda b: b.decode(errors="replace")[-8000:]
    return proc.returncode, tail(stdout), tail(stderr)


# ── so-testing ──────────────────────────────────────────────────────────────


def _so_headline_score(report: dict[str, Any]) -> float | None:
    """Mean of the available per-suite composite scores (streaming has none)."""
    suites = report.get("suites") or {}
    scores = [
        s["composite_score"]
        for s in suites.values()
        if isinstance(s, dict) and s.get("composite_score") is not None
    ]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 2)


async def run_so_testing(
    cfg: SoTestingConfig,
    base_url: str,
    model: str,
) -> SuiteResult:
    """Run the so-testing CLI against `base_url` and parse its JSON report."""
    api_key = os.environ.get(cfg.api_key_env) or "EMPTY"
    out_path = tempfile.NamedTemporaryFile(
        prefix="so_testing_", suffix=".json", delete=False
    ).name
    cmd = [
        _resolve_executable(cfg.interpreter, cfg.cwd), "-m", cfg.module, "run",
        "--base-url", base_url,
        "--model", model,
        "--api-key", api_key,
        "--suite", ",".join(cfg.suites),
        "--runs", str(cfg.runs),
        "--temperature", str(cfg.temperature),
        "--max-tokens", str(cfg.max_tokens),
        "--json", out_path,
    ]
    logger.info("Running so-testing: %s", " ".join(cmd[:6]) + " ...")
    rc, _stdout, stderr = await _run_subprocess(
        cmd, cwd=os.path.expanduser(cfg.cwd) if cfg.cwd else None,
        timeout_sec=cfg.timeout_sec,
    )

    report: dict[str, Any] = {}
    try:
        with open(out_path, encoding="utf-8") as fh:
            report = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        if rc == 0:
            return SuiteResult(
                suite="so_testing", status="failed",
                error=f"so-testing exited 0 but report unreadable: {e}; stderr={stderr[-500:]}",
            )
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass

    if rc != 0:
        return SuiteResult(
            suite="so_testing", status="failed",
            data=report,
            error=f"so-testing exited rc={rc}: {stderr[-800:]}",
        )

    return SuiteResult(
        suite="so_testing",
        status="done",
        suite_version=str(report.get("report_version", "")),
        score=_so_headline_score(report),
        data=report,
    )


# ── terminal-bench (harbor) ─────────────────────────────────────────────────


def _parse_harbor_results(jobs_path: str) -> tuple[float | None, dict[str, Any]]:
    """Best-effort parse of a harbor jobs dir → (accuracy, summary).

    harbor (terminal-bench 2.x) writes a results JSON under the jobs dir. The
    exact filename/shape can vary by version, so we scan for a JSON file
    carrying an accuracy-like field and project a compact summary. Returns
    (None, {...}) when no recognizable results file is found — the caller
    still records the run, just without a headline score.

    NOTE: confirm the exact layout against a real jobs/<run-id> output and
    tighten this if needed (tracked in the design doc as the one open item).
    """
    accuracy_keys = ("accuracy", "resolved_rate", "pass_rate", "score")
    candidate: dict[str, Any] | None = None
    for root, _dirs, files in os.walk(jobs_path):
        for name in files:
            if not name.endswith(".json"):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, encoding="utf-8") as fh:
                    blob = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(blob, dict) and any(k in blob for k in accuracy_keys):
                candidate = blob
                # Prefer a top-level results/summary file over a per-task one.
                if name in ("results.json", "summary.json", "run_metadata.json"):
                    break
        if candidate is not None:
            break

    if candidate is None:
        return None, {"parsed": False, "note": "no harbor results json found"}

    accuracy = next(
        (float(candidate[k]) for k in accuracy_keys if isinstance(candidate.get(k), (int, float))),
        None,
    )
    summary = {
        "parsed": True,
        "accuracy": accuracy,
        "n_resolved": candidate.get("n_resolved") or candidate.get("resolved"),
        "n_tasks": candidate.get("n_tasks") or candidate.get("total"),
        "raw": candidate,
    }
    return accuracy, summary


async def run_terminal_bench(
    cfg: TerminalBenchConfig,
    base_url: str,
    model: str,
    fingerprint: str,
) -> SuiteResult:
    """Run terminal-bench via `harbor run` against `base_url`, parse jobs dir.

    Reproduces the operator's working invocation, parameterised per finalist.
    The jobs dir is namespaced by fingerprint so concurrent/repeat runs don't
    collide and the parser reads exactly this run's output.
    """
    jobs_subdir = os.path.join(cfg.jobs_dir, fingerprint)
    cmd = [
        _resolve_executable(cfg.harbor_bin, cfg.cwd), "run",
        "--dataset", cfg.dataset,
        "--agent", cfg.agent,
        "--model", f"openai/{model}",
        "--agent-kwarg", f"api_base={base_url}",
        "--agent-kwarg", f"temperature={cfg.temperature}",
        "--agent-kwarg", f"max_turns={cfg.max_turns}",
        "--n-concurrent", str(cfg.n_concurrent),
        "--agent-timeout-multiplier", str(cfg.agent_timeout_multiplier),
        "--jobs-dir", jobs_subdir,
        "-k", str(cfg.k),
    ]
    cwd = os.path.expanduser(cfg.cwd) if cfg.cwd else None
    logger.info("Running terminal-bench (harbor): model=%s jobs=%s", model, jobs_subdir)
    rc, stdout, stderr = await _run_subprocess(
        cmd, cwd=cwd, timeout_sec=cfg.timeout_sec
    )

    # jobs_subdir is relative to cfg.cwd when set.
    abs_jobs = (
        os.path.join(cwd, jobs_subdir)
        if cwd and not os.path.isabs(jobs_subdir)
        else jobs_subdir
    )
    accuracy, summary = _parse_harbor_results(abs_jobs)
    summary["jobs_dir"] = abs_jobs
    summary["stdout_tail"] = stdout[-2000:]

    if rc != 0:
        return SuiteResult(
            suite="terminal_bench", status="failed",
            suite_version=cfg.dataset, data=summary,
            error=f"harbor exited rc={rc}: {stderr[-800:]}",
        )

    return SuiteResult(
        suite="terminal_bench",
        status="done",
        suite_version=cfg.dataset,
        score=accuracy,
        data=summary,
    )
