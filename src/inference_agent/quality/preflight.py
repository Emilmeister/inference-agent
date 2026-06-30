"""Fast-fail preflight for the quality suites.

When `quality.enabled`, this verifies — at agent startup, BEFORE the
(potentially hours-long) optimization loop — that every enabled suite is
actually runnable: the interpreter/harbor binary resolves, its `cwd` exists,
and a quick launch (import the so-testing module / `harbor --help`) succeeds.
A misconfiguration fails loudly here instead of after the search converges and
a finalist container has already been relaunched.
"""

from __future__ import annotations

import logging
import os
import shutil

from inference_agent.models_pkg.config import (
    QualityConfig,
    SoTestingConfig,
    TerminalBenchConfig,
)
from inference_agent.quality.runner import _resolve_executable, _run_subprocess

logger = logging.getLogger(__name__)

_LAUNCH_TIMEOUT_SEC = 60


class QualityPreflightError(RuntimeError):
    """Raised when an enabled quality suite is not found / cannot launch."""


def _executable_problem(resolved: str, original: str) -> str | None:
    """Return a problem string if the executable can't be found, else None."""
    has_sep = os.sep in original or (os.altsep is not None and os.altsep in original)
    if has_sep:
        if not os.path.isfile(resolved):
            return f"not found at {resolved}"
        if not os.access(resolved, os.X_OK):
            return f"not executable: {resolved}"
        return None
    if shutil.which(resolved) is None:
        return f"'{resolved}' not found on PATH"
    return None


def _cwd_problem(cwd: str | None) -> str | None:
    if cwd is None:
        return None
    expanded = os.path.expanduser(cwd)
    if not os.path.isdir(expanded):
        return f"cwd does not exist: {expanded}"
    return None


async def _check_so_testing(cfg: SoTestingConfig) -> list[str]:
    errors: list[str] = []
    exe = _resolve_executable(cfg.interpreter, cfg.cwd)
    problem = _executable_problem(exe, cfg.interpreter)
    if problem:
        errors.append(f"so-testing interpreter {problem}")
    cwd_problem = _cwd_problem(cfg.cwd)
    if cwd_problem:
        errors.append(f"so-testing {cwd_problem}")
    if errors:
        return errors  # don't try to launch a broken interpreter/cwd

    rc, _out, err = await _run_subprocess(
        [exe, "-c", f"import importlib; importlib.import_module({cfg.module!r})"],
        cwd=os.path.expanduser(cfg.cwd) if cfg.cwd else None,
        timeout_sec=_LAUNCH_TIMEOUT_SEC,
    )
    if rc != 0:
        errors.append(
            f"so-testing cannot launch ({exe} -m {cfg.module}): "
            f"{err[-300:] or 'module import failed — wrong venv/cwd?'}"
        )
    return errors


async def _check_terminal_bench(cfg: TerminalBenchConfig) -> list[str]:
    errors: list[str] = []
    exe = _resolve_executable(cfg.harbor_bin, cfg.cwd)
    problem = _executable_problem(exe, cfg.harbor_bin)
    if problem:
        errors.append(f"terminal-bench harbor {problem}")
    cwd_problem = _cwd_problem(cfg.cwd)
    if cwd_problem:
        errors.append(f"terminal-bench {cwd_problem}")
    if errors:
        return errors

    rc, _out, err = await _run_subprocess(
        [exe, "--help"],
        cwd=os.path.expanduser(cfg.cwd) if cfg.cwd else None,
        timeout_sec=_LAUNCH_TIMEOUT_SEC,
    )
    if rc != 0:
        errors.append(
            f"terminal-bench harbor cannot launch ({exe} --help rc={rc}): {err[-300:]}"
        )
    return errors


async def preflight_quality(qcfg: QualityConfig) -> None:
    """Validate enabled quality suites are runnable; raise on the first run."""
    if not qcfg.enabled:
        return
    if not (qcfg.so_testing.enabled or qcfg.terminal_bench.enabled):
        return

    errors: list[str] = []
    if qcfg.so_testing.enabled:
        errors.extend(await _check_so_testing(qcfg.so_testing))
    if qcfg.terminal_bench.enabled:
        errors.extend(await _check_terminal_bench(qcfg.terminal_bench))

    if errors:
        raise QualityPreflightError(
            "Quality validation is enabled but its suites are not runnable. "
            "Fix config.quality (interpreter / harbor_bin / cwd) or disable the "
            "suite:\n  - " + "\n  - ".join(errors)
        )
    logger.info(
        "Quality preflight passed (so_testing=%s, terminal_bench=%s)",
        qcfg.so_testing.enabled, qcfg.terminal_bench.enabled,
    )
