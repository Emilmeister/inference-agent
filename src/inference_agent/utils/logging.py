"""Structured logging utilities — experiment context in log records."""

from __future__ import annotations

import contextvars
import logging
from typing import Any

# Context variables for structured logging
_experiment_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "experiment_id", default=""
)
_experiment_engine: contextvars.ContextVar[str] = contextvars.ContextVar(
    "experiment_engine", default=""
)
_experiment_phase: contextvars.ContextVar[str] = contextvars.ContextVar(
    "experiment_phase", default=""
)


def set_experiment_context(
    experiment_id: str = "",
    engine: str = "",
    phase: str = "",
) -> None:
    """Set experiment context for structured logging."""
    if experiment_id:
        _experiment_id.set(experiment_id)
    if engine:
        _experiment_engine.set(engine)
    if phase:
        _experiment_phase.set(phase)


def clear_experiment_context() -> None:
    """Clear experiment context."""
    _experiment_id.set("")
    _experiment_engine.set("")
    _experiment_phase.set("")


class ExperimentContextFilter(logging.Filter):
    """Adds experiment_id, engine, and phase to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.experiment_id = _experiment_id.get("")  # type: ignore[attr-defined]
        record.engine = _experiment_engine.get("")  # type: ignore[attr-defined]
        record.phase = _experiment_phase.get("")  # type: ignore[attr-defined]
        return True


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with experiment context support."""
    level = logging.DEBUG if verbose else logging.INFO

    # Format includes experiment context when available
    fmt = "%(asctime)s %(levelname)-5s [%(name)s]"
    fmt += " [%(experiment_id)s|%(engine)s]" if True else ""
    fmt += " %(message)s"

    handler = logging.StreamHandler()
    handler.addFilter(ExperimentContextFilter())
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(level)
    # Remove existing handlers to avoid duplicates
    root.handlers.clear()
    root.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in [
        "httpcore", "httpx", "openai", "urllib3", "docker",
        "asyncio", "huggingface_hub", "filelock",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
