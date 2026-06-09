"""Unit tests for the adaptive agentic ceiling miss classifier.

A non-viable agentic phase is either a HARD miss (real ceiling — lock now) or a
MARGINAL miss (latency just over SLO, no request failures — worth one same-level
retry to rule out transient noise). See executor._classify_agentic_miss.
"""

from __future__ import annotations

from inference_agent.benchmark.runner import AgenticSLO
from inference_agent.nodes.executor import (
    _AGENTIC_MARGINAL_TOLERANCE,
    _classify_agentic_miss,
)
from inference_agent.models_pkg.domain import ConcurrencyResult, PercentileStats


_SLO = AgenticSLO(
    ttft_p95_ms=60000,
    tpot_p95_ms=100,
    e2e_p95_ms=4_320_000,
    session_error_rate=0.05,
)


def _result(ttft_p95: float, tpot_p95: float, error_rate: float = 0.0) -> ConcurrencyResult:
    return ConcurrencyResult(
        concurrency=1,
        prompt_length=83000,
        max_output_tokens=16000,
        ttft_ms=PercentileStats(p95=ttft_p95),
        tpot_ms=PercentileStats(p95=tpot_p95),
        e2e_latency_ms=PercentileStats(p95=0.0),
        error_rate=error_rate,
    )


def test_marginal_tpot_just_over_slo():
    # Real run c=20: tpot 105.9 vs 100 (6% over), ttft fine → marginal → retry.
    assert _classify_agentic_miss(_result(47847, 105.9), _SLO) == "marginal"


def test_hard_ttft_blown():
    # Real run c=24: ttft 82595 vs 60000 (38% over) → hard → lock immediately.
    assert _classify_agentic_miss(_result(82595, 103.8), _SLO) == "hard"


def test_errors_are_always_hard():
    # Request-level failures are never "noise", even with fine latency.
    assert _classify_agentic_miss(_result(50000, 90, error_rate=0.10), _SLO) == "hard"


def test_tolerance_boundary():
    # Exactly at the tolerance edge is marginal; just past it is hard.
    edge = _SLO.tpot_p95_ms * (1.0 + _AGENTIC_MARGINAL_TOLERANCE)
    assert _classify_agentic_miss(_result(50000, edge), _SLO) == "marginal"
    assert _classify_agentic_miss(_result(50000, edge + 0.1), _SLO) == "hard"


def test_worst_metric_decides():
    # tpot marginal but ttft hard → overall hard (worst metric wins).
    assert _classify_agentic_miss(_result(80000, 101), _SLO) == "hard"
