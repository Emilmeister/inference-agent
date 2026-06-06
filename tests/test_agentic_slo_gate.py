"""Tests for the agentic SLO gate (runner side).

Two units under test:
  * `build_shared_prefix` — deterministic so every session in a phase sees the
    exact same tokens (→ prefix-cache reuse).
  * `_evaluate_agentic_slo` — pure SLO check; returns the list of violations
    (empty means viable). The runner uses this to set ConcurrencyResult.viable.
"""

from inference_agent.benchmark.runner import (
    AgenticSLO,
    _evaluate_agentic_slo,
    build_shared_prefix,
)
from inference_agent.models import BenchmarkConfig


class TestBuildSharedPrefix:
    def test_deterministic_for_same_seed(self):
        """Two calls with the same seed produce identical bytes — that is the
        contract sessions rely on for prefix-cache KV reuse."""
        a = build_shared_prefix(2_000, seed=42)
        b = build_shared_prefix(2_000, seed=42)
        assert a == b
        assert len(a) > 100  # sanity check non-empty

    def test_different_seeds_diverge(self):
        a = build_shared_prefix(2_000, seed=1)
        b = build_shared_prefix(2_000, seed=2)
        assert a != b

    def test_zero_tokens_returns_empty(self):
        assert build_shared_prefix(0, seed=0) == ""


class TestAgenticSLOFromConfig:
    def test_explicit_e2e_used_as_is(self):
        cfg = BenchmarkConfig(
            agentic_ttft_p95_slo_ms=3_000,
            agentic_tpot_p95_slo_ms=60,
            agentic_session_error_rate_slo=0.05,
            agentic_e2e_p95_slo_ms=120_000,
            agentic_session_timeout_sec=300,
        )
        slo = AgenticSLO.from_config(cfg)
        assert slo.e2e_p95_ms == 120_000
        assert slo.ttft_p95_ms == 3_000
        assert slo.tpot_p95_ms == 60
        assert slo.session_error_rate == 0.05

    def test_e2e_zero_auto_uses_80pct_of_session_timeout(self):
        cfg = BenchmarkConfig(
            agentic_e2e_p95_slo_ms=0,           # 0 → auto
            agentic_session_timeout_sec=600,
        )
        slo = AgenticSLO.from_config(cfg)
        # 0.8 * 600s * 1000 = 480_000 ms
        assert slo.e2e_p95_ms == 480_000.0


class TestEvaluateAgenticSlo:
    def _slo(self) -> AgenticSLO:
        return AgenticSLO(
            ttft_p95_ms=3_000,
            tpot_p95_ms=60,
            e2e_p95_ms=240_000,
            session_error_rate=0.05,
        )

    def test_all_within_slo_returns_empty(self):
        violations = _evaluate_agentic_slo(
            ttft_p95=2_800, tpot_p95=55, e2e_p95=200_000, error_rate=0.02,
            slo=self._slo(),
        )
        assert violations == []

    def test_ttft_breach(self):
        violations = _evaluate_agentic_slo(
            ttft_p95=3_500, tpot_p95=40, e2e_p95=100_000, error_rate=0.0,
            slo=self._slo(),
        )
        assert any("ttft" in v for v in violations)
        assert len(violations) == 1

    def test_tpot_breach(self):
        violations = _evaluate_agentic_slo(
            ttft_p95=2_000, tpot_p95=75, e2e_p95=100_000, error_rate=0.0,
            slo=self._slo(),
        )
        assert any("tpot" in v for v in violations)

    def test_error_rate_breach(self):
        violations = _evaluate_agentic_slo(
            ttft_p95=2_000, tpot_p95=40, e2e_p95=100_000, error_rate=0.10,
            slo=self._slo(),
        )
        assert any("error_rate" in v for v in violations)

    def test_multiple_breaches_all_reported(self):
        violations = _evaluate_agentic_slo(
            ttft_p95=3_500, tpot_p95=75, e2e_p95=100_000, error_rate=0.10,
            slo=self._slo(),
        )
        # Each axis surfaces its own reason — analyzer can show all of them.
        assert len(violations) == 3

    def test_zero_percentiles_do_not_trip(self):
        """A phase that produced zero successful samples has p95=0 for ttft
        and tpot. Those must not be treated as 'oh nice, well under SLO' —
        but they ALSO should not trip the gate on their own, since the
        error_rate gate will fire instead with the real cause."""
        violations = _evaluate_agentic_slo(
            ttft_p95=0, tpot_p95=0, e2e_p95=0, error_rate=1.0,
            slo=self._slo(),
        )
        # Only the error_rate gate fires.
        assert violations == ["error_rate=100.0% > slo=5%"]
