"""Tests for the derived max_viable_agentic_concurrency metric.

The SLO gate moved into the runner (it owns the percentiles and decides
`viable` per phase). The aggregator just reads `result.viable` and finds
the max-c phase that passed. These tests exercise the aggregator alone,
so they set `viable` directly per phase to model the runner's verdict.
"""

from inference_agent.models import BenchmarkConfig, ConcurrencyResult, PercentileStats
from inference_agent.nodes.executor import _aggregate_benchmark


def _agentic(
    concurrency: int,
    ttft_p95: float,
    e2e_p95: float,
    *,
    viable: bool,
    tpot_p95: float = 30.0,
    throughput: float = 100.0,
    error_rate: float = 0.0,
    slo_violations: list[str] | None = None,
) -> ConcurrencyResult:
    return ConcurrencyResult(
        concurrency=concurrency,
        prompt_length=24_000,
        max_output_tokens=2_400,
        num_requests=concurrency * 4,
        workload_id="agentic_long_context",
        phase_id=f"agentic_c{concurrency}_p24000",
        output_tokens_per_sec=throughput,
        ttft_ms=PercentileStats(mean=ttft_p95 * 0.8, p95=ttft_p95),
        tpot_ms=PercentileStats(mean=tpot_p95 * 0.8, p95=tpot_p95),
        e2e_latency_ms=PercentileStats(mean=e2e_p95 * 0.8, p95=e2e_p95),
        errors=int(round(error_rate * concurrency * 4)),
        error_rate=error_rate,
        viable=viable,
        slo_violations=slo_violations or [],
    )


def _baseline_anchor() -> ConcurrencyResult:
    """Non-agentic c=1 anchor so the aggregator's main path stays sensible."""
    return ConcurrencyResult(
        concurrency=1,
        prompt_length=512,
        max_output_tokens=256,
        num_requests=10,
        workload_id="agent_short",
        output_tokens_per_sec=50.0,
        ttft_ms=PercentileStats(mean=40, p95=50),
    )


def _cfg(**overrides) -> BenchmarkConfig:
    return BenchmarkConfig(
        phase_error_rate_threshold=0.1,
        agentic_ttft_p95_slo_ms=3_000.0,
        agentic_tpot_p95_slo_ms=60.0,
        agentic_session_error_rate_slo=0.05,
        agentic_e2e_p95_slo_ms=200_000.0,
        agentic_session_timeout_sec=300,
        **overrides,
    )


class TestMaxViableAgenticConcurrency:
    def test_all_levels_pass_means_ceiling_hit(self):
        results = [
            _baseline_anchor(),
            _agentic(concurrency=4, ttft_p95=2_000, e2e_p95=60_000, viable=True),
            _agentic(concurrency=8, ttft_p95=2_500, e2e_p95=80_000, viable=True),
            _agentic(concurrency=16, ttft_p95=2_900, e2e_p95=100_000, viable=True),
        ]
        out = _aggregate_benchmark(results, {}, {}, _cfg())
        assert out.max_viable_agentic_concurrency == 16
        assert out.agentic_concurrency_ceiling_hit is True

    def test_only_lower_levels_pass_no_ceiling(self):
        results = [
            _baseline_anchor(),
            _agentic(concurrency=4, ttft_p95=2_000, e2e_p95=60_000, viable=True),
            _agentic(concurrency=8, ttft_p95=2_500, e2e_p95=80_000, viable=True),
            _agentic(
                concurrency=16, ttft_p95=4_500, e2e_p95=100_000,
                viable=False, slo_violations=["ttft_p95=4500ms > slo=3000ms"],
            ),
        ]
        out = _aggregate_benchmark(results, {}, {}, _cfg())
        assert out.max_viable_agentic_concurrency == 8
        assert out.agentic_concurrency_ceiling_hit is False

    def test_no_level_passes_means_zero(self):
        results = [
            _baseline_anchor(),
            _agentic(concurrency=4, ttft_p95=4_000, e2e_p95=60_000, viable=False),
            _agentic(concurrency=8, ttft_p95=5_000, e2e_p95=80_000, viable=False),
            _agentic(concurrency=16, ttft_p95=6_000, e2e_p95=100_000, viable=False),
        ]
        out = _aggregate_benchmark(results, {}, {}, _cfg())
        assert out.max_viable_agentic_concurrency == 0
        assert out.agentic_concurrency_ceiling_hit is False

    def test_high_error_rate_disqualifies_phase(self):
        # c=16 has fine percentiles but the runner marked viable=False
        # because session error_rate broke SLO.
        results = [
            _baseline_anchor(),
            _agentic(concurrency=4, ttft_p95=2_000, e2e_p95=60_000, viable=True),
            _agentic(concurrency=8, ttft_p95=2_500, e2e_p95=80_000, viable=True),
            _agentic(
                concurrency=16, ttft_p95=2_800, e2e_p95=90_000,
                error_rate=0.5, viable=False,
                slo_violations=["error_rate=50% > slo=5%"],
            ),
        ]
        out = _aggregate_benchmark(results, {}, {}, _cfg())
        assert out.max_viable_agentic_concurrency == 8

    def test_no_agentic_phases_returns_zero(self):
        out = _aggregate_benchmark([_baseline_anchor()], {}, {}, _cfg())
        assert out.max_viable_agentic_concurrency == 0
        assert out.agentic_saturation_concurrency == 0
        assert out.agentic_peak_output_tokens_per_sec == 0.0
        assert out.agentic_concurrency_ceiling_hit is False

    def test_max_viable_phase_exposes_tpot_and_ttft(self):
        """The aggregator surfaces tpot/ttft AT the max-viable phase so the
        analyzer can tie-break two configs with the same max_viable_c."""
        results = [
            _baseline_anchor(),
            _agentic(
                concurrency=8, ttft_p95=1_800, tpot_p95=42, e2e_p95=60_000,
                throughput=300, viable=True,
            ),
            _agentic(
                concurrency=16, ttft_p95=2_800, tpot_p95=55, e2e_p95=90_000,
                throughput=500, viable=True,
            ),
        ]
        out = _aggregate_benchmark(results, {}, {}, _cfg())
        assert out.max_viable_agentic_concurrency == 16
        # tpot/ttft come from the c=16 phase (the headline concurrency).
        assert out.agentic_tpot_p95_ms == 55
        assert out.agentic_ttft_p95_ms == 2_800


class TestSaturationConcurrency:
    def test_saturation_reflects_peak_throughput_phase(self):
        results = [
            _baseline_anchor(),
            _agentic(concurrency=4, ttft_p95=2_000, e2e_p95=60_000, throughput=200, viable=True),
            _agentic(concurrency=8, ttft_p95=2_500, e2e_p95=80_000, throughput=350, viable=True),
            _agentic(concurrency=16, ttft_p95=2_900, e2e_p95=100_000, throughput=300, viable=True),
        ]
        out = _aggregate_benchmark(results, {}, {}, _cfg())
        assert out.agentic_saturation_concurrency == 8
        assert out.agentic_peak_output_tokens_per_sec == 350.0

    def test_non_viable_excluded_from_peak(self):
        """A phase with high raw throughput but viable=False should NOT win the
        saturation crown — that would credit a config we're explicitly not
        going to ship."""
        results = [
            _baseline_anchor(),
            _agentic(concurrency=4, ttft_p95=2_000, e2e_p95=60_000, throughput=200, viable=True),
            _agentic(concurrency=8, ttft_p95=2_500, e2e_p95=80_000, throughput=350, viable=True),
            _agentic(
                concurrency=16, ttft_p95=4_500, e2e_p95=100_000,
                throughput=900, viable=False,
                slo_violations=["ttft_p95=4500ms > slo=3000ms"],
            ),
        ]
        out = _aggregate_benchmark(results, {}, {}, _cfg())
        assert out.agentic_saturation_concurrency == 8
        assert out.agentic_peak_output_tokens_per_sec == 350.0
