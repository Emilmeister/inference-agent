"""Tests for the derived max_viable_agentic_concurrency metric (and friends)."""

from inference_agent.models import BenchmarkConfig, ConcurrencyResult, PercentileStats
from inference_agent.nodes.executor import _aggregate_benchmark


def _agentic(
    concurrency: int,
    ttft_p95: float,
    e2e_p95: float,
    throughput: float = 100.0,
    error_rate: float = 0.0,
) -> ConcurrencyResult:
    return ConcurrencyResult(
        concurrency=concurrency,
        prompt_length=64_000,
        max_output_tokens=16_384,
        num_requests=concurrency * 4,
        workload_id="agentic_long_context",
        phase_id=f"agentic_c{concurrency}_p64000",
        output_tokens_per_sec=throughput,
        ttft_ms=PercentileStats(mean=ttft_p95 * 0.8, p95=ttft_p95),
        e2e_latency_ms=PercentileStats(mean=e2e_p95 * 0.8, p95=e2e_p95),
        errors=int(round(error_rate * concurrency * 4)),
        error_rate=error_rate,
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
        agentic_ttft_p95_slo_ms=10_000.0,
        agentic_e2e_p95_slo_ms=200_000.0,   # explicit, not auto
        agentic_session_timeout_sec=300,
        **overrides,
    )


class TestMaxViableAgenticConcurrency:
    def test_all_levels_pass_means_ceiling_hit(self):
        results = [
            _baseline_anchor(),
            _agentic(concurrency=4, ttft_p95=2_000, e2e_p95=60_000),
            _agentic(concurrency=8, ttft_p95=4_000, e2e_p95=80_000),
            _agentic(concurrency=16, ttft_p95=8_000, e2e_p95=100_000),
        ]
        out = _aggregate_benchmark(results, {}, {}, _cfg())
        assert out.max_viable_agentic_concurrency == 16
        assert out.agentic_concurrency_ceiling_hit is True

    def test_only_lower_levels_pass_no_ceiling(self):
        # 16 violates TTFT SLO (15s > 10s)
        results = [
            _baseline_anchor(),
            _agentic(concurrency=4, ttft_p95=2_000, e2e_p95=60_000),
            _agentic(concurrency=8, ttft_p95=4_000, e2e_p95=80_000),
            _agentic(concurrency=16, ttft_p95=15_000, e2e_p95=100_000),
        ]
        out = _aggregate_benchmark(results, {}, {}, _cfg())
        assert out.max_viable_agentic_concurrency == 8
        assert out.agentic_concurrency_ceiling_hit is False

    def test_no_level_passes_means_zero(self):
        # All TTFTs above 10s SLO.
        results = [
            _baseline_anchor(),
            _agentic(concurrency=4, ttft_p95=11_000, e2e_p95=60_000),
            _agentic(concurrency=8, ttft_p95=12_000, e2e_p95=80_000),
            _agentic(concurrency=16, ttft_p95=15_000, e2e_p95=100_000),
        ]
        out = _aggregate_benchmark(results, {}, {}, _cfg())
        assert out.max_viable_agentic_concurrency == 0
        assert out.agentic_concurrency_ceiling_hit is False

    def test_high_error_rate_disqualifies_phase(self):
        # 16 has fine TTFT but error_rate above threshold.
        results = [
            _baseline_anchor(),
            _agentic(concurrency=4, ttft_p95=2_000, e2e_p95=60_000),
            _agentic(concurrency=8, ttft_p95=4_000, e2e_p95=80_000),
            _agentic(concurrency=16, ttft_p95=5_000, e2e_p95=90_000, error_rate=0.5),
        ]
        out = _aggregate_benchmark(results, {}, {}, _cfg())
        assert out.max_viable_agentic_concurrency == 8

    def test_e2e_slo_can_disqualify_independently(self):
        results = [
            _baseline_anchor(),
            _agentic(concurrency=4, ttft_p95=2_000, e2e_p95=60_000),
            _agentic(concurrency=8, ttft_p95=4_000, e2e_p95=210_000),  # > 200_000 SLO
        ]
        out = _aggregate_benchmark(results, {}, {}, _cfg())
        assert out.max_viable_agentic_concurrency == 4

    def test_no_agentic_phases_returns_zero(self):
        out = _aggregate_benchmark([_baseline_anchor()], {}, {}, _cfg())
        assert out.max_viable_agentic_concurrency == 0
        assert out.agentic_saturation_concurrency == 0
        assert out.agentic_peak_output_tokens_per_sec == 0.0
        assert out.agentic_concurrency_ceiling_hit is False

    def test_no_benchmark_config_returns_zero(self):
        # Existing call sites without benchmark_config (3-arg) keep working.
        results = [
            _agentic(concurrency=8, ttft_p95=4_000, e2e_p95=80_000),
        ]
        out = _aggregate_benchmark(results, {}, {})
        assert out.max_viable_agentic_concurrency == 0


class TestSaturationConcurrency:
    def test_saturation_reflects_peak_throughput_phase(self):
        results = [
            _baseline_anchor(),
            _agentic(concurrency=4, ttft_p95=2_000, e2e_p95=60_000, throughput=200),
            _agentic(concurrency=8, ttft_p95=4_000, e2e_p95=80_000, throughput=350),
            _agentic(concurrency=16, ttft_p95=8_000, e2e_p95=100_000, throughput=300),
        ]
        out = _aggregate_benchmark(results, {}, {}, _cfg())
        assert out.agentic_saturation_concurrency == 8
        assert out.agentic_peak_output_tokens_per_sec == 350.0


class TestE2ESloAutoFromSessionTimeout:
    def test_e2e_slo_auto_uses_80pct_of_session_timeout(self):
        # session_timeout_sec=300 → auto e2e_slo = 240_000ms.
        cfg = BenchmarkConfig(
            phase_error_rate_threshold=0.1,
            agentic_ttft_p95_slo_ms=10_000.0,
            agentic_e2e_p95_slo_ms=0.0,           # 0 → auto
            agentic_session_timeout_sec=300,
        )
        results = [
            _baseline_anchor(),
            _agentic(concurrency=4, ttft_p95=2_000, e2e_p95=200_000),  # < 240_000 → ok
            _agentic(concurrency=8, ttft_p95=4_000, e2e_p95=250_000),  # > 240_000 → fail
        ]
        out = _aggregate_benchmark(results, {}, {}, cfg)
        assert out.max_viable_agentic_concurrency == 4
