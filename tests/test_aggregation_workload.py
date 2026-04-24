"""Tests for workload-aware benchmark aggregation."""

from inference_agent.models import ConcurrencyResult, PercentileStats
from inference_agent.nodes.executor import _aggregate_benchmark


def _make_conc_result(
    concurrency: int = 1,
    throughput: float = 100.0,
    ttft_p95: float = 50.0,
    tpot_p95: float = 10.0,
    workload_id: str = "agent_short",
    phase_id: str = "",
    prompt_length: int = 512,
    errors: int = 0,
    error_rate: float = 0.0,
) -> ConcurrencyResult:
    return ConcurrencyResult(
        concurrency=concurrency,
        prompt_length=prompt_length,
        max_output_tokens=256,
        num_requests=100,
        workload_id=workload_id,
        phase_id=phase_id or f"c{concurrency}_p{prompt_length}",
        output_tokens_per_sec=throughput,
        requests_per_sec=throughput / 256,
        total_tokens_per_sec=throughput * 2,
        ttft_ms=PercentileStats(mean=ttft_p95 * 0.8, p95=ttft_p95),
        tpot_ms=PercentileStats(mean=tpot_p95 * 0.8, p95=tpot_p95),
        errors=errors,
        error_rate=error_rate,
    )


class TestWorkloadAwarePeakThroughput:
    def test_excludes_stress_from_peak(self):
        """Stress phases should NOT contribute to peak throughput."""
        results = [
            _make_conc_result(concurrency=64, throughput=400, workload_id="throughput"),
            _make_conc_result(concurrency=512, throughput=600, workload_id="stress"),
        ]
        result = _aggregate_benchmark(results, {}, {})
        assert result.peak_output_tokens_per_sec == 400.0

    def test_excludes_long_context_from_peak(self):
        """Long context phases should NOT contribute to peak throughput."""
        results = [
            _make_conc_result(concurrency=64, throughput=400, workload_id="throughput"),
            _make_conc_result(concurrency=1, throughput=500, workload_id="long_context"),
        ]
        result = _aggregate_benchmark(results, {}, {})
        assert result.peak_output_tokens_per_sec == 400.0

    def test_includes_agent_short_and_throughput(self):
        """Both agent_short and throughput workloads count for peak."""
        results = [
            _make_conc_result(concurrency=16, throughput=300, workload_id="agent_short"),
            _make_conc_result(concurrency=128, throughput=500, workload_id="throughput"),
        ]
        result = _aggregate_benchmark(results, {}, {})
        assert result.peak_output_tokens_per_sec == 500.0

    def test_fallback_if_only_stress(self):
        """If only stress results exist, use them (fallback)."""
        results = [
            _make_conc_result(concurrency=512, throughput=600, workload_id="stress"),
        ]
        result = _aggregate_benchmark(results, {}, {})
        assert result.peak_output_tokens_per_sec == 600.0

    def test_backward_compat_empty_workload(self):
        """Results without workload_id (backward compat) are included."""
        results = [
            _make_conc_result(concurrency=64, throughput=400, workload_id=""),
        ]
        result = _aggregate_benchmark(results, {}, {})
        assert result.peak_output_tokens_per_sec == 400.0


class TestWorkloadAwareLatency:
    def test_median_not_min(self):
        """Low-concurrency latency should be median, not min."""
        results = [
            _make_conc_result(concurrency=1, ttft_p95=20, workload_id="agent_short"),
            _make_conc_result(concurrency=1, ttft_p95=40, workload_id="agent_short"),
            _make_conc_result(concurrency=1, ttft_p95=100, workload_id="agent_short"),
        ]
        result = _aggregate_benchmark(results, {}, {})
        assert result.low_concurrency_ttft_p95_ms == 40.0  # median

    def test_excludes_long_context_from_latency(self):
        """Long context c=1 phases should NOT affect low_concurrency_ttft_p95."""
        results = [
            _make_conc_result(concurrency=1, ttft_p95=50, workload_id="agent_short"),
            _make_conc_result(concurrency=1, ttft_p95=500, workload_id="long_context"),
        ]
        result = _aggregate_benchmark(results, {}, {})
        assert result.low_concurrency_ttft_p95_ms == 50.0

    def test_no_c1_results(self):
        """If no c=1 results, latency is 0."""
        results = [
            _make_conc_result(concurrency=64, ttft_p95=100, workload_id="throughput"),
        ]
        result = _aggregate_benchmark(results, {}, {})
        assert result.low_concurrency_ttft_p95_ms == 0.0

    def test_fallback_to_any_c1(self):
        """If no agent_short c=1, fall back to any c=1."""
        results = [
            _make_conc_result(concurrency=1, ttft_p95=300, workload_id="long_context"),
        ]
        result = _aggregate_benchmark(results, {}, {})
        # Fallback: uses long_context c=1
        assert result.low_concurrency_ttft_p95_ms == 300.0

    def test_tpot_also_uses_median(self):
        """TPOT p95 should also use median of c=1 agent_short phases."""
        results = [
            _make_conc_result(concurrency=1, tpot_p95=5, workload_id="agent_short"),
            _make_conc_result(concurrency=1, tpot_p95=15, workload_id="agent_short"),
            _make_conc_result(concurrency=1, tpot_p95=25, workload_id="agent_short"),
        ]
        result = _aggregate_benchmark(results, {}, {})
        assert result.low_concurrency_tpot_p95_ms == 15.0

    def test_single_c1_result(self):
        """Single c=1 result: median is the value itself."""
        results = [
            _make_conc_result(concurrency=1, ttft_p95=42, workload_id="agent_short"),
        ]
        result = _aggregate_benchmark(results, {}, {})
        assert result.low_concurrency_ttft_p95_ms == 42.0


class TestErrorRateInResults:
    def test_error_rate_computed(self):
        r = ConcurrencyResult(
            concurrency=1,
            prompt_length=512,
            max_output_tokens=256,
            num_requests=100,
            errors=10,
            error_rate=0.1,
        )
        assert r.error_rate == 0.1

    def test_zero_errors(self):
        r = ConcurrencyResult(
            concurrency=1,
            prompt_length=512,
            max_output_tokens=256,
            num_requests=100,
            errors=0,
            error_rate=0.0,
        )
        assert r.error_rate == 0.0
