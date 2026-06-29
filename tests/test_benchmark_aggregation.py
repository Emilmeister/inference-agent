"""Tests for benchmark result aggregation."""

from inference_agent.models import ConcurrencyResult, PercentileStats
from inference_agent.nodes.executor import (
    _aggregate_benchmark,
    _empirical_prefix_hit_rate,
)


def _make_conc_result(
    concurrency: int = 1,
    throughput: float = 100.0,
    ttft_p95: float = 50.0,
    tpot_p95: float = 10.0,
    ttft_cv: float = 0.0,
    e2e_cv: float = 0.0,
    workload_id: str = "agent_short",
) -> ConcurrencyResult:
    return ConcurrencyResult(
        concurrency=concurrency,
        prompt_length=512,
        max_output_tokens=256,
        num_requests=100,
        workload_id=workload_id,
        output_tokens_per_sec=throughput,
        requests_per_sec=throughput / 256,
        total_tokens_per_sec=throughput * 2,
        ttft_ms=PercentileStats(mean=ttft_p95 * 0.8, p95=ttft_p95, cv=ttft_cv),
        tpot_ms=PercentileStats(mean=tpot_p95 * 0.8, p95=tpot_p95),
        e2e_latency_ms=PercentileStats(mean=100.0, p95=200.0, cv=e2e_cv),
    )


class TestAggregateBenchmark:
    def test_empty_results(self):
        result = _aggregate_benchmark([], {}, {})
        assert result.peak_output_tokens_per_sec == 0.0

    def test_peak_throughput(self):
        results = [
            _make_conc_result(concurrency=1, throughput=100),
            _make_conc_result(concurrency=64, throughput=500),
            _make_conc_result(concurrency=128, throughput=450),
        ]
        result = _aggregate_benchmark(results, {}, {})
        assert result.peak_output_tokens_per_sec == 500.0

    def test_low_concurrency_latency(self):
        """Low-concurrency latency uses median (not min) of c=1 phases."""
        results = [
            _make_conc_result(concurrency=1, ttft_p95=30),
            _make_conc_result(concurrency=1, ttft_p95=50),
            _make_conc_result(concurrency=64, ttft_p95=200),
        ]
        result = _aggregate_benchmark(results, {}, {})
        # median of [30, 50] = 40.0
        assert result.low_concurrency_ttft_p95_ms == 40.0

    def test_no_low_concurrency(self):
        results = [_make_conc_result(concurrency=64)]
        result = _aggregate_benchmark(results, {}, {})
        assert result.low_concurrency_ttft_p95_ms == 0.0

    def test_gpu_metrics(self):
        gpu_agg = {
            0: {"util_avg": 85.0, "mem_peak": 70000, "power_avg": 300, "temp_max": 75},
            1: {"util_avg": 80.0, "mem_peak": 68000, "power_avg": 290, "temp_max": 72},
        }
        results = [_make_conc_result()]
        result = _aggregate_benchmark(results, gpu_agg, {})
        assert len(result.gpu_utilization_percent) == 2
        assert result.gpu_utilization_percent[0] == 85.0

    def test_kv_cache_metrics(self):
        kv = {"kv_cache_usage_percent": 75.0, "prefix_cache_hit_rate": 0.5}
        results = [_make_conc_result()]
        result = _aggregate_benchmark(results, {}, kv)
        assert result.kv_cache_usage_percent == 75.0
        assert result.prefix_cache_hit_rate == 0.5

    def test_peak_throughput_cv_from_winning_phase(self):
        """peak_throughput_e2e_cv should come from the phase that won peak."""
        results = [
            _make_conc_result(concurrency=64, throughput=300, e2e_cv=0.1),
            _make_conc_result(concurrency=128, throughput=500, e2e_cv=0.4),  # winner
            _make_conc_result(concurrency=64, throughput=200, e2e_cv=0.2),
        ]
        result = _aggregate_benchmark(results, {}, {})
        assert result.peak_output_tokens_per_sec == 500.0
        assert result.peak_throughput_e2e_cv == 0.4

    def test_low_concurrency_ttft_cv_median(self):
        """low_concurrency_ttft_cv = median of TTFT cv across c=1 agent_short phases."""
        results = [
            _make_conc_result(concurrency=1, ttft_p95=30, ttft_cv=0.1),
            _make_conc_result(concurrency=1, ttft_p95=50, ttft_cv=0.3),
            _make_conc_result(concurrency=1, ttft_p95=40, ttft_cv=0.5),
            _make_conc_result(concurrency=64, ttft_p95=200, ttft_cv=0.9),  # ignored
        ]
        result = _aggregate_benchmark(results, {}, {})
        # median of [0.1, 0.3, 0.5] = 0.3
        assert result.low_concurrency_ttft_cv == 0.3

    def test_cv_zero_when_no_low_concurrency(self):
        results = [_make_conc_result(concurrency=64, ttft_cv=0.5)]
        result = _aggregate_benchmark(results, {}, {})
        assert result.low_concurrency_ttft_cv == 0.0

    def test_prometheus_hit_rate_preferred_over_empirical(self):
        """A non-zero Prometheus hit rate wins; the empirical fallback is ignored."""
        results = [
            ConcurrencyResult(
                concurrency=8, prompt_length=4096, max_output_tokens=256,
                workload_id="agentic_long_context",
                total_input_tokens=1000, total_cached_tokens=900,
            ),
        ]
        kv = {"prefix_cache_hit_rate": 0.2}
        result = _aggregate_benchmark(results, {}, kv)
        assert result.prefix_cache_hit_rate == 0.2  # Prometheus value, not 0.9

    def test_empirical_hit_rate_fallback_when_prometheus_absent(self):
        """When Prometheus reports nothing (0), fall back to Σcached/Σinput (token-weighted)."""
        results = [
            ConcurrencyResult(
                concurrency=8, prompt_length=4096, max_output_tokens=256,
                workload_id="agentic_long_context",
                total_input_tokens=1000, total_cached_tokens=800,
            ),
            ConcurrencyResult(
                concurrency=1, prompt_length=512, max_output_tokens=256,
                workload_id="agent_short",
                total_input_tokens=200, total_cached_tokens=0,
            ),
        ]
        result = _aggregate_benchmark(results, {}, {})
        # Σcached / Σinput = 800 / 1200
        assert result.prefix_cache_hit_rate == 800.0 / 1200.0


class TestEmpiricalPrefixHitRate:
    def test_no_input_is_zero(self):
        results = [ConcurrencyResult(concurrency=1, prompt_length=0, max_output_tokens=0)]
        assert _empirical_prefix_hit_rate(results) == 0.0

    def test_no_cached_is_zero(self):
        results = [
            ConcurrencyResult(
                concurrency=1, prompt_length=512, max_output_tokens=256,
                total_input_tokens=500, total_cached_tokens=0,
            ),
        ]
        assert _empirical_prefix_hit_rate(results) == 0.0

    def test_token_weighted_pooled_ratio(self):
        # Two phases pooled by raw token count, NOT by rate: phase A is low-volume
        # but fully cached, phase B is high-volume uncached. Token-weighted =
        # (1000+0)/(1000+9000) = 0.1, regardless of per-phase rates.
        results = [
            ConcurrencyResult(
                concurrency=8, prompt_length=4096, max_output_tokens=256,
                total_input_tokens=1000, total_cached_tokens=1000,
            ),
            ConcurrencyResult(
                concurrency=128, prompt_length=512, max_output_tokens=256,
                total_input_tokens=9000, total_cached_tokens=0,
            ),
        ]
        assert _empirical_prefix_hit_rate(results) == 0.1

    def test_bounded_to_one(self):
        # Defensive: cached should never exceed input, but clamp if it does.
        results = [
            ConcurrencyResult(
                concurrency=1, prompt_length=512, max_output_tokens=256,
                total_input_tokens=100, total_cached_tokens=150,
            ),
        ]
        assert _empirical_prefix_hit_rate(results) == 1.0
