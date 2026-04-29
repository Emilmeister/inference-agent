"""Tests for benchmark result aggregation."""

from inference_agent.models import ConcurrencyResult, PercentileStats
from inference_agent.nodes.executor import _aggregate_benchmark


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
