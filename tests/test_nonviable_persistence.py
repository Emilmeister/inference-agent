"""NOT_VIABLE agentic phases are persisted in concurrency_results with full
metrics, but must not leak into the 'viable' views.

Since the executor now appends a non-viable agentic ConcurrencyResult to
benchmark.concurrency_results (so its measured metrics reach the DB and the
dashboard), every "viable" projection has to filter on r.viable. These tests
pin that:
  - the aggregator credits only viable phases (max_viable / peak),
  - ExperimentSummary.from_result splits viable vs ceiling concurrencies.
"""

from __future__ import annotations

from inference_agent.models import (
    BenchmarkConfig,
    BenchmarkResult,
    CeilingProbeInfo,
    ConcurrencyResult,
    EngineType,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    GPUInfo,
    HardwareProfile,
    PercentileStats,
    SmokeTestResult,
)
from inference_agent.nodes.executor import _aggregate_benchmark


def _agentic(concurrency: int, *, viable: bool, throughput: float = 100.0) -> ConcurrencyResult:
    return ConcurrencyResult(
        concurrency=concurrency,
        prompt_length=24_000,
        max_output_tokens=2_400,
        num_requests=concurrency * 4,
        workload_id="agentic_long_context",
        phase_id=f"agentic_c{concurrency}",
        output_tokens_per_sec=throughput,
        ttft_ms=PercentileStats(mean=2000, p95=2500 if viable else 9000),
        tpot_ms=PercentileStats(mean=30, p95=40),
        e2e_latency_ms=PercentileStats(mean=50_000, p95=90_000),
        viable=viable,
        slo_violations=[] if viable else ["ttft_p95=9000ms > slo=3000ms"],
    )


def _cfg() -> BenchmarkConfig:
    return BenchmarkConfig(agentic_ttft_p95_slo_ms=3_000.0, agentic_session_timeout_sec=300)


def test_nonviable_phase_in_results_does_not_inflate_max_viable():
    # c=32 is non-viable but PRESENT in concurrency_results (persisted) and has
    # the highest throughput — it must not win peak or raise max_viable.
    results = [
        _agentic(8, viable=True, throughput=80.0),
        _agentic(16, viable=True, throughput=120.0),
        _agentic(32, viable=False, throughput=200.0),
    ]
    out = _aggregate_benchmark(results, {}, {}, _cfg())
    assert out.max_viable_agentic_concurrency == 16
    # peak agentic throughput credited only to viable phases (120, not 200).
    assert out.agentic_peak_output_tokens_per_sec == 120.0
    # all three phases are still persisted for the dashboard.
    assert {r.concurrency for r in out.concurrency_results} == {8, 16, 32}
    assert sum(1 for r in out.concurrency_results if not r.viable) == 1


def _result_with(results: list[ConcurrencyResult], ceiling_c: list[int]) -> ExperimentResult:
    from inference_agent.models import ExperimentSummary  # local to avoid top clutter

    bench = _aggregate_benchmark(results, {}, {}, _cfg())
    hw = HardwareProfile(
        gpus=[GPUInfo(index=0, name="t", vram_total_mb=80000, vram_free_mb=70000)],
        gpu_count=1, nvlink_available=False,
        model_name="test/model", model_max_context=8192,
    )
    res = ExperimentResult(
        experiment_id="x",
        engine=EngineType.VLLM,
        model="test/model",
        hardware=hw,
        config=ExperimentConfig(engine=EngineType.VLLM, max_model_len=4096),
        status=ExperimentStatus.SUCCESS,
        smoke_tests=SmokeTestResult(),
        benchmark=bench,
        ceiling_probe_phases=[
            CeilingProbeInfo(
                phase_id=f"agentic_c{c}", workload_id="agentic_long_context",
                concurrency=c, prompt_length=24_000, error_rate=0.0, errors=0,
                reason="ttft_p95=9000ms > slo=3000ms",
            )
            for c in ceiling_c
        ],
    )
    return res, ExperimentSummary.from_result(res)


def test_summary_splits_viable_and_ceiling_concurrencies():
    results = [
        _agentic(8, viable=True),
        _agentic(16, viable=True),
        _agentic(32, viable=False),  # persisted but a ceiling level
    ]
    _, summary = _result_with(results, ceiling_c=[32])
    # The non-viable c=32 (now in concurrency_results) must be excluded from
    # the viable set and surface only as a ceiling concurrency.
    assert summary.agentic_concurrencies_viable == [8, 16]
    assert summary.agentic_concurrencies_ceiling == [32]
    assert summary.agentic_concurrencies_probed == [8, 16, 32]
