"""Tests for analyzer pure logic — Pareto front, scoring, plateau detection."""

from inference_agent.models import (
    EngineType,
    ExperimentScores,
    ExperimentStatus,
    ExperimentSummary,
    ParetoPoint,
)
from inference_agent.nodes.analyzer import (
    _check_plateau,
    _compute_pareto_front,
    _compute_scores,
)


def _make_summary(
    exp_id: str,
    throughput: float = 100.0,
    ttft_p95: float = 50.0,
    status: ExperimentStatus = ExperimentStatus.SUCCESS,
    engine: EngineType = EngineType.VLLM,
    correctness_gate_passed: bool = True,
) -> ExperimentSummary:
    return ExperimentSummary(
        experiment_id=exp_id,
        engine=engine,
        status=status,
        peak_throughput=throughput,
        low_concurrency_ttft_p95=ttft_p95,
        correctness_gate_passed=correctness_gate_passed,
    )


class TestComputeParetoFront:
    def test_empty_history(self):
        assert _compute_pareto_front([]) == []

    def test_single_experiment(self):
        history = [_make_summary("a", throughput=100, ttft_p95=50)]
        pareto = _compute_pareto_front(history)
        assert len(pareto) == 1
        assert pareto[0].config_id == "a"

    def test_dominated_point_excluded(self):
        history = [
            _make_summary("a", throughput=200, ttft_p95=30),  # dominates b
            _make_summary("b", throughput=100, ttft_p95=50),  # dominated
        ]
        pareto = _compute_pareto_front(history)
        assert len(pareto) == 1
        assert pareto[0].config_id == "a"

    def test_pareto_optimal_pair(self):
        history = [
            _make_summary("a", throughput=200, ttft_p95=100),  # high tp, high lat
            _make_summary("b", throughput=100, ttft_p95=30),   # low tp, low lat
        ]
        pareto = _compute_pareto_front(history)
        assert len(pareto) == 2
        ids = {p.config_id for p in pareto}
        assert ids == {"a", "b"}

    def test_failed_experiments_excluded(self):
        history = [
            _make_summary("a", throughput=200, ttft_p95=30, status=ExperimentStatus.FAILED, correctness_gate_passed=False),
            _make_summary("b", throughput=100, ttft_p95=50),
        ]
        pareto = _compute_pareto_front(history)
        assert len(pareto) == 1
        assert pareto[0].config_id == "b"

    def test_correctness_failed_excluded(self):
        """Experiments that failed correctness gate are not eligible for Pareto."""
        history = [
            _make_summary("a", throughput=200, ttft_p95=30, correctness_gate_passed=False),
            _make_summary("b", throughput=100, ttft_p95=50, correctness_gate_passed=True),
        ]
        pareto = _compute_pareto_front(history)
        assert len(pareto) == 1
        assert pareto[0].config_id == "b"

    def test_zero_metrics_excluded(self):
        history = [
            _make_summary("a", throughput=0, ttft_p95=0),
            _make_summary("b", throughput=100, ttft_p95=50),
        ]
        pareto = _compute_pareto_front(history)
        assert len(pareto) == 1


class TestCheckPlateau:
    def test_not_enough_history(self):
        history = [_make_summary("a"), _make_summary("b")]
        assert _check_plateau(history, 100, 50, window=5, threshold=0.02) is False

    def test_no_plateau_with_improvement(self):
        # The plateau check compares recent experiments against the *best* values.
        # For this test: best_throughput=140, and exp_4 has throughput=140 which
        # matches best but doesn't EXCEED it, so no throughput improvement.
        # We need an experiment that actually beats the best.
        history = [
            _make_summary(f"exp_{i}", throughput=100 + i * 10) for i in range(5)
        ]
        # Set best_throughput to 130, so exp_4 (140) is an improvement
        assert _check_plateau(history, 130, 50, window=5, threshold=0.02) is False

    def test_plateau_detected(self):
        # All experiments have same throughput and latency — no improvement
        history = [_make_summary(f"exp_{i}", throughput=100, ttft_p95=50) for i in range(5)]
        assert _check_plateau(history, 200, 30, window=5, threshold=0.02) is True

    def test_latency_improvement_breaks_plateau(self):
        history = [_make_summary(f"exp_{i}", throughput=100, ttft_p95=50) for i in range(4)]
        # Last experiment has better latency than current best
        history.append(_make_summary("exp_4", throughput=100, ttft_p95=25))
        # best_latency=30, and exp_4 has 25 < 30 — improvement
        assert _check_plateau(history, 200, 30, window=5, threshold=0.02) is False


class TestComputeScores:
    def test_basic_scores(self):
        from unittest.mock import MagicMock

        result = MagicMock()
        result.experiment_id = "test"
        result.benchmark.peak_output_tokens_per_sec = 150.0
        result.benchmark.low_concurrency_ttft_p95_ms = 40.0

        scores = _compute_scores(result, best_throughput=200, best_latency=50, pareto=[])
        assert scores.throughput_score == 0.75  # 150/200
        # latency_score = best_latency / lat = 50/40 = 1.25, capped at 1.0
        assert scores.latency_score == 1.0
        assert scores.is_pareto_optimal is False

    def test_pareto_optimal_flag(self):
        from unittest.mock import MagicMock

        result = MagicMock()
        result.experiment_id = "test"
        result.benchmark.peak_output_tokens_per_sec = 100.0
        result.benchmark.low_concurrency_ttft_p95_ms = 50.0

        pareto = [ParetoPoint(config_id="test", engine=EngineType.VLLM, throughput=100, ttft_p95=50)]
        scores = _compute_scores(result, best_throughput=100, best_latency=50, pareto=pareto)
        assert scores.is_pareto_optimal is True

    def test_zero_best_values(self):
        from unittest.mock import MagicMock

        result = MagicMock()
        result.experiment_id = "test"
        result.benchmark.peak_output_tokens_per_sec = 0.0
        result.benchmark.low_concurrency_ttft_p95_ms = 0.0

        scores = _compute_scores(result, best_throughput=0, best_latency=0, pareto=[])
        assert scores.throughput_score == 0.0
        assert scores.latency_score == 0.0
