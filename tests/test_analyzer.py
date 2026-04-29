"""Tests for analyzer pure logic — Pareto front, scoring, plateau detection."""

from __future__ import annotations

import pytest

from inference_agent.models import (
    AgentConfig,
    AnalyzerOutput,
    BenchmarkResult,
    EngineType,
    ExperimentConfig,
    ExperimentResult,
    ExperimentScores,
    ExperimentStatus,
    ExperimentSummary,
    GPUInfo,
    HardwareProfile,
    ParetoPoint,
)
from inference_agent.nodes import analyzer as analyzer_module
from inference_agent.nodes.analyzer import (
    _check_plateau,
    _compute_pareto_front,
    _compute_scores,
    analyzer_node,
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
    @staticmethod
    def _mock_result(
        throughput: float,
        ttft_p95: float,
        peak_cv: float = 0.0,
        ttft_cv: float = 0.0,
    ):
        from unittest.mock import MagicMock

        result = MagicMock()
        result.experiment_id = "test"
        result.benchmark.peak_output_tokens_per_sec = throughput
        result.benchmark.low_concurrency_ttft_p95_ms = ttft_p95
        result.benchmark.peak_throughput_e2e_cv = peak_cv
        result.benchmark.low_concurrency_ttft_cv = ttft_cv
        return result

    def test_basic_scores(self):
        result = self._mock_result(throughput=150.0, ttft_p95=40.0)
        scores = _compute_scores(result, best_throughput=200, best_latency=50, pareto=[])
        assert scores.throughput_score == 0.75  # 150/200, no derate (cv=0)
        # latency_score = best_latency / lat = 50/40 = 1.25, capped at 1.0
        assert scores.latency_score == 1.0
        assert scores.is_pareto_optimal is False

    def test_pareto_optimal_flag(self):
        result = self._mock_result(throughput=100.0, ttft_p95=50.0)
        pareto = [ParetoPoint(config_id="test", engine=EngineType.VLLM, throughput=100, ttft_p95=50)]
        scores = _compute_scores(result, best_throughput=100, best_latency=50, pareto=pareto)
        assert scores.is_pareto_optimal is True

    def test_zero_best_values(self):
        result = self._mock_result(throughput=0.0, ttft_p95=0.0)
        scores = _compute_scores(result, best_throughput=0, best_latency=0, pareto=[])
        assert scores.throughput_score == 0.0
        assert scores.latency_score == 0.0

    def test_high_cv_derates_throughput(self):
        # Same raw throughput, different cv → noisy config gets a lower score.
        clean = self._mock_result(throughput=150.0, ttft_p95=40.0, peak_cv=0.0)
        noisy = self._mock_result(throughput=150.0, ttft_p95=40.0, peak_cv=0.5)
        clean_scores = _compute_scores(clean, best_throughput=200, best_latency=50, pareto=[])
        noisy_scores = _compute_scores(noisy, best_throughput=200, best_latency=50, pareto=[])
        # Derate factor: 1 - 0.3 * 0.5 = 0.85
        assert clean_scores.throughput_score == pytest.approx(0.75)
        assert noisy_scores.throughput_score == pytest.approx(0.75 * 0.85)
        assert noisy_scores.throughput_score < clean_scores.throughput_score

    def test_cv_derate_is_capped(self):
        # cv >= NOISE_CV_CAP (1.0) is treated as 1.0 — extreme dispersion
        # never zeroes out the score, just maxes the derate.
        result = self._mock_result(throughput=200.0, ttft_p95=50.0, peak_cv=5.0)
        scores = _compute_scores(result, best_throughput=200, best_latency=50, pareto=[])
        # 1 - 0.3 * 1.0 = 0.7 → throughput_score = 1.0 * 0.7 = 0.7
        assert scores.throughput_score == pytest.approx(0.7)

    def test_ttft_cv_derates_latency_score(self):
        result = self._mock_result(throughput=100.0, ttft_p95=50.0, ttft_cv=0.4)
        scores = _compute_scores(result, best_throughput=100, best_latency=50, pareto=[])
        # latency_score before derate = 50/50 = 1.0; derate = 1 - 0.3*0.4 = 0.88
        assert scores.latency_score == pytest.approx(0.88)

    def test_pareto_flag_not_affected_by_cv(self):
        # Pareto front is the hard mathematical filter — derate must not flip it.
        result = self._mock_result(throughput=150.0, ttft_p95=40.0, peak_cv=0.9, ttft_cv=0.9)
        pareto = [ParetoPoint(config_id="test", engine=EngineType.VLLM, throughput=150, ttft_p95=40)]
        scores = _compute_scores(result, best_throughput=200, best_latency=50, pareto=pareto)
        assert scores.is_pareto_optimal is True


# ─── Analyzer node — split between session and loaded_top_history ──────────


def _make_result(
    exp_id: str,
    throughput: float = 120.0,
    ttft_p95: float = 40.0,
    correctness_gate_passed: bool = True,
    status: ExperimentStatus = ExperimentStatus.SUCCESS,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=exp_id,
        engine=EngineType.VLLM,
        model="test/model",
        hardware=HardwareProfile(
            gpus=[GPUInfo(index=0, name="A100", vram_total_mb=81920, vram_free_mb=80000)],
            gpu_count=1,
            nvlink_available=False,
            model_name="test/model",
        ),
        config=ExperimentConfig(engine=EngineType.VLLM, max_model_len=4096),
        status=status,
        correctness_gate_passed=correctness_gate_passed,
        benchmark=BenchmarkResult(
            peak_output_tokens_per_sec=throughput,
            low_concurrency_ttft_p95_ms=ttft_p95,
        ),
    )


class _StubLLM:
    """Replace `structured_output` so analyzer_node tests don't hit any LLM."""

    def __init__(self) -> None:
        self.output = AnalyzerOutput(
            commentary="test analysis",
            classification="none",
            decision="continue",
            next_goal="explore",
            planner_hint="",
        )

    async def __call__(self, prompt, schema, llm_cfg):  # noqa: ARG002
        return self.output


@pytest.fixture
def stub_llm(monkeypatch):
    stub = _StubLLM()
    monkeypatch.setattr(analyzer_module, "structured_output", stub)
    return stub


class TestAnalyzerNodeWithLoadedHistory:
    @pytest.mark.asyncio
    async def test_plateau_not_triggered_by_loaded_tops(self, stub_llm):
        """Loaded tops alone must not trigger plateau on iteration 1."""
        config = AgentConfig()
        config.experiments.plateau_window = 5
        # Loaded top is much stronger than the new experiment — would trigger
        # plateau if it counted in the plateau window.
        loaded = [_make_summary(f"prev_{i}", throughput=1000.0, ttft_p95=10.0) for i in range(5)]
        result = _make_result("new_1", throughput=120.0, ttft_p95=40.0)

        out = await analyzer_node({
            "config": config,
            "current_result": result,
            "experiment_history": [],
            "loaded_top_history": loaded,
            "experiments_count": 0,
        })

        assert out["status"] == "running"
        assert out["stop_reason"] is None

    @pytest.mark.asyncio
    async def test_best_throughput_tracks_session_only(self, stub_llm):
        """state['best_throughput'] reflects session, not loaded tops."""
        config = AgentConfig()
        loaded = [_make_summary("prev_1", throughput=5000.0, ttft_p95=5.0)]
        result = _make_result("new_1", throughput=120.0, ttft_p95=40.0)

        out = await analyzer_node({
            "config": config,
            "current_result": result,
            "experiment_history": [],
            "loaded_top_history": loaded,
            "experiments_count": 0,
        })

        assert out["best_throughput"] == 120.0
        assert out["best_throughput_config_id"] == "new_1"

    @pytest.mark.asyncio
    async def test_pareto_combines_loaded_and_session(self, stub_llm):
        """Pareto front merges loaded tops with current session."""
        config = AgentConfig()
        # Loaded top dominates throughput axis; current dominates latency axis.
        loaded = [_make_summary("prev_tp", throughput=500.0, ttft_p95=200.0)]
        result = _make_result("new_lat", throughput=100.0, ttft_p95=10.0)

        out = await analyzer_node({
            "config": config,
            "current_result": result,
            "experiment_history": [],
            "loaded_top_history": loaded,
            "experiments_count": 0,
        })

        ids = {p.config_id for p in out["pareto_front"]}
        assert ids == {"prev_tp", "new_lat"}

    @pytest.mark.asyncio
    async def test_scores_normalized_against_union_best(self, stub_llm):
        """Throughput score is normalized against the cross-run best."""
        config = AgentConfig()
        loaded = [_make_summary("prev_top", throughput=400.0, ttft_p95=20.0)]
        result = _make_result("new_1", throughput=100.0, ttft_p95=80.0)

        out = await analyzer_node({
            "config": config,
            "current_result": result,
            "experiment_history": [],
            "loaded_top_history": loaded,
            "experiments_count": 0,
        })

        enriched = out["current_result"]
        # 100 / 400 = 0.25 — would be 1.0 if we ignored loaded tops
        assert enriched.scores.throughput_score == pytest.approx(0.25)
