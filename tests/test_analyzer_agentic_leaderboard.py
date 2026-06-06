"""Tests for the analyzer's agentic leaderboard + Pareto front.

The primary front lives in (max_viable_agentic_c ↑, agentic_tpot_p95 ↓)
space. The leaderboard tie-break order:
  1. max_viable_agentic_concurrency DESC
  2. agentic_tpot_p95 ASC
  3. agentic_peak_throughput DESC
"""

from inference_agent.models import (
    EngineType,
    ExperimentStatus,
    ExperimentSummary,
)
from inference_agent.nodes.analyzer import (
    _compute_agentic_pareto_front,
    _is_agentic_eligible,
)


def _summary(
    exp_id: str,
    *,
    agentic_c: int,
    tpot: float = 30.0,
    agentic_throughput: float = 200.0,
    status: ExperimentStatus = ExperimentStatus.SUCCESS,
    correctness: bool = True,
    engine: EngineType = EngineType.VLLM,
) -> ExperimentSummary:
    return ExperimentSummary(
        experiment_id=exp_id,
        engine=engine,
        status=status,
        correctness_gate_passed=correctness,
        agentic_max_viable_concurrency=agentic_c,
        agentic_tpot_p95=tpot,
        agentic_peak_throughput=agentic_throughput,
    )


class TestAgenticEligibility:
    def test_success_with_max_viable_c_eligible(self):
        s = _summary("a", agentic_c=16)
        assert _is_agentic_eligible(s) is True

    def test_zero_max_viable_c_not_eligible(self):
        """An exp that never sustained even c=8 under SLO is NOT a candidate
        for the agentic leaderboard, even if it ran clean otherwise."""
        s = _summary("a", agentic_c=0)
        assert _is_agentic_eligible(s) is False

    def test_correctness_failed_not_eligible(self):
        s = _summary("a", agentic_c=16, correctness=False)
        assert _is_agentic_eligible(s) is False

    def test_failed_status_not_eligible(self):
        s = _summary("a", agentic_c=16, status=ExperimentStatus.FAILED)
        assert _is_agentic_eligible(s) is False


class TestAgenticParetoFront:
    def test_empty_history(self):
        assert _compute_agentic_pareto_front([]) == []

    def test_single_eligible_on_front(self):
        s = _summary("a", agentic_c=16, tpot=30)
        front = _compute_agentic_pareto_front([s])
        assert len(front) == 1
        assert front[0].config_id == "a"
        assert front[0].agentic_max_viable_c == 16
        assert front[0].agentic_tpot_p95 == 30

    def test_dominated_point_excluded(self):
        """b has both fewer agents AND worse tpot → dominated by a."""
        history = [
            _summary("a", agentic_c=24, tpot=35),
            _summary("b", agentic_c=16, tpot=40),
        ]
        front = _compute_agentic_pareto_front(history)
        assert {p.config_id for p in front} == {"a"}

    def test_both_pareto_optimal(self):
        """a has more agents, b has snappier tpot — both stay on the front."""
        history = [
            _summary("a", agentic_c=24, tpot=50),
            _summary("b", agentic_c=8, tpot=20),
        ]
        front = _compute_agentic_pareto_front(history)
        assert {p.config_id for p in front} == {"a", "b"}

    def test_zero_agentic_c_excluded(self):
        history = [
            _summary("a", agentic_c=0, tpot=10),  # ineligible
            _summary("b", agentic_c=16, tpot=40),
        ]
        front = _compute_agentic_pareto_front(history)
        assert {p.config_id for p in front} == {"b"}

    def test_correctness_failed_excluded(self):
        history = [
            _summary("a", agentic_c=24, tpot=20, correctness=False),
            _summary("b", agentic_c=16, tpot=40),
        ]
        front = _compute_agentic_pareto_front(history)
        assert {p.config_id for p in front} == {"b"}
