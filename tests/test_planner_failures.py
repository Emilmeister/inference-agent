"""Tests for planner failure pattern aggregation."""

from inference_agent.models import (
    EngineType,
    ExperimentStatus,
    ExperimentSummary,
)
from inference_agent.nodes.planner import _aggregate_failure_patterns


def _make_summary(
    exp_id: str,
    engine: EngineType = EngineType.VLLM,
    status: ExperimentStatus = ExperimentStatus.SUCCESS,
    error: str | None = None,
    failure_classification: str | None = None,
) -> ExperimentSummary:
    return ExperimentSummary(
        experiment_id=exp_id,
        engine=engine,
        status=status,
        error=error,
        failure_classification=failure_classification,
    )


class TestAggregateFailurePatterns:
    def test_empty_history(self):
        assert _aggregate_failure_patterns([]) == ""

    def test_no_failures(self):
        history = [_make_summary("a"), _make_summary("b")]
        assert _aggregate_failure_patterns(history) == ""

    def test_single_failure(self):
        history = [
            _make_summary("a", status=ExperimentStatus.FAILED,
                         failure_classification="oom", error="CUDA OOM"),
        ]
        result = _aggregate_failure_patterns(history)
        assert "vllm" in result
        assert "oom" in result
        assert "1x" in result

    def test_multiple_same_failure(self):
        history = [
            _make_summary("a", status=ExperimentStatus.FAILED,
                         failure_classification="oom", error="OOM 1"),
            _make_summary("b", status=ExperimentStatus.FAILED,
                         failure_classification="oom", error="OOM 2"),
        ]
        result = _aggregate_failure_patterns(history)
        assert "2x" in result

    def test_different_engines(self):
        history = [
            _make_summary("a", engine=EngineType.VLLM,
                         status=ExperimentStatus.FAILED,
                         failure_classification="startup_crash"),
            _make_summary("b", engine=EngineType.SGLANG,
                         status=ExperimentStatus.FAILED,
                         failure_classification="healthcheck_timeout"),
        ]
        result = _aggregate_failure_patterns(history)
        assert "vllm" in result
        assert "sglang" in result
        assert "startup_crash" in result
        assert "healthcheck_timeout" in result

    def test_correctness_failures(self):
        history = [
            _make_summary("a", engine=EngineType.SGLANG,
                         status=ExperimentStatus.FAILED_CORRECTNESS,
                         failure_classification="correctness_failure"),
            _make_summary("b", engine=EngineType.SGLANG,
                         status=ExperimentStatus.FAILED_CORRECTNESS,
                         failure_classification="correctness_failure"),
        ]
        result = _aggregate_failure_patterns(history)
        assert "correctness_failure" in result
        assert "sglang" in result

    def test_partial_counted(self):
        """PARTIAL status experiments are included in failure patterns."""
        history = [
            _make_summary("a", status=ExperimentStatus.PARTIAL,
                         failure_classification="benchmark_error"),
        ]
        result = _aggregate_failure_patterns(history)
        assert "benchmark_error" in result

    def test_success_not_counted(self):
        """SUCCESS experiments are not failures."""
        history = [
            _make_summary("a", status=ExperimentStatus.SUCCESS),
            _make_summary("b", status=ExperimentStatus.FAILED,
                         failure_classification="oom"),
        ]
        result = _aggregate_failure_patterns(history)
        # Only 1 failure, not 2
        assert "1x" in result

    def test_unknown_classification(self):
        """Failures without classification are grouped as 'unknown'."""
        history = [
            _make_summary("a", status=ExperimentStatus.FAILED,
                         failure_classification=None, error="something broke"),
        ]
        result = _aggregate_failure_patterns(history)
        assert "unknown" in result

    def test_error_message_included(self):
        history = [
            _make_summary("a", status=ExperimentStatus.FAILED,
                         failure_classification="oom",
                         error="CUDA error: out of memory allocating 32GB"),
        ]
        result = _aggregate_failure_patterns(history)
        assert "CUDA error" in result
