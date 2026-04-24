"""Tests for failure classification logic."""

from inference_agent.models import ExperimentError
from inference_agent.nodes.executor import _classify_failure


class TestClassifyFailure:
    def test_no_failure(self):
        assert _classify_failure([], [], True, False, False) is None

    def test_startup_crash(self):
        errors = [ExperimentError(
            stage="startup",
            message="Container start failed",
            details={"logs": "some generic error"},
        )]
        assert _classify_failure(errors, [], False, False, True) == "startup_crash"

    def test_startup_oom(self):
        errors = [ExperimentError(
            stage="startup",
            message="Container start failed",
            details={"logs": "CUDA out of memory. Tried to allocate 2.00 GiB"},
        )]
        assert _classify_failure(errors, [], False, False, True) == "oom"

    def test_healthcheck_timeout(self):
        errors = [ExperimentError(
            stage="healthcheck",
            message="Engine did not become healthy within 900s",
            details={"exit_code": 1, "time_elapsed_sec": 900},
        )]
        assert _classify_failure(errors, [], False, False, True) == "healthcheck_timeout"

    def test_healthcheck_oom_exit_137(self):
        errors = [ExperimentError(
            stage="healthcheck",
            message="Engine did not become healthy within 900s",
            details={"exit_code": 137},
        )]
        assert _classify_failure(errors, [], False, False, True) == "oom"

    def test_correctness_failure(self):
        assert _classify_failure([], [], False, False, False) == "correctness_failure"

    def test_runtime_crash_post_correctness(self):
        assert _classify_failure([], [], True, True, False) == "runtime_crash"

    def test_runtime_crash_container_died(self):
        assert _classify_failure([], [], True, False, True) == "runtime_crash"

    def test_benchmark_error(self):
        phase_errors = [ExperimentError(
            stage="benchmark_phase",
            message="Phase c64_p512 error_rate=0.50 exceeds threshold",
        )]
        assert _classify_failure([], phase_errors, True, False, False) == "benchmark_error"

    def test_startup_takes_priority(self):
        """Startup errors should be classified first, even if other errors exist."""
        startup = [ExperimentError(stage="startup", message="crash", details={"logs": "error"})]
        phase = [ExperimentError(stage="benchmark_phase", message="timeout")]
        assert _classify_failure(startup, phase, False, False, True) == "startup_crash"

    def test_oom_detection_case_insensitive(self):
        errors = [ExperimentError(
            stage="startup",
            message="Container start failed",
            details={"logs": "RuntimeError: CUDA error: out of memory"},
        )]
        # "cuda" is detected in logs
        assert _classify_failure(errors, [], False, False, True) == "oom"
