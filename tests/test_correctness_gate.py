"""Tests for correctness gate logic — smoke test gate, status classification, eligibility."""

from inference_agent.models import (
    EngineType,
    ExperimentStatus,
    ExperimentSummary,
    SmokeTestResult,
)
from inference_agent.nodes.analyzer import _is_eligible


class TestSmokeTestGate:
    """Test that gate_passed property works correctly."""

    def test_all_pass(self):
        r = SmokeTestResult(basic_chat=True, tool_calling=True, json_schema=True)
        assert r.gate_passed is True

    def test_basic_chat_fails(self):
        r = SmokeTestResult(basic_chat=False, tool_calling=True, json_schema=True)
        assert r.gate_passed is False

    def test_tool_calling_fails(self):
        r = SmokeTestResult(basic_chat=True, tool_calling=False, json_schema=True)
        assert r.gate_passed is False

    def test_json_schema_fails(self):
        r = SmokeTestResult(basic_chat=True, tool_calling=True, json_schema=False)
        assert r.gate_passed is False

    def test_all_fail(self):
        r = SmokeTestResult()
        assert r.gate_passed is False

    def test_json_mode_not_required(self):
        """json_mode and tool_required are NOT required for the gate."""
        r = SmokeTestResult(
            basic_chat=True, tool_calling=True, json_schema=True,
            json_mode=False, tool_required=False,
        )
        assert r.gate_passed is True

    def test_gate_with_details(self):
        r = SmokeTestResult(
            basic_chat=True, basic_chat_detail="PASS: got response",
            tool_calling=True, tool_calling_detail="PASS: tool_calls",
            tool_required=True, tool_required_detail="PASS: forced",
            json_mode=True, json_mode_detail="PASS: valid JSON",
            json_schema=True, json_schema_detail="PASS: 3 languages",
        )
        assert r.gate_passed is True


class TestExperimentStatusValues:
    """Test that FAILED_CORRECTNESS status exists and serializes correctly."""

    def test_failed_correctness_value(self):
        assert ExperimentStatus.FAILED_CORRECTNESS.value == "failed_correctness"

    def test_failed_correctness_from_string(self):
        status = ExperimentStatus("failed_correctness")
        assert status == ExperimentStatus.FAILED_CORRECTNESS

    def test_all_statuses(self):
        expected = {"success", "failed", "partial", "failed_correctness"}
        actual = {s.value for s in ExperimentStatus}
        assert actual == expected


class TestEligibility:
    """Test that _is_eligible correctly filters experiments."""

    def _make(self, **kwargs) -> ExperimentSummary:
        defaults = {
            "experiment_id": "test",
            "engine": EngineType.VLLM,
            "status": ExperimentStatus.SUCCESS,
            "peak_throughput": 100.0,
            "low_concurrency_ttft_p95": 50.0,
            "correctness_gate_passed": True,
        }
        defaults.update(kwargs)
        return ExperimentSummary(**defaults)

    def test_eligible(self):
        assert _is_eligible(self._make()) is True

    def test_failed_not_eligible(self):
        assert _is_eligible(self._make(status=ExperimentStatus.FAILED)) is False

    def test_partial_not_eligible(self):
        assert _is_eligible(self._make(status=ExperimentStatus.PARTIAL)) is False

    def test_failed_correctness_not_eligible(self):
        assert _is_eligible(
            self._make(status=ExperimentStatus.FAILED_CORRECTNESS)
        ) is False

    def test_no_correctness_gate_not_eligible(self):
        assert _is_eligible(self._make(correctness_gate_passed=False)) is False

    def test_zero_throughput_not_eligible(self):
        assert _is_eligible(self._make(peak_throughput=0.0)) is False

    def test_zero_latency_not_eligible(self):
        assert _is_eligible(self._make(low_concurrency_ttft_p95=0.0)) is False

    def test_all_requirements_needed(self):
        """All conditions must be met simultaneously."""
        # Success + correctness + metrics = eligible
        assert _is_eligible(self._make()) is True
        # Missing any one condition = not eligible
        assert _is_eligible(self._make(correctness_gate_passed=False)) is False
        assert _is_eligible(self._make(status=ExperimentStatus.FAILED)) is False
        assert _is_eligible(self._make(peak_throughput=0)) is False
