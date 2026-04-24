"""Tests for percentile calculation with linear interpolation."""

from inference_agent.benchmark.runner import _compute_percentiles, _percentile


class TestPercentile:
    def test_empty_list(self):
        assert _percentile([], 0.5) == 0.0

    def test_single_value(self):
        assert _percentile([42.0], 0.5) == 42.0
        assert _percentile([42.0], 0.95) == 42.0

    def test_two_values_median(self):
        result = _percentile([10.0, 20.0], 0.5)
        assert result == 15.0

    def test_two_values_p95(self):
        result = _percentile([10.0, 20.0], 0.95)
        assert result == 19.5

    def test_interpolation(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        # p50 = index 2.0 → value 3.0
        assert _percentile(values, 0.5) == 3.0
        # p25 = index 1.0 → value 2.0
        assert _percentile(values, 0.25) == 2.0
        # p75 = index 3.0 → value 4.0
        assert _percentile(values, 0.75) == 4.0

    def test_interpolation_between_values(self):
        values = [10.0, 20.0, 30.0, 40.0]
        # p50 = index 1.5 → 20 + 0.5 * (30 - 20) = 25.0
        assert _percentile(values, 0.5) == 25.0

    def test_p0_and_p100(self):
        values = [1.0, 5.0, 10.0]
        assert _percentile(values, 0.0) == 1.0
        assert _percentile(values, 1.0) == 10.0


class TestComputePercentiles:
    def test_empty(self):
        stats = _compute_percentiles([])
        assert stats.mean == 0.0
        assert stats.p95 == 0.0

    def test_single_value(self):
        stats = _compute_percentiles([100.0])
        assert stats.mean == 100.0
        assert stats.median == 100.0
        assert stats.min == 100.0
        assert stats.max == 100.0

    def test_basic_stats(self):
        values = list(range(1, 101))  # 1 to 100
        stats = _compute_percentiles([float(v) for v in values])
        assert stats.mean == 50.5
        assert stats.min == 1.0
        assert stats.max == 100.0
        assert stats.median == 50.5  # interpolated
        assert stats.p95 > 90

    def test_small_sample(self):
        """Verify interpolation works correctly on small samples."""
        stats = _compute_percentiles([1.0, 2.0, 3.0])
        assert stats.median == 2.0
        # p95 = index 1.9 → 2 + 0.9 * (3 - 2) = 2.9
        assert abs(stats.p95 - 2.9) < 0.01
