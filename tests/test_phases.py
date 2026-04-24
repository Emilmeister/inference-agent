"""Tests for config-driven benchmark phases and workload classification."""

from inference_agent.benchmark.runner import (
    _classify_workload,
    get_benchmark_phases,
)
from inference_agent.models import BenchmarkConfig


class TestClassifyWorkload:
    def test_agent_short_low_concurrency(self):
        assert _classify_workload(1, 512) == "agent_short"
        assert _classify_workload(4, 2048) == "agent_short"
        assert _classify_workload(16, 4096) == "agent_short"

    def test_throughput_high_concurrency(self):
        assert _classify_workload(64, 512) == "throughput"
        assert _classify_workload(128, 2048) == "throughput"
        assert _classify_workload(256, 512) == "throughput"

    def test_stress_very_high_concurrency(self):
        assert _classify_workload(512, 512) == "stress"
        assert _classify_workload(1024, 128) == "stress"

    def test_long_context(self):
        assert _classify_workload(1, 16384) == "long_context"
        assert _classify_workload(4, 32768) == "long_context"
        assert _classify_workload(1, 65536) == "long_context"

    def test_long_context_even_with_high_concurrency(self):
        """Long prompt always classifies as long_context regardless of concurrency."""
        assert _classify_workload(128, 16384) == "long_context"

    def test_boundary_values(self):
        """Test the exact threshold boundaries."""
        # prompt_length < 8192 → not long_context
        assert _classify_workload(1, 8191) == "agent_short"
        # prompt_length >= 8192 → long_context
        assert _classify_workload(1, 8192) == "long_context"
        # concurrency 63 → agent_short
        assert _classify_workload(63, 512) == "agent_short"
        # concurrency 64 → throughput
        assert _classify_workload(64, 512) == "throughput"
        # concurrency 511 → throughput
        assert _classify_workload(511, 512) == "throughput"
        # concurrency 512 → stress
        assert _classify_workload(512, 512) == "stress"


class TestGetBenchmarkPhases:
    def test_default_config_has_warmup(self):
        phases = get_benchmark_phases(model_max_context=32768)
        warmup = [p for p in phases if p[1] == "warmup"]
        assert len(warmup) == 1
        assert warmup[0][0] == "warmup"

    def test_phases_filtered_by_context(self):
        """Phases that exceed context are skipped."""
        phases = get_benchmark_phases(model_max_context=4096)
        # No long-context phases should be present (16384 + 8192 > 4096)
        long = [p for p in phases if p[1] == "long_context"]
        assert len(long) == 0

    def test_all_phases_have_workload_id(self):
        phases = get_benchmark_phases(model_max_context=262144)
        for phase_id, workload_id, conc, plen, max_out in phases:
            assert workload_id in ("warmup", "agent_short", "throughput", "stress", "long_context"), \
                f"Unknown workload_id={workload_id} for phase {phase_id}"

    def test_custom_concurrency_levels(self):
        cfg = BenchmarkConfig(
            concurrency_levels=[1, 8, 32],
            prompt_lengths=[512],
        )
        phases = get_benchmark_phases(model_max_context=32768, benchmark_config=cfg)
        concurrencies = {p[2] for p in phases if p[1] != "warmup"}
        assert concurrencies == {1, 8, 32}

    def test_custom_prompt_lengths(self):
        cfg = BenchmarkConfig(
            concurrency_levels=[1],
            prompt_lengths=[256, 1024],
        )
        phases = get_benchmark_phases(model_max_context=32768, benchmark_config=cfg)
        prompt_lengths = {p[3] for p in phases if p[1] != "warmup"}
        assert prompt_lengths == {256, 1024}

    def test_max_output_tokens_from_config(self):
        cfg = BenchmarkConfig(
            concurrency_levels=[1],
            prompt_lengths=[512],
            max_output_tokens=128,
        )
        phases = get_benchmark_phases(model_max_context=32768, benchmark_config=cfg)
        non_warmup = [p for p in phases if p[1] != "warmup"]
        assert all(p[4] == 128 for p in non_warmup)

    def test_long_context_uses_long_max_output(self):
        cfg = BenchmarkConfig(
            concurrency_levels=[1],
            prompt_lengths=[16384],
            max_output_tokens=256,
            long_context_max_output_tokens=4096,
        )
        phases = get_benchmark_phases(model_max_context=262144, benchmark_config=cfg)
        long = [p for p in phases if p[1] == "long_context"]
        assert len(long) == 1
        assert long[0][4] == 4096

    def test_long_context_max_concurrency(self):
        """Long context phases are limited to concurrency <= 4."""
        cfg = BenchmarkConfig(
            concurrency_levels=[1, 4, 16, 64, 128],
            prompt_lengths=[32768],
            long_context_max_output_tokens=256,
        )
        phases = get_benchmark_phases(model_max_context=262144, benchmark_config=cfg)
        long = [p for p in phases if p[1] == "long_context"]
        concurrencies = {p[2] for p in long}
        # Only c=1 and c=4 should be present (c>=16 skipped for long context)
        assert max(concurrencies) <= 4

    def test_phase_ids_unique(self):
        phases = get_benchmark_phases(model_max_context=262144)
        # Skip warmup for uniqueness check (it's always "warmup")
        non_warmup_ids = [p[0] for p in phases if p[1] != "warmup"]
        assert len(non_warmup_ids) == len(set(non_warmup_ids))

    def test_max_model_len_overrides_context(self):
        """max_model_len should be used instead of model_max_context when provided."""
        phases_full = get_benchmark_phases(model_max_context=262144)
        phases_limited = get_benchmark_phases(
            model_max_context=262144, max_model_len=4096,
        )
        assert len(phases_limited) < len(phases_full)

    def test_phases_sorted_by_concurrency(self):
        """Phases should be ordered by concurrency (ascending)."""
        phases = get_benchmark_phases(model_max_context=32768)
        non_warmup = [p for p in phases if p[1] != "warmup"]
        concurrencies = [p[2] for p in non_warmup]
        assert concurrencies == sorted(concurrencies)

    def test_empty_concurrency_levels(self):
        cfg = BenchmarkConfig(concurrency_levels=[], prompt_lengths=[512])
        phases = get_benchmark_phases(model_max_context=32768, benchmark_config=cfg)
        # Only warmup
        assert len(phases) == 1
        assert phases[0][1] == "warmup"

    def test_workload_distribution_default(self):
        """With default config, all workload types should be present for large context."""
        phases = get_benchmark_phases(model_max_context=262144)
        workloads = {p[1] for p in phases}
        assert workloads == {"warmup", "agent_short", "throughput", "stress", "long_context"}
