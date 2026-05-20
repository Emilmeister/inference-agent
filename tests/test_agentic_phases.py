"""Tests for agentic_long_context phase building in get_benchmark_phases."""

from inference_agent.benchmark.runner import get_benchmark_phases
from inference_agent.models import BenchmarkConfig


class TestAgenticPhasesEnabled:
    def test_disabled_by_default(self):
        """Without enable_agentic_long_context, no agentic phases are generated."""
        cfg = BenchmarkConfig(
            concurrency_levels=[1],
            prompt_lengths=[512],
        )
        phases = get_benchmark_phases(model_max_context=262144, benchmark_config=cfg)
        agentic = [p for p in phases if p[1] == "agentic_long_context"]
        assert agentic == []

    def test_added_when_enabled_and_context_fits(self):
        """3 concurrency levels → 3 agentic phases at the configured prefix length."""
        cfg = BenchmarkConfig(
            concurrency_levels=[1],
            prompt_lengths=[512],
            enable_agentic_long_context=True,
            agentic_prefix_tokens=64_000,
            agentic_max_output_tokens=16_384,
            agentic_tool_result_max_tokens=5_120,
            agentic_turns_per_session=4,
            agentic_concurrency_levels=[4, 8, 16],
        )
        # Required: 64k + 4 * (16k + 5k) ≈ 148k. Use 200k to be safe.
        phases = get_benchmark_phases(
            model_max_context=200_000, benchmark_config=cfg,
        )
        agentic = [p for p in phases if p[1] == "agentic_long_context"]
        assert len(agentic) == 3
        concurrencies = sorted(p[2] for p in agentic)
        assert concurrencies == [4, 8, 16]
        # All agentic phases use prefix_tokens as prompt_length and the configured max_out.
        assert all(p[3] == 64_000 for p in agentic)
        assert all(p[4] == 16_384 for p in agentic)

    def test_skipped_when_context_too_small(self):
        """If max context can't hold prefix + turns × (out + tool), skip agentic."""
        cfg = BenchmarkConfig(
            concurrency_levels=[1],
            prompt_lengths=[512],
            enable_agentic_long_context=True,
            agentic_prefix_tokens=64_000,
            agentic_max_output_tokens=16_384,
            agentic_tool_result_max_tokens=5_120,
            agentic_turns_per_session=4,
            agentic_concurrency_levels=[4, 8, 16],
        )
        # 64k context — way less than 148k required for one full agentic session.
        phases = get_benchmark_phases(
            model_max_context=64_000, benchmark_config=cfg,
        )
        agentic = [p for p in phases if p[1] == "agentic_long_context"]
        assert agentic == []

    def test_phase_id_and_workload_unique(self):
        cfg = BenchmarkConfig(
            concurrency_levels=[1],
            prompt_lengths=[512],
            enable_agentic_long_context=True,
            agentic_concurrency_levels=[4, 8],
            agentic_prefix_tokens=8_000,
            agentic_max_output_tokens=512,
            agentic_tool_result_max_tokens=512,
            agentic_turns_per_session=2,
        )
        phases = get_benchmark_phases(
            model_max_context=64_000, benchmark_config=cfg,
        )
        agentic = [p for p in phases if p[1] == "agentic_long_context"]
        ids = [p[0] for p in agentic]
        assert ids == ["agentic_c4_p8000", "agentic_c8_p8000"]

    def test_max_model_len_overrides_context(self):
        """When experiment caps max_model_len below required, agentic still skipped."""
        cfg = BenchmarkConfig(
            concurrency_levels=[1],
            prompt_lengths=[512],
            enable_agentic_long_context=True,
        )
        phases = get_benchmark_phases(
            model_max_context=200_000,
            max_model_len=64_000,
            benchmark_config=cfg,
        )
        agentic = [p for p in phases if p[1] == "agentic_long_context"]
        assert agentic == []
