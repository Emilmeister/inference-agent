"""Tests for agentic_long_context phase building in get_benchmark_phases.

The workload is shaped as `shared_prefix + unique_per_session + turns × (max_out + tool_result_max)`.
Phases are emitted at each configured agentic concurrency, with `prompt_length`
set to `shared + unique` (the total first-turn user message size).
"""

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
        """3 concurrency levels → 3 agentic phases at shared+unique prompt size."""
        cfg = BenchmarkConfig(
            concurrency_levels=[1],
            prompt_lengths=[512],
            enable_agentic_long_context=True,
            agentic_shared_prefix_tokens=16_000,
            agentic_unique_prompt_tokens=8_000,
            agentic_max_output_tokens=2_400,
            agentic_tool_result_max_tokens=1_536,
            agentic_turns_per_session=4,
            agentic_concurrency_levels=[4, 8, 16],
        )
        # Required: 24k + 4 * (2.4k + 1.5k) ≈ 40k. 64k headroom is fine.
        phases = get_benchmark_phases(
            model_max_context=64_000, benchmark_config=cfg,
        )
        agentic = [p for p in phases if p[1] == "agentic_long_context"]
        assert len(agentic) == 3
        concurrencies = sorted(p[2] for p in agentic)
        assert concurrencies == [4, 8, 16]
        # prompt_length is the SUM of shared + unique (= total first-turn user text).
        assert all(p[3] == 24_000 for p in agentic)
        assert all(p[4] == 2_400 for p in agentic)

    def test_skipped_when_context_too_small(self):
        """If max context can't hold shared+unique + turns × (out + tool), skip agentic."""
        cfg = BenchmarkConfig(
            concurrency_levels=[1],
            prompt_lengths=[512],
            enable_agentic_long_context=True,
            agentic_shared_prefix_tokens=16_000,
            agentic_unique_prompt_tokens=8_000,
            agentic_max_output_tokens=8_192,
            agentic_tool_result_max_tokens=5_120,
            agentic_turns_per_session=4,
            agentic_concurrency_levels=[4, 8, 16],
        )
        # Required: 24k + 4 * (8k + 5k) ≈ 76k — way more than 32k context.
        phases = get_benchmark_phases(
            model_max_context=32_000, benchmark_config=cfg,
        )
        agentic = [p for p in phases if p[1] == "agentic_long_context"]
        assert agentic == []

    def test_phase_id_and_workload_unique(self):
        cfg = BenchmarkConfig(
            concurrency_levels=[1],
            prompt_lengths=[512],
            enable_agentic_long_context=True,
            agentic_concurrency_levels=[4, 8],
            agentic_shared_prefix_tokens=6_000,
            agentic_unique_prompt_tokens=2_000,
            agentic_max_output_tokens=512,
            agentic_tool_result_max_tokens=512,
            agentic_turns_per_session=2,
        )
        phases = get_benchmark_phases(
            model_max_context=64_000, benchmark_config=cfg,
        )
        agentic = [p for p in phases if p[1] == "agentic_long_context"]
        ids = [p[0] for p in agentic]
        # phase_id encodes shared+unique as the prompt length.
        assert ids == ["agentic_c4_p8000", "agentic_c8_p8000"]

    def test_max_model_len_overrides_context(self):
        """When experiment caps max_model_len below required, agentic still skipped."""
        cfg = BenchmarkConfig(
            concurrency_levels=[1],
            prompt_lengths=[512],
            enable_agentic_long_context=True,
            agentic_shared_prefix_tokens=16_000,
            agentic_unique_prompt_tokens=8_000,
            agentic_max_output_tokens=8_192,
            agentic_tool_result_max_tokens=5_120,
            agentic_turns_per_session=4,
            agentic_concurrency_levels=[4, 8, 16],
        )
        phases = get_benchmark_phases(
            model_max_context=200_000,
            max_model_len=32_000,  # below the 76k agentic budget
            benchmark_config=cfg,
        )
        agentic = [p for p in phases if p[1] == "agentic_long_context"]
        assert agentic == []
