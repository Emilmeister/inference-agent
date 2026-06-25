"""Unit tests for resolve_speculative_draft_model.

The executor must prefetch the speculative-decoding draft model before launch
(engines run with HF_HUB_OFFLINE=1). The draft can be specified two ways:
the dedicated `speculative_draft_model` field (planner path) or embedded in a
raw `--speculative-config` JSON in `extra_engine_args` (operator-baseline path,
the only way to also pass method/attention_backend). Both must be resolved.
"""

from __future__ import annotations

from inference_agent.models import ExperimentConfig
from inference_agent.nodes.executor import resolve_speculative_draft_model


def _ec(**kwargs) -> ExperimentConfig:
    return ExperimentConfig(engine="vllm", **kwargs)


def test_dedicated_field_wins():
    exp = _ec(speculative_algorithm="eagle3", speculative_draft_model="Org/Draft")
    assert resolve_speculative_draft_model(exp) == "Org/Draft"


def test_json_speculative_config_separate_arg():
    exp = _ec(
        extra_engine_args=[
            "--speculative-config",
            '{"method": "eagle3", "model": "Inferact/MiniMax-M3-EAGLE3", '
            '"num_speculative_tokens": 3, "attention_backend": "FLASH_ATTN"}',
        ]
    )
    assert resolve_speculative_draft_model(exp) == "Inferact/MiniMax-M3-EAGLE3"


def test_json_speculative_config_equals_form():
    exp = _ec(extra_engine_args=['--speculative-config={"model": "Foo/Bar"}'])
    assert resolve_speculative_draft_model(exp) == "Foo/Bar"


def test_none_when_absent():
    assert resolve_speculative_draft_model(_ec()) is None


def test_malformed_json_returns_none():
    exp = _ec(extra_engine_args=["--speculative-config", "not-json"])
    assert resolve_speculative_draft_model(exp) is None


def test_missing_model_key_returns_none():
    exp = _ec(extra_engine_args=["--speculative-config", '{"method": "eagle3"}'])
    assert resolve_speculative_draft_model(exp) is None


def test_placeholder_dedicated_falls_through_to_json():
    # planner sometimes emits "none"/"null" sentinels — those must not shadow a
    # real draft sitting in the speculative-config JSON.
    exp = _ec(
        speculative_draft_model="none",
        extra_engine_args=["--speculative-config", '{"model": "X/Y"}'],
    )
    assert resolve_speculative_draft_model(exp) == "X/Y"
