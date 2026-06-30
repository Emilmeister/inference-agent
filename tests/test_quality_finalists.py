"""Tests for finalist selection from analyzer state."""

from __future__ import annotations

from inference_agent.quality.finalists import (
    distinct_experiment_ids,
    select_finalist_refs,
)


def _state(**kw):
    base = {
        "best_agentic_config_id": "exp_agentic",
        "best_latency_config_id": "exp_latency",
        "best_balanced_config_id": "exp_balanced",
        "best_throughput_config_id": "exp_throughput",
    }
    base.update(kw)
    return base


def test_select_default_three_categories():
    refs = select_finalist_refs(_state(), ["agentic", "latency", "balanced"])
    assert [(r.category, r.experiment_id) for r in refs] == [
        ("agentic", "exp_agentic"),
        ("latency", "exp_latency"),
        ("balanced", "exp_balanced"),
    ]


def test_skips_empty_winners():
    refs = select_finalist_refs(
        _state(best_latency_config_id=""), ["agentic", "latency", "balanced"]
    )
    assert [r.category for r in refs] == ["agentic", "balanced"]


def test_unknown_category_ignored():
    refs = select_finalist_refs(_state(), ["agentic", "bogus"])
    assert [r.category for r in refs] == ["agentic"]


def test_throughput_category_maps_to_legacy_key():
    refs = select_finalist_refs(_state(), ["throughput"])
    assert refs[0].experiment_id == "exp_throughput"


def test_distinct_experiment_ids_dedups_preserving_order():
    refs = select_finalist_refs(
        _state(best_balanced_config_id="exp_agentic"),  # same as agentic
        ["agentic", "latency", "balanced"],
    )
    assert distinct_experiment_ids(refs) == ["exp_agentic", "exp_latency"]
