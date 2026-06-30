"""Finalist selection from analyzer state.

When the optimization loop stops, the heads of the three leaderboards are the
finalists we validate for prod-readiness:

  - ``agentic``    → the agentic-first winner (most parallel agents under SLO);
                     this is the codebase's primary "throughput" objective.
  - ``latency``    → lowest TTFT p95 at concurrency 1.
  - ``balanced``   → best agentic concurrency while keeping TTFT under the SLO.

``throughput`` (raw synthetic peak) is also available for callers that want the
legacy axis, mapped to the historical ``best_throughput_config_id``.

Each leaderboard stores the winner's ``experiment_id`` in state. A single
experiment can win more than one category — callers dedup by experiment id (and
later by quality fingerprint) before running the costly suites.
"""

from __future__ import annotations

from dataclasses import dataclass

from inference_agent.state import AgentState

# Category label → the AgentState key holding that leaderboard's winner id.
_CATEGORY_STATE_KEY: dict[str, str] = {
    "agentic": "best_agentic_config_id",
    "throughput": "best_throughput_config_id",
    "latency": "best_latency_config_id",
    "balanced": "best_balanced_config_id",
}

VALID_CATEGORIES: frozenset[str] = frozenset(_CATEGORY_STATE_KEY)


@dataclass(frozen=True)
class FinalistRef:
    """A finalist: which leaderboard it won and the experiment that won it."""

    category: str
    experiment_id: str


def select_finalist_refs(
    state: AgentState,
    categories: list[str],
) -> list[FinalistRef]:
    """Resolve the requested leaderboard categories to finalist references.

    Skips categories with no winner yet (empty id) and unknown category names.
    Order follows `categories`. The same experiment id may appear under more
    than one category — that is intentional (a config can be the best on
    several axes); the caller deduplicates as needed.
    """
    refs: list[FinalistRef] = []
    for category in categories:
        key = _CATEGORY_STATE_KEY.get(category)
        if key is None:
            continue
        experiment_id = state.get(key) or ""
        if experiment_id:
            refs.append(FinalistRef(category=category, experiment_id=experiment_id))
    return refs


def distinct_experiment_ids(refs: list[FinalistRef]) -> list[str]:
    """Unique experiment ids across refs, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for ref in refs:
        if ref.experiment_id not in seen:
            seen.add(ref.experiment_id)
            out.append(ref.experiment_id)
    return out
