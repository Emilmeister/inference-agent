"""Quality validation of finalist configs (so-testing + terminal-bench).

After the optimization loop converges, the `quality_finalize` node validates
the top finalists (agentic / latency / balanced winners) with two external
suites — so-testing (tool-calls + structured output) and terminal-bench
(agentic scenarios via harbor) — to produce a prod-readiness signal. Quality
is REPORT-ONLY: it never feeds back into the planner/analyzer or leaderboards.

This package holds the pure, side-effect-free pieces:
  - `fingerprint` — a stable hash over the quality-relevant config dimensions
    so finalists that differ only in batching/memory knobs share one (costly)
    terminal-bench run instead of re-running it per finalist.
  - `finalists` — selecting the finalist experiment ids from analyzer state.

The container relaunch + subprocess orchestration lives in `runner.py` and the
node in `inference_agent.nodes.quality_finalize`.
"""

from inference_agent.quality.fingerprint import quality_fingerprint
from inference_agent.quality.finalists import FinalistRef, select_finalist_refs

__all__ = [
    "FinalistRef",
    "quality_fingerprint",
    "select_finalist_refs",
]
