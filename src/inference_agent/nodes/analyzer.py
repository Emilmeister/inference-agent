"""Analyzer node — LLM-driven analysis, Pareto front, stop conditions."""

from __future__ import annotations

import json
import logging

from inference_agent.models import (
    AnalyzerOutput,
    ExperimentResult,
    ExperimentScores,
    ExperimentStatus,
    ExperimentSummary,
    OptimizationClassification,
    OptimizationGoal,
    ParetoPoint,
)
from inference_agent.state import AgentState
from inference_agent.utils.llm import structured_output

logger = logging.getLogger(__name__)

ANALYZER_SYSTEM_PROMPT = """\
You are an expert LLM inference performance analyst. Analyze the latest experiment \
and all historical results to:

1. Write a commentary (2-4 sentences) about the latest experiment — evaluate it against \
all 3 optimization goals (throughput, latency, balanced).
2. Classify this experiment: "best_throughput", "best_latency", "best_balanced", or "none".
3. Decide whether to continue or stop.
4. If continuing, choose the next optimization goal and provide a hint for the planner.

## Noise (cv = coefficient of variation, stdev/mean) — read this before ranking
Each leaderboard row carries a `cv` next to its headline metric. cv reflects \
the per-request dispersion in the underlying phase:
- cv ≤ 0.2 — tight distribution, ranking on the headline number is reliable.
- 0.2 < cv ≤ 0.5 — moderate spread, prefer this config only with clear margin.
- cv > 0.5 — noisy phase; do NOT classify it as best on a small lead. Prefer a \
slightly slower but stable competitor, or recommend re-running this config.
The score column already applies a soft noise derate, so a config with high cv \
won't dominate by raw throughput alone — but you still see the raw numbers and \
should call out noise explicitly in the commentary when it's load-bearing.

## Latest Experiment
{latest_json}

## Leaderboards
### Top 5 by Throughput
{top_throughput}

### Top 5 by Latency (lowest TTFT p95)
{top_latency}

### Top 5 Balanced (best throughput with TTFT p95 < {latency_threshold} ms)
{top_balanced}

## Pareto Front
{pareto_front}

## Budget
Experiment {exp_count} of {max_experiments}. Plateau window: last {plateau_window} experiments.

## Stop Conditions
- Budget exhausted: {exp_count} >= {max_experiments}
- Plateau on ALL 3 goals: last {plateau_window} experiments improved none of the goals \
by more than {plateau_threshold}%
- You judge the search space is exhausted

Respond with JSON:
{{
  "commentary": "...",
  "classification": "best_throughput|best_latency|best_balanced|none",
  "decision": "continue|stop",
  "stop_reason": "..." or null,
  "next_goal": "optimize_throughput|optimize_latency|optimize_balanced|explore",
  "planner_hint": "..."
}}
"""


def _is_eligible(h: ExperimentSummary) -> bool:
    """Check if an experiment is eligible for performance leaderboards.

    Eligibility requires:
    - Experiment succeeded (status=success)
    - Correctness gate passed (basic_chat + tool_calling + json_schema)
    - Non-zero performance metrics
    """
    return (
        h.status.value == "success"
        and h.correctness_gate_passed
        and h.peak_throughput > 0
        and h.low_concurrency_ttft_p95 > 0
    )


def _compute_pareto_front(
    history: list[ExperimentSummary],
) -> list[ParetoPoint]:
    """Compute Pareto front in (throughput↑, latency↓) space."""
    # Only consider eligible experiments (correctness gate + success + valid metrics)
    candidates = [h for h in history if _is_eligible(h)]

    if not candidates:
        return []

    # Sort by throughput descending
    candidates.sort(key=lambda h: h.peak_throughput, reverse=True)

    pareto: list[ParetoPoint] = []
    best_latency = float("inf")

    for h in candidates:
        # A point is Pareto-optimal if no other point has both
        # higher throughput AND lower latency
        if h.low_concurrency_ttft_p95 < best_latency:
            pareto.append(ParetoPoint(
                config_id=h.experiment_id,
                engine=h.engine,
                throughput=h.peak_throughput,
                ttft_p95=h.low_concurrency_ttft_p95,
            ))
            best_latency = h.low_concurrency_ttft_p95

    return pareto


def _check_plateau(
    history: list[ExperimentSummary],
    best_throughput: float,
    best_latency: float,
    window: int,
    threshold: float,
) -> bool:
    """Check if the last `window` SUCCESSFUL experiments showed no improvement."""
    # Only count successful experiments (failed ones don't tell us about perf)
    successful = [
        h for h in history
        if h.peak_throughput > 0 or h.low_concurrency_ttft_p95 > 0
    ]
    if len(successful) < window:
        return False

    recent = successful[-window:]

    # Check throughput improvement
    throughput_improved = any(
        h.peak_throughput > best_throughput * (1 - threshold)
        and h.peak_throughput > best_throughput
        for h in recent
    )

    # Check latency improvement
    latency_improved = any(
        h.low_concurrency_ttft_p95 < best_latency * (1 + threshold)
        and h.low_concurrency_ttft_p95 < best_latency
        and h.low_concurrency_ttft_p95 > 0
        for h in recent
    )

    # Plateau if neither improved
    return not throughput_improved and not latency_improved


# Soft noise penalty applied to throughput_score / latency_score. Capped so
# even very wide distributions never zero out a config (Pareto front is the
# hard mathematical filter, derate is just for *ranking*).
#   factor = 1 - NOISE_DERATE_ALPHA * min(cv, NOISE_CV_CAP)
# With alpha=0.3, cap=1.0 → cv=0.5 → 15% derate, cv=1.0 → 30% derate.
NOISE_DERATE_ALPHA = 0.3
NOISE_CV_CAP = 1.0


def _noise_factor(cv: float) -> float:
    """Multiplicative derate for a normalized score given a noise indicator."""
    if cv <= 0:
        return 1.0
    return 1.0 - NOISE_DERATE_ALPHA * min(cv, NOISE_CV_CAP)


def _compute_scores(
    result: ExperimentResult,
    best_throughput: float,
    best_latency: float,
    pareto: list[ParetoPoint],
) -> ExperimentScores:
    """Compute normalized scores for the experiment.

    Throughput and latency scores are derated by a soft noise factor based on
    the per-request dispersion (cv) at the phases that produced the headline
    metrics. The Pareto-optimal flag is intentionally not derated — a config
    that dominates on raw numbers stays on the front; the score is what
    influences ranking and tie-breaking.
    """
    tp = result.benchmark.peak_output_tokens_per_sec
    lat = result.benchmark.low_concurrency_ttft_p95_ms

    tp_score = tp / best_throughput if best_throughput > 0 else 0.0
    lat_score = best_latency / lat if lat > 0 else 0.0

    tp_score *= _noise_factor(result.benchmark.peak_throughput_e2e_cv)
    lat_score *= _noise_factor(result.benchmark.low_concurrency_ttft_cv)

    # Balanced score: geometric mean of throughput and latency scores
    balanced = (tp_score * lat_score) ** 0.5 if tp_score > 0 and lat_score > 0 else 0.0

    # Check if Pareto-optimal
    is_pareto = any(p.config_id == result.experiment_id for p in pareto)

    return ExperimentScores(
        throughput_score=min(tp_score, 1.0),
        latency_score=min(lat_score, 1.0),
        balanced_score=min(balanced, 1.0),
        is_pareto_optimal=is_pareto,
    )


async def analyzer_node(state: AgentState) -> dict:
    """Analyze the latest experiment result and decide next steps."""
    config = state["config"]
    result = state["current_result"]
    session_history = state.get("experiment_history", [])
    loaded_top_history = state.get("loaded_top_history", [])
    exp_count = state.get("experiments_count", 0) + 1

    # Create summary for this experiment
    summary = ExperimentSummary.from_result(result)

    # Session-only history (accumulated this run; used for plateau detection
    # and best_* state updates). Topd from prior runs live separately and only
    # contribute to leaderboards/Pareto/scoring.
    session_with_current = session_history + [summary]
    union_history = session_with_current + loaded_top_history

    # Update session leaderboards (state-tracked best_* — current session only)
    best_throughput = state.get("best_throughput", 0.0)
    best_throughput_id = state.get("best_throughput_config_id", "")
    best_latency = state.get("best_latency_ttft_p95", float("inf"))
    best_latency_id = state.get("best_latency_config_id", "")
    best_balanced_id = state.get("best_balanced_config_id", "")
    best_balanced_tp = state.get("best_balanced_throughput", 0.0)
    best_balanced_lat = state.get("best_balanced_latency", float("inf"))

    tp = result.benchmark.peak_output_tokens_per_sec
    lat = result.benchmark.low_concurrency_ttft_p95_ms

    # Only update leaderboards if this experiment is eligible
    # (correctness gate passed, success status, valid metrics)
    is_eligible_result = (
        result.status == ExperimentStatus.SUCCESS
        and result.correctness_gate_passed
        and tp > 0 and lat > 0
    )

    if is_eligible_result and tp > best_throughput:
        best_throughput = tp
        best_throughput_id = result.experiment_id

    if is_eligible_result and 0 < lat < best_latency:
        best_latency = lat
        best_latency_id = result.experiment_id

    # Update balanced: best throughput among eligible configs with acceptable latency
    latency_threshold = config.benchmark.latency_threshold_ms
    if is_eligible_result and lat < latency_threshold and tp > best_balanced_tp:
        best_balanced_tp = tp
        best_balanced_lat = lat
        best_balanced_id = result.experiment_id

    # Pareto front and scoring use the union (loaded tops + session). This way
    # the planner sees the global picture and `is_pareto_optimal` reflects
    # cross-run optimality. Plateau detection below uses session-only.
    pareto = _compute_pareto_front(union_history)

    # For scoring use cross-run bests (so the current experiment is normalized
    # against the strongest configs seen anywhere, not just this session).
    eligible_union = [h for h in union_history if _is_eligible(h)]
    union_best_tp = max(
        (h.peak_throughput for h in eligible_union),
        default=best_throughput,
    )
    union_best_lat = min(
        (h.low_concurrency_ttft_p95 for h in eligible_union),
        default=best_latency,
    )
    scores = _compute_scores(result, union_best_tp, union_best_lat, pareto)

    # Check hard stop conditions
    hard_stop = False
    stop_reason = None

    if exp_count >= config.experiments.max_experiments:
        hard_stop = True
        stop_reason = f"Budget exhausted ({exp_count}/{config.experiments.max_experiments})"

    # Plateau uses ONLY the current session — loaded tops would make plateau
    # trip on iteration 1 (newcomers struggle to beat historical bests).
    if not hard_stop and _check_plateau(
        session_with_current,
        best_throughput,
        best_latency,
        config.experiments.plateau_window,
        config.experiments.plateau_threshold,
    ):
        hard_stop = True
        stop_reason = f"Plateau: no improvement in last {config.experiments.plateau_window} experiments"

    # Ask LLM for analysis using claude

    # Prepare leaderboard data (only eligible experiments) — uses the union
    # so the LLM sees prior-run tops alongside current session.
    eligible = [h for h in union_history if _is_eligible(h)]
    sorted_by_tp = sorted(eligible, key=lambda h: h.peak_throughput, reverse=True)[:5]
    sorted_by_lat = sorted(
        eligible,
        key=lambda h: h.low_concurrency_ttft_p95,
    )[:5]
    balanced_candidates = [
        h for h in eligible
        if h.low_concurrency_ttft_p95 < latency_threshold
    ]
    sorted_balanced = sorted(balanced_candidates, key=lambda h: h.peak_throughput, reverse=True)[:5]

    def _format_leaderboard(items: list[ExperimentSummary]) -> str:
        if not items:
            return "None yet"
        lines = []
        for h in items:
            lines.append(
                f"  {h.experiment_id} ({h.engine.value}): "
                f"throughput={h.peak_throughput:.1f} (cv={h.peak_throughput_e2e_cv:.2f}), "
                f"ttft_p95={h.low_concurrency_ttft_p95:.1f}ms (cv={h.low_concurrency_ttft_cv:.2f}), "
                f"config={json.dumps(h.config_digest)}"
            )
        return "\n".join(lines)

    prompt = ANALYZER_SYSTEM_PROMPT.format(
        latest_json=json.dumps({
            "id": result.experiment_id,
            "engine": result.engine.value,
            "status": result.status.value,
            "config": result.config.model_dump(exclude={"rationale"}),
            "peak_throughput": tp,
            "peak_throughput_e2e_cv": result.benchmark.peak_throughput_e2e_cv,
            "ttft_p95": lat,
            "low_concurrency_ttft_cv": result.benchmark.low_concurrency_ttft_cv,
            "smoke_tests": result.smoke_tests.model_dump(),
            "rationale": result.config.rationale,
        }, indent=2, default=str),
        top_throughput=_format_leaderboard(sorted_by_tp),
        top_latency=_format_leaderboard(sorted_by_lat),
        top_balanced=_format_leaderboard(sorted_balanced),
        latency_threshold=latency_threshold,
        pareto_front=json.dumps([p.model_dump() for p in pareto], indent=2, default=str),
        exp_count=exp_count,
        max_experiments=config.experiments.max_experiments,
        plateau_window=config.experiments.plateau_window,
        plateau_threshold=config.experiments.plateau_threshold * 100,
    )

    full_prompt = prompt + "\n\nAnalyze the latest experiment."

    # Short-circuit on validation failures: the experiment never ran, so the
    # LLM has nothing to analyze and tends to hallucinate diagnoses (OOM,
    # backend incompat, etc.) that mislead the planner. Use a deterministic
    # stub instead and skip the LLM call.
    if (result.failure_classification or "") == "validation":
        analysis = AnalyzerOutput(
            commentary=(
                f"Validation rejected the config before launch: "
                f"{(result.error or '').removeprefix('Validation failed: ')[:200]}. "
                "No runtime signal — try a different configuration."
            ),
            classification="none",
            decision="stop" if hard_stop else "continue",
            next_goal="explore",
            planner_hint="Previous config was rejected by the validator; pick different parameters.",
        )
    else:
        try:
            analysis = await structured_output(
                full_prompt, AnalyzerOutput, config.agent_llm
            )
        except Exception as e:
            logger.warning("LLM structured output failed for analyzer: %s", e)
            analysis = AnalyzerOutput(
                commentary="Analysis unavailable (LLM error)",
                classification="none",
                decision="stop" if hard_stop else "continue",
                next_goal="explore",
            )

    # Apply LLM decisions
    commentary = analysis.commentary
    try:
        classification = OptimizationClassification(analysis.classification)
    except ValueError:
        classification = OptimizationClassification.NONE

    decision = analysis.decision
    if hard_stop:
        decision = "stop"

    next_goal_str = analysis.next_goal
    try:
        next_goal = OptimizationGoal(
            f"optimize_{next_goal_str}" if not next_goal_str.startswith("optimize") else next_goal_str
        )
    except ValueError:
        next_goal = OptimizationGoal.EXPLORE

    if decision == "stop" and not stop_reason:
        stop_reason = analysis.stop_reason or "LLM decided search space is exhausted"

    # Update summary with analysis
    summary.llm_commentary = commentary
    summary.optimization_classification = classification
    summary.scores = scores

    logger.info(
        "Analysis: %s | classification=%s | decision=%s | next_goal=%s",
        commentary,
        classification.value,
        decision,
        next_goal.value,
    )

    status = "completed" if decision == "stop" else "running"

    # Build enriched result with LLM analysis
    enriched_result = ExperimentResult(
        **{
            **result.model_dump(),
            "llm_commentary": commentary,
            "optimization_classification": classification,
            "scores": scores,
        }
    )

    return {
        "experiment_history": [summary],
        "experiments_count": exp_count,
        "best_throughput": best_throughput,
        "best_throughput_config_id": best_throughput_id,
        "best_latency_ttft_p95": best_latency,
        "best_latency_config_id": best_latency_id,
        "best_balanced_config_id": best_balanced_id,
        "best_balanced_throughput": best_balanced_tp,
        "best_balanced_latency": best_balanced_lat,
        "pareto_front": pareto,
        "next_optimization_goal": next_goal,
        "status": status,
        "stop_reason": stop_reason,
        "current_result": enriched_result,
    }
