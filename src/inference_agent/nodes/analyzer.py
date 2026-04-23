"""Analyzer node — LLM-driven analysis, Pareto front, stop conditions."""

from __future__ import annotations

import json
import logging

from langchain_openai import ChatOpenAI

from inference_agent.models import (
    AnalyzerOutput,
    ExperimentResult,
    ExperimentScores,
    ExperimentSummary,
    OptimizationClassification,
    OptimizationGoal,
    ParetoPoint,
)
from inference_agent.nodes.reporter import save_experiment
from inference_agent.state import AgentState

logger = logging.getLogger(__name__)

ANALYZER_SYSTEM_PROMPT = """\
You are an expert LLM inference performance analyst. Analyze the latest experiment \
and all historical results to:

1. Write a commentary (2-4 sentences) about the latest experiment — evaluate it against \
all 3 optimization goals (throughput, latency, balanced).
2. Classify this experiment: "best_throughput", "best_latency", "best_balanced", or "none".
3. Decide whether to continue or stop.
4. If continuing, choose the next optimization goal and provide a hint for the planner.

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


def _compute_pareto_front(
    history: list[ExperimentSummary],
) -> list[ParetoPoint]:
    """Compute Pareto front in (throughput↑, latency↓) space."""
    # Only consider successful experiments with valid metrics
    candidates = [
        h for h in history
        if h.status.value == "success"
        and h.peak_throughput > 0
        and h.low_concurrency_ttft_p95 > 0
    ]

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
    """Check if the last `window` experiments showed no improvement on any goal."""
    if len(history) < window:
        return False

    # Don't declare plateau if we have no successful experiments yet —
    # keep trying until at least one experiment produces real metrics
    has_any_success = any(
        h.peak_throughput > 0 or h.low_concurrency_ttft_p95 > 0
        for h in history
    )
    if not has_any_success:
        return False

    recent = history[-window:]

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


def _compute_scores(
    result: ExperimentResult,
    best_throughput: float,
    best_latency: float,
    pareto: list[ParetoPoint],
) -> ExperimentScores:
    """Compute normalized scores for the experiment."""
    tp = result.benchmark.peak_output_tokens_per_sec
    lat = result.benchmark.low_concurrency_ttft_p95_ms

    tp_score = tp / best_throughput if best_throughput > 0 else 0.0
    lat_score = best_latency / lat if lat > 0 else 0.0

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
    history = state.get("experiment_history", [])
    exp_count = state.get("experiments_count", 0) + 1

    # Create summary for this experiment
    summary = ExperimentSummary.from_result(result)

    # Update leaderboards
    best_throughput = state.get("best_throughput", 0.0)
    best_throughput_id = state.get("best_throughput_config_id", "")
    best_latency = state.get("best_latency_ttft_p95", float("inf"))
    best_latency_id = state.get("best_latency_config_id", "")
    best_balanced_id = state.get("best_balanced_config_id", "")
    best_balanced_tp = state.get("best_balanced_throughput", 0.0)
    best_balanced_lat = state.get("best_balanced_latency", float("inf"))

    tp = result.benchmark.peak_output_tokens_per_sec
    lat = result.benchmark.low_concurrency_ttft_p95_ms

    if tp > best_throughput:
        best_throughput = tp
        best_throughput_id = result.experiment_id

    if 0 < lat < best_latency:
        best_latency = lat
        best_latency_id = result.experiment_id

    # Update balanced: best throughput among configs with acceptable latency
    latency_threshold = config.benchmark.latency_threshold_ms
    if lat > 0 and lat < latency_threshold and tp > best_balanced_tp:
        best_balanced_tp = tp
        best_balanced_lat = lat
        best_balanced_id = result.experiment_id

    # Compute Pareto front with updated history
    all_history = history + [summary]
    pareto = _compute_pareto_front(all_history)

    # Compute scores
    scores = _compute_scores(result, best_throughput, best_latency, pareto)

    # Check hard stop conditions
    hard_stop = False
    stop_reason = None

    if exp_count >= config.experiments.max_experiments:
        hard_stop = True
        stop_reason = f"Budget exhausted ({exp_count}/{config.experiments.max_experiments})"

    if not hard_stop and _check_plateau(
        all_history,
        best_throughput,
        best_latency,
        config.experiments.plateau_window,
        config.experiments.plateau_threshold,
    ):
        hard_stop = True
        stop_reason = f"Plateau: no improvement in last {config.experiments.plateau_window} experiments"

    # Ask LLM for analysis using structured output
    llm = ChatOpenAI(
        base_url=config.agent_llm.base_url,
        api_key=config.agent_llm.api_key,
        model=config.agent_llm.model,
        temperature=0.2,
    )
    structured_llm = llm.with_structured_output(AnalyzerOutput)

    # Prepare leaderboard data
    sorted_by_tp = sorted(all_history, key=lambda h: h.peak_throughput, reverse=True)[:5]
    sorted_by_lat = sorted(
        [h for h in all_history if h.low_concurrency_ttft_p95 > 0],
        key=lambda h: h.low_concurrency_ttft_p95,
    )[:5]
    balanced_candidates = [
        h for h in all_history
        if h.low_concurrency_ttft_p95 > 0
        and h.low_concurrency_ttft_p95 < latency_threshold
    ]
    sorted_balanced = sorted(balanced_candidates, key=lambda h: h.peak_throughput, reverse=True)[:5]

    def _format_leaderboard(items: list[ExperimentSummary]) -> str:
        if not items:
            return "None yet"
        lines = []
        for h in items:
            lines.append(
                f"  {h.experiment_id} ({h.engine.value}): "
                f"throughput={h.peak_throughput:.1f}, "
                f"ttft_p95={h.low_concurrency_ttft_p95:.1f}ms, "
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
            "ttft_p95": lat,
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

    try:
        analysis: AnalyzerOutput = await structured_llm.ainvoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Analyze the latest experiment."},
        ])
    except Exception as e:
        logger.warning("Structured output failed for analyzer: %s", e)
        analysis = AnalyzerOutput(
            commentary="Analysis unavailable (structured output error)",
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
        commentary[:100],
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

    # Save JSON file HERE — after LLM enrichment, guaranteeing
    # commentary, scores, and Pareto data are included
    save_experiment(enriched_result, config.storage.experiments_dir)

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
