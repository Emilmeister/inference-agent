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
the 3 ACTIVE goals: AGENTIC (primary), LATENCY, BALANCED. Pure synthetic-throughput \
peaks are informational only — do NOT use them to declare leadership.
2. Classify this experiment: "best_agentic", "best_latency", "best_balanced", or "none". \
DO NOT emit "best_throughput" — that classification is retired; configurations that \
shine at c=256, p=512 but cannot serve agentic traffic under SLO are NOT winners.
3. Decide whether to continue or stop.
4. If continuing, choose the next optimization goal and provide a hint for the planner.

## The agentic SLO contract
A phase is `viable` only when ALL of these hold simultaneously: TTFT p95 ≤ \
agentic SLO, tpot p95 ≤ agentic SLO, session error_rate ≤ agentic SLO. \
`max_viable_agentic_concurrency` (called `agentic.max_viable_concurrency` in \
the payload) is the largest c that passed. **This is the headline metric** \
for the agentic-first ranking — higher beats lower; tie-breaks go to lower \
`agentic_tpot_p95`, then higher `agentic_peak_throughput`.

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

## Per-phase failures (`failed_phases` in the latest experiment payload)
If `failed_phases` is present, one or more benchmark phases breached the \
error-rate gate with a NON-timeout-shaped failure (HTTP 5xx, connection \
refused, parse error, or any non-agentic workload failure). These are real \
malfunctions, not ceiling probes. The headline peak_throughput / ttft_p95 \
are computed from the SURVIVING phases only, so they can look fine even \
when entire workload categories failed 100%. You MUST:
- Acknowledge each failed phase in your commentary (phase_id, workload, \
  concurrency, sample_error).
- NEVER write "no benchmark errors" or "agentic stability improvement" when \
  `failed_phases` is non-empty — that contradicts the structured signal and \
  misleads the planner.
- If the same workload fails across multiple consecutive experiments, treat \
  it as a systemic issue and route the planner_hint toward addressing the \
  cause rather than chasing throughput knobs.

## Agentic ceiling probe (`agentic` in the latest experiment payload)
If `agentic` is present, the run included an agentic_long_context sweep. \
Fields: `max_viable_concurrency` (headline — how many parallel code-agents \
this config served while meeting SLO), `probed`, `viable`, `ceiling_probed`. \
The `ceiling_probed` concurrencies are EXPECTED outcomes of the sweep, NOT \
failures — they tell you where the config ran out of room. Rules:
- DO NOT treat ceiling_probed entries as malfunctions. They are not in \
  `failed_phases` for a reason.
- DO surface `max_viable_concurrency` when discussing the run's agentic \
  capacity. A higher value beats a lower one for agentic-throughput goals.
- If `max_viable_concurrency == 0` AND `ceiling_probed` is non-empty, the \
  ceiling is below the lowest probed concurrency — flag this as a \
  configuration/SLO issue, not as a knob to tune.

## Latest Experiment
{latest_json}

## Leaderboards
### Top 5 by Max Viable Agentic Concurrency (primary)
{top_agentic}

### Top 5 by Latency (lowest TTFT p95)
{top_latency}

### Top 5 Balanced (best agentic concurrency with TTFT p95 < {latency_threshold} ms)
{top_balanced}

## Agentic Pareto Front (max_viable_c ↑ vs agentic_tpot_p95 ↓)
{agentic_pareto_front}

## Throughput vs TTFT Pareto Front (informational)
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
  "classification": "best_agentic|best_latency|best_balanced|none",
  "decision": "continue|stop",
  "stop_reason": "..." or null,
  "next_goal": "optimize_agentic|optimize_latency|optimize_balanced|explore",
  "planner_hint": "..."
}}
"""


def _is_eligible(h: ExperimentSummary) -> bool:
    """Check if an experiment is eligible for the throughput/latency boards.

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


def _is_agentic_eligible(h: ExperimentSummary) -> bool:
    """Eligibility for the agentic leaderboard / Pareto front.

    Looser than `_is_eligible`: a config that crashed correctness or had
    zero synthetic throughput can still be a non-starter for the agentic
    rank. We only need: status=success, correctness gate passed, and an
    actual non-zero max_viable_agentic_concurrency.
    """
    return (
        h.status.value == "success"
        and h.correctness_gate_passed
        and h.agentic_max_viable_concurrency > 0
    )


def _compute_pareto_front(
    history: list[ExperimentSummary],
) -> list[ParetoPoint]:
    """Compute Pareto front in (throughput↑, latency↓) space.

    Informational / dashboard-only. The primary front for the agentic-first
    goal is `_compute_agentic_pareto_front`.
    """
    candidates = [h for h in history if _is_eligible(h)]
    if not candidates:
        return []

    candidates.sort(key=lambda h: h.peak_throughput, reverse=True)

    pareto: list[ParetoPoint] = []
    best_latency = float("inf")
    for h in candidates:
        if h.low_concurrency_ttft_p95 < best_latency:
            pareto.append(ParetoPoint(
                config_id=h.experiment_id,
                engine=h.engine,
                throughput=h.peak_throughput,
                ttft_p95=h.low_concurrency_ttft_p95,
            ))
            best_latency = h.low_concurrency_ttft_p95
    return pareto


def _compute_agentic_pareto_front(
    history: list[ExperimentSummary],
) -> list[ParetoPoint]:
    """Pareto front in (max_viable_agentic_c ↑, agentic_tpot_p95 ↓) space.

    Primary front for the agentic-first goal — points on it represent
    configurations that no other config dominates on both axes (more
    parallel agents AND snappier per-token latency).
    """
    candidates = [h for h in history if _is_agentic_eligible(h)]
    if not candidates:
        return []

    # Sort by max viable concurrency DESC, so we sweep from "most agents"
    # downward and keep points that improve on tpot.
    candidates.sort(key=lambda h: h.agentic_max_viable_concurrency, reverse=True)

    pareto: list[ParetoPoint] = []
    best_tpot = float("inf")
    for h in candidates:
        tpot = h.agentic_tpot_p95 if h.agentic_tpot_p95 > 0 else float("inf")
        if tpot < best_tpot:
            pareto.append(ParetoPoint(
                config_id=h.experiment_id,
                engine=h.engine,
                agentic_max_viable_c=h.agentic_max_viable_concurrency,
                agentic_tpot_p95=h.agentic_tpot_p95,
            ))
            best_tpot = tpot
    return pareto


def _check_plateau(
    history: list[ExperimentSummary],
    best_agentic_c: int,
    best_latency: float,
    window: int,
    threshold: float,
) -> bool:
    """Check if the last `window` experiments improved on the prior best.

    Plateau triggers only when NEITHER the agentic concurrency NOR the cold-
    start latency moved meaningfully. We do NOT consider raw throughput for
    plateau anymore — c=256, p=512 wins are not what we're optimizing for.

    `best_agentic_c` / `best_latency` MUST be the bests from BEFORE the
    latest experiment was folded in — otherwise the just-set new best can
    never be beaten by itself. Caller is responsible for passing pre-update
    bests.

    "Improvement" is strict and threshold-bounded:
      - agentic_c improves iff new > floor(best * (1 + threshold)) AND new > best
      - latency improves iff ttft <= best * (1 - threshold)
    """
    countable = [
        h for h in history
        if h.agentic_max_viable_concurrency > 0 or h.low_concurrency_ttft_p95 > 0
    ]
    if len(countable) < window:
        return False

    recent = countable[-window:]

    # For an integer metric (concurrency) the multiplicative threshold can
    # round down to zero on small bests — guard with a strict ">" so a
    # first-ever non-zero value always counts as improvement.
    agentic_improved = any(
        h.agentic_max_viable_concurrency > 0
        and h.agentic_max_viable_concurrency
        > max(best_agentic_c, 0) * (1 + threshold)
        and h.agentic_max_viable_concurrency > best_agentic_c
        for h in recent
    )

    latency_improved = any(
        0 < h.low_concurrency_ttft_p95 <= best_latency * (1 - threshold)
        for h in recent
    )

    return not (agentic_improved or latency_improved)


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
    best_agentic_c: int,
    best_throughput: float,
    best_latency: float,
    pareto: list[ParetoPoint],
    agentic_pareto: list[ParetoPoint],
) -> ExperimentScores:
    """Compute normalized scores for the experiment.

    `agentic_score` is the primary axis — it's just max_viable_c normalized
    against the best known so far, no noise derate (concurrency is an
    integer, the runner-side SLO gate already excluded noisy phases). Then
    throughput/latency/balanced scores remain for backward compat and the
    dashboard but DO NOT drive leaderboards anymore.
    """
    bench = result.benchmark

    agentic_c = bench.max_viable_agentic_concurrency
    agentic_score = agentic_c / best_agentic_c if best_agentic_c > 0 else 0.0

    tp = bench.peak_output_tokens_per_sec
    lat = bench.low_concurrency_ttft_p95_ms

    tp_score = tp / best_throughput if best_throughput > 0 else 0.0
    lat_score = best_latency / lat if lat > 0 else 0.0

    tp_score *= _noise_factor(bench.peak_throughput_e2e_cv)
    lat_score *= _noise_factor(bench.low_concurrency_ttft_cv)

    # Balanced score: geometric mean of agentic_score and latency_score —
    # because the active "balanced" goal is "maximize agentic c while keeping
    # cold-start latency low", not "maximize raw throughput while keeping
    # latency low".
    balanced = (
        (agentic_score * lat_score) ** 0.5
        if agentic_score > 0 and lat_score > 0
        else 0.0
    )

    is_pareto = any(p.config_id == result.experiment_id for p in pareto)
    is_agentic_pareto = any(p.config_id == result.experiment_id for p in agentic_pareto)

    return ExperimentScores(
        agentic_score=min(agentic_score, 1.0),
        throughput_score=min(tp_score, 1.0),
        latency_score=min(lat_score, 1.0),
        balanced_score=min(balanced, 1.0),
        is_pareto_optimal=is_pareto,
        is_agentic_pareto_optimal=is_agentic_pareto,
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

    # Snapshot session leaderboards BEFORE this experiment is folded in. The
    # plateau check below must compare the window against the *prior* best,
    # otherwise a record-breaking run sets best_*=its_own_value and then
    # trivially fails to beat itself, falsely tripping plateau.
    best_agentic_c = state.get("best_agentic_max_viable_c", 0)
    best_agentic_id = state.get("best_agentic_config_id", "")
    best_agentic_tpot = state.get("best_agentic_tpot_p95", float("inf"))
    best_agentic_tp = state.get("best_agentic_throughput", 0.0)
    best_throughput = state.get("best_throughput", 0.0)
    best_throughput_id = state.get("best_throughput_config_id", "")
    best_latency = state.get("best_latency_ttft_p95", float("inf"))
    best_latency_id = state.get("best_latency_config_id", "")
    best_balanced_id = state.get("best_balanced_config_id", "")
    best_balanced_tp = state.get("best_balanced_throughput", 0.0)
    best_balanced_lat = state.get("best_balanced_latency", float("inf"))

    agentic_c_now = result.benchmark.max_viable_agentic_concurrency
    agentic_tpot_now = result.benchmark.agentic_tpot_p95_ms
    agentic_peak_now = result.benchmark.agentic_peak_output_tokens_per_sec
    tp = result.benchmark.peak_output_tokens_per_sec
    lat = result.benchmark.low_concurrency_ttft_p95_ms

    # Two Pareto fronts: agentic (primary) and throughput-vs-latency
    # (informational / backward-compat).
    pareto = _compute_pareto_front(union_history)
    agentic_pareto = _compute_agentic_pareto_front(union_history)

    # Scoring uses cross-run bests so a current experiment is normalized
    # against the strongest configs seen anywhere (not just this session).
    eligible_throughput_union = [h for h in union_history if _is_eligible(h)]
    eligible_agentic_union = [h for h in union_history if _is_agentic_eligible(h)]
    union_best_tp = max(
        (h.peak_throughput for h in eligible_throughput_union),
        default=best_throughput,
    )
    union_best_lat = min(
        (h.low_concurrency_ttft_p95 for h in eligible_throughput_union),
        default=best_latency,
    )
    union_best_agentic = max(
        (h.agentic_max_viable_concurrency for h in eligible_agentic_union),
        default=best_agentic_c,
    )
    scores = _compute_scores(
        result, union_best_agentic, union_best_tp, union_best_lat,
        pareto, agentic_pareto,
    )

    # Check hard stop conditions
    hard_stop = False
    stop_reason = None

    if exp_count >= config.experiments.max_experiments:
        hard_stop = True
        stop_reason = f"Budget exhausted ({exp_count}/{config.experiments.max_experiments})"

    # Setup-bug guard: if the last 3 experiments all failed with the same
    # failure_classification, this is almost certainly a configuration or
    # environment problem (HF cache layout, missing image, wrong tokenizer,
    # bad weights path) — not a hyperparameter the planner can tune. Stop
    # the loop early instead of burning the rest of the experiment budget
    # producing identical crashes that the LLM will keep guessing about.
    if not hard_stop:
        tail = session_with_current[-3:]
        if len(tail) == 3 and all(h.status != ExperimentStatus.SUCCESS for h in tail):
            tail_classes = {h.failure_classification for h in tail}
            if len(tail_classes) == 1 and None not in tail_classes:
                rc = tail_classes.pop()
                # Generic "startup_crash" is NOT specific enough to stop on —
                # that's the catch-all when no fatal pattern matched. Only
                # trip the guard on a concrete classification (oom,
                # weights_not_found, tokenizer_init_failed, …).
                if rc not in {"startup_crash", "healthcheck_timeout"}:
                    hard_stop = True
                    stop_reason = (
                        f"3 consecutive failures with identical "
                        f"root_cause='{rc}'. This is a setup problem, not "
                        f"a hyperparameter problem — fix it and re-run. "
                        f"Inspect storage.logs_dir for the full container "
                        f"logs of each attempt."
                    )

    # Plateau uses ONLY the current session — loaded tops would make plateau
    # trip on iteration 1 (newcomers struggle to beat historical bests).
    # Pass the PRE-update bests; see _check_plateau docstring for rationale.
    if not hard_stop and _check_plateau(
        session_with_current,
        best_agentic_c,
        best_latency,
        config.experiments.plateau_window,
        config.experiments.plateau_threshold,
    ):
        hard_stop = True
        stop_reason = f"Plateau: no improvement in last {config.experiments.plateau_window} experiments"

    # Now fold the current experiment into the session leaderboards.
    latency_threshold = config.benchmark.latency_threshold_ms
    is_eligible_throughput = (
        result.status == ExperimentStatus.SUCCESS
        and result.correctness_gate_passed
        and tp > 0 and lat > 0
    )
    is_eligible_agentic = (
        result.status == ExperimentStatus.SUCCESS
        and result.correctness_gate_passed
        and agentic_c_now > 0
    )

    # best_agentic — higher concurrency wins; ties broken by lower tpot p95.
    if is_eligible_agentic:
        better = agentic_c_now > best_agentic_c or (
            agentic_c_now == best_agentic_c
            and 0 < agentic_tpot_now < best_agentic_tpot
        )
        if better:
            best_agentic_c = agentic_c_now
            best_agentic_id = result.experiment_id
            best_agentic_tpot = agentic_tpot_now or float("inf")
            best_agentic_tp = agentic_peak_now

    # best_throughput stays tracked for historical/dashboard use even though
    # it does not drive leaderboards or stop decisions anymore.
    if is_eligible_throughput and tp > best_throughput:
        best_throughput = tp
        best_throughput_id = result.experiment_id

    if is_eligible_throughput and 0 < lat < best_latency:
        best_latency = lat
        best_latency_id = result.experiment_id

    # best_balanced: best agentic concurrency under TTFT-threshold gate.
    if (
        is_eligible_agentic
        and 0 < lat < latency_threshold
        and agentic_c_now > best_balanced_tp
    ):
        best_balanced_tp = agentic_c_now  # field name kept for compat; semantics: best agentic c under SLO
        best_balanced_lat = lat
        best_balanced_id = result.experiment_id

    # Ask LLM for analysis using the agentic-first leaderboard.

    # Prepare leaderboard data — uses the union so the LLM sees prior-run
    # tops alongside the current session.
    eligible_agentic = [h for h in union_history if _is_agentic_eligible(h)]
    eligible_throughput = [h for h in union_history if _is_eligible(h)]

    # Primary: max viable agentic c DESC, then agentic_tpot ASC, then
    # agentic_peak_throughput DESC.
    sorted_by_agentic = sorted(
        eligible_agentic,
        key=lambda h: (
            -h.agentic_max_viable_concurrency,
            h.agentic_tpot_p95 or float("inf"),
            -h.agentic_peak_throughput,
        ),
    )[:5]

    sorted_by_lat = sorted(
        eligible_throughput,
        key=lambda h: h.low_concurrency_ttft_p95,
    )[:5]

    # Balanced: agentic concurrency wins UNDER a TTFT-threshold gate.
    balanced_candidates = [
        h for h in eligible_agentic
        if 0 < h.low_concurrency_ttft_p95 < latency_threshold
    ]
    sorted_balanced = sorted(
        balanced_candidates,
        key=lambda h: h.agentic_max_viable_concurrency,
        reverse=True,
    )[:5]

    def _format_agentic_leaderboard(items: list[ExperimentSummary]) -> str:
        if not items:
            return "None yet"
        lines = []
        for h in items:
            lines.append(
                f"  {h.experiment_id} ({h.engine.value}): "
                f"max_viable_agents={h.agentic_max_viable_concurrency}, "
                f"agentic_tpot_p95={h.agentic_tpot_p95:.1f}ms, "
                f"agentic_ttft_p95={h.agentic_ttft_p95:.0f}ms, "
                f"agentic_throughput={h.agentic_peak_throughput:.1f} tok/s, "
                f"config={json.dumps(h.config_digest)}"
            )
        return "\n".join(lines)

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

    # Compact per-phase failure view — same shape as ExperimentSummary's,
    # but built directly from result.errors so it's available even before
    # ExperimentSummary is constructed by the reporter.
    failed_phases_view = []
    for err in result.errors:
        if err.stage != "benchmark_phase":
            continue
        d = err.details or {}
        samples = d.get("error_samples") or []
        failed_phases_view.append({
            "phase": str(d.get("phase_id", "")),
            "workload": str(d.get("workload_id", "")),
            "concurrency": int(d.get("concurrency") or 0),
            "prompt_length": int(d.get("prompt_length") or 0),
            "error_rate": round(float(d.get("error_rate") or 0.0), 3),
            "errors": int(d.get("errors") or 0),
            "sample_error": str(samples[0])[:240] if samples else err.message[:240],
        })

    latest_payload: dict = {
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
    }
    if result.failure_classification:
        latest_payload["failure_classification"] = result.failure_classification
    if result.error:
        # Cap to avoid blowing the prompt; failed_phases below carry structure.
        latest_payload["error"] = result.error[:1200]
    if failed_phases_view:
        latest_payload["failed_phases"] = failed_phases_view

    # Agentic ceiling-probe view: deliberate sweep, NOT failures.
    probed = sorted({p.concurrency for p in result.ceiling_probe_phases})
    viable = sorted({
        r.concurrency for r in result.benchmark.concurrency_results
        if r.workload_id == "agentic_long_context"
    })
    if probed or viable:
        latest_payload["agentic"] = {
            "max_viable_concurrency": result.benchmark.max_viable_agentic_concurrency,
            "probed": sorted(set(probed) | set(viable)),
            "viable": viable,
            "ceiling_probed": probed,
        }

    prompt = ANALYZER_SYSTEM_PROMPT.format(
        latest_json=json.dumps(latest_payload, indent=2, default=str),
        top_agentic=_format_agentic_leaderboard(sorted_by_agentic),
        top_latency=_format_leaderboard(sorted_by_lat),
        top_balanced=_format_agentic_leaderboard(sorted_balanced),
        latency_threshold=latency_threshold,
        agentic_pareto_front=json.dumps(
            [p.model_dump() for p in agentic_pareto], indent=2, default=str,
        ),
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
        "best_agentic_max_viable_c": best_agentic_c,
        "best_agentic_config_id": best_agentic_id,
        "best_agentic_tpot_p95": best_agentic_tpot,
        "best_agentic_throughput": best_agentic_tp,
        "best_throughput": best_throughput,
        "best_throughput_config_id": best_throughput_id,
        "best_latency_ttft_p95": best_latency,
        "best_latency_config_id": best_latency_id,
        "best_balanced_config_id": best_balanced_id,
        "best_balanced_throughput": best_balanced_tp,
        "best_balanced_latency": best_balanced_lat,
        "pareto_front": pareto,
        "agentic_pareto_front": agentic_pareto,
        "next_optimization_goal": next_goal,
        "status": status,
        "stop_reason": stop_reason,
        "current_result": enriched_result,
    }
