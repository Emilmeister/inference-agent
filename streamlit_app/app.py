"""Streamlit dashboard for visualizing inference benchmark experiments.

Source: the inference-api REST service (configured via INFERENCE_API_URL +
INFERENCE_API_TOKEN). The dashboard never touches Postgres directly.

Run: `streamlit run streamlit_app/app.py`
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api import (
    Filters,
    HardwareKey,
    delete_experiments,
    get_experiment_payload,
    list_agentic_turn_metrics,
    list_distinct_engines,
    list_distinct_hardware,
    list_distinct_models,
    list_experiment_phases,
    list_experiment_summaries,
)


st.set_page_config(
    page_title="Inference Benchmark Dashboard",
    layout="wide",
)

st.title("LLM Inference Benchmark Dashboard")


# ---- Helpers ----


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_metric(value: float, suffix: str = "", precision: int = 0) -> str:
    if value <= 0:
        return "n/a"
    return f"{value:,.{precision}f}{suffix}"


def _pareto_ids(frame: pd.DataFrame) -> list[str]:
    """Return non-dominated ids for throughput-vs-latency."""
    valid = frame[(frame["peak_throughput"] > 0) & (frame["ttft_p95"] > 0)]
    ids: list[str] = []
    best_latency = float("inf")
    for _, row in valid.sort_values("peak_throughput", ascending=False).iterrows():
        if row["ttft_p95"] < best_latency:
            ids.append(str(row["experiment_id"]))
            best_latency = row["ttft_p95"]
    return ids


def _agentic_pareto_ids(frame: pd.DataFrame) -> list[str]:
    """Return non-dominated ids in (max_viable_agentic_c ↑, agentic_tpot_p95 ↓).

    The primary front under the agentic-first goal — a point stays on the
    front iff no other point has both more parallel agents AND lower per-
    token latency at the max-viable phase.
    """
    if "max_viable_agentic_concurrency" not in frame.columns:
        return []
    valid = frame[
        (frame["max_viable_agentic_concurrency"] > 0)
        & (frame.get("agentic_tpot_p95", pd.Series(dtype=float)) > 0)
    ]
    if valid.empty:
        return []
    ids: list[str] = []
    best_tpot = float("inf")
    for _, row in valid.sort_values(
        "max_viable_agentic_concurrency", ascending=False,
    ).iterrows():
        if row["agentic_tpot_p95"] < best_tpot:
            ids.append(str(row["experiment_id"]))
            best_tpot = row["agentic_tpot_p95"]
    return ids


def _summary_label(row: pd.Series) -> str:
    return (
        f"{row['experiment_id']} ({row['engine']} "
        f"TP={row['tp']} q={row['quantization']}) - "
        f"{row['peak_throughput']:.0f} tok/s"
    )


def _baseline_row(frame: pd.DataFrame) -> pd.Series | None:
    """Return the most-recent baseline row in `frame`, or None.

    Baseline rows carry is_baseline=True (operator anchor). When several exist
    (re-runs over time) the newest by timestamp wins so the dashboard always
    measures impact against the latest reference.
    """
    if frame.empty or "is_baseline" not in frame.columns:
        return None
    base = frame[frame["is_baseline"] == True]  # noqa: E712 — pandas mask
    if base.empty:
        return None
    if "timestamp" in base.columns and base["timestamp"].notna().any():
        return base.sort_values("timestamp", ascending=False).iloc[0]
    return base.iloc[0]


def _baseline_badge(row: pd.Series) -> str:
    """Short marker prefix for a row that is the baseline (else empty)."""
    return "⭐ " if bool(row.get("is_baseline", False)) else ""


def _flatten_dict(d: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict for diffing. Lists become tuples (hashable, comparable)."""
    out: dict[str, Any] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        elif isinstance(v, list):
            out[key] = tuple(v)
        else:
            out[key] = v
    return out


_FLAG_PRESENT = "✓ (set)"  # marker for a valueless boolean CLI flag


def _parse_cli_tokens(tokens: Any) -> dict[str, Any]:
    """Parse a flat CLI token list into {flag: value} pairs (order-independent).

    `--flag value`  → {"--flag": "value"}
    `--flag=value`  → {"--flag": "value"}
    `--bool-flag`   → {"--bool-flag": _FLAG_PRESENT}   (no following value)

    A token is a flag if it starts with "-"; the next token is its value unless
    it also starts with "-" (then the flag is valueless). This lets the Compare
    tab diff engine args flag-by-flag instead of comparing the whole ordered
    list as one opaque string — so reordered-but-equal arg lists show no diff,
    and only the genuinely added/removed/changed flags surface.
    """
    out: dict[str, Any] = {}
    if not tokens:
        return out
    toks = [str(t) for t in tokens]
    i, n = 0, len(toks)
    while i < n:
        tok = toks[i]
        if tok.startswith("-"):
            if tok.startswith("--") and "=" in tok:
                key, _, val = tok.partition("=")
                out[key] = val
                i += 1
                continue
            if i + 1 < n and not toks[i + 1].startswith("-"):
                out[tok] = toks[i + 1]
                i += 2
            else:
                out[tok] = _FLAG_PRESENT
                i += 1
        else:
            # Stray positional token (shouldn't happen for engine args) — keep
            # it visible rather than silently dropping.
            out[f"<positional #{i}>"] = tok
            i += 1
    return out


_COMPARE_IGNORE_KEYS = frozenset({
    # LLM-генерируемый текст — всегда разный, шум в diff
    "rationale",
    # Уникальный id эксперимента — различается ВСЕГДА, бесполезен в diff
    "experiment_id",
    # Метка бейзлайна — это метаданные, не knob запуска; уже показана
    # звёздочкой ⭐ в лейблах, в diff только шумит ("True" vs "—").
    "is_baseline",
})

# nerdctl-level flags from build_common_container_args — container plumbing, not
# engine knobs.
_NERDCTL_FLAGS = frozenset({"--name", "--gpus", "--shm-size", "--network"})
# Injected by the engine builder for EVERY run (identical) — not a tunable knob.
_INJECTED_ENGINE_FLAGS = frozenset({"--model", "--model-path", "--host", "--port"})
# Env the harness force-adds to every container (offline mode / token) — not a
# user/LLM knob, so excluded from the env diff.
_HARNESS_ENV = frozenset({"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_TOKEN"})


def _engine_flags_from_argv(container_args: Any) -> dict[str, Any]:
    """Extract the real, post-dedup engine serve flags from the launched argv.

    ``container_args`` is the full ``nerdctl run ... <image> <serve flags>`` token
    list that was actually executed — after ``dedup_flags`` merged config.yaml's
    always-appended ``vllm_extra_args`` into the per-experiment ones. Diffing THIS
    instead of ``config.extra_engine_args`` is what kills the phantom diffs:

      * ``--reasoning-parser`` / ``--tool-call-parser`` / ``--enable-auto-tool-choice``
        are appended from config.yaml to every run, so the baseline omits them
        from its stored config while planned runs carry them — but in the real
        command both have them.
      * ``--speculative-config`` lives in the baseline's raw extra_engine_args but
        in a structured field for planned runs — same flag, different slot.

    Sourcing from the executed command makes both sides line up. Drops nerdctl
    plumbing, the injected model/host/port, and ``-e`` env tokens (env is diffed
    separately from ``config.extra_env``).
    """
    flags: dict[str, Any] = {}
    if not container_args:
        return flags
    toks = [str(t) for t in container_args]
    i, n = 0, len(toks)
    while i < n:
        tok = toks[i]
        if tok.startswith("--"):
            if "=" in tok:
                key, _, val = tok.partition("=")
                consume = 1
            elif i + 1 < n and not toks[i + 1].startswith("-"):
                key, val, consume = tok, toks[i + 1], 2
            else:
                key, val, consume = tok, _FLAG_PRESENT, 1
            if key not in _NERDCTL_FLAGS and key not in _INJECTED_ENGINE_FLAGS:
                flags[key] = val
            i += consume
        else:
            # positionals (nerdctl/run/image/python3/...) and short flags
            # (-d, -v <vol>, -e <KEY=VAL>, -m <mod>) — not engine knobs.
            i += 1
    return flags


def _env_inputs_from_config(cfg: dict) -> dict[str, Any]:
    """User/LLM-set container env, excluding harness-injected vars.

    config.extra_env is a faithful source (config.yaml appends only engine args,
    never env), so unlike extra_engine_args it does not suffer the slot-mismatch
    problem and can be diffed directly.
    """
    env = cfg.get("extra_env") or {}
    return {f"env {k}": v for k, v in env.items() if k not in _HARNESS_ENV}


def _runtime_inputs(payload: dict) -> dict[str, Any]:
    """Snapshot of fields that describe HOW the run was launched (inputs, not measurements).

    Engine flags come from the actually-executed ``container_args`` (ground truth
    after dedup), so the same setting stored in different slots (config field vs
    raw extra_engine_args vs config.yaml append) compares equal on both sides.
    Falls back to the legacy config-field view for experiments recorded before
    ``container_args`` was captured.
    """
    if not payload:
        return {}
    out: dict[str, Any] = {
        "engine": payload.get("engine"),
        "engine_version": payload.get("engine_version"),
        "model": payload.get("model"),
        "container_image_digest": payload.get("container_image_digest"),
        "benchmark_seed": payload.get("benchmark_seed"),
    }
    cfg = dict(payload.get("config") or {})
    container_args = payload.get("container_args")
    if container_args:
        # Engine flags from the real command; env from the faithful config field.
        for flag, val in _engine_flags_from_argv(container_args).items():
            out[f"engine_arg {flag}"] = val
        out.update(_env_inputs_from_config(cfg))
    else:
        # Legacy fallback: structured config fields + per-flag extra_engine_args.
        # Suffers the slot-mismatch phantom diffs, but it's the best we have for
        # experiments predating container_args capture.
        engine_args = cfg.pop("extra_engine_args", None)
        flat_cfg = _flatten_dict(cfg)
        for k in _COMPARE_IGNORE_KEYS:
            flat_cfg.pop(k, None)
        out.update(flat_cfg)
        for flag, val in _parse_cli_tokens(engine_args).items():
            out[f"engine_arg {flag}"] = val
    return out


def _diff_runtime_inputs(a: dict, b: dict) -> list[dict[str, Any]]:
    """Return rows {field, A, B} for keys where the two runtime snapshots differ."""
    flat_a = _runtime_inputs(a)
    flat_b = _runtime_inputs(b)
    keys = sorted(set(flat_a) | set(flat_b))
    out: list[dict[str, Any]] = []
    for k in keys:
        va = flat_a.get(k, "<missing>")
        vb = flat_b.get(k, "<missing>")
        if va != vb:
            out.append({
                "field": k,
                "A": "—" if va == "<missing>" else va,
                "B": "—" if vb == "<missing>" else vb,
            })
    return out


def _build_search_widget(field: str, series: pd.Series, key_prefix: str) -> pd.Series:
    """Render a Streamlit filter widget appropriate for `series`; return a boolean mask.

    Bool        → radio any/true/false.
    Numeric     → multiselect if ≤12 distinct values, else range slider.
    Categorical → multiselect of distinct values.
    """
    key = f"{key_prefix}_{field}"
    non_null = series.dropna()
    if non_null.empty:
        st.caption(f"{field}: all values are missing.")
        return pd.Series(True, index=series.index)

    if pd.api.types.is_bool_dtype(series) or set(non_null.unique()) <= {True, False}:
        choice = st.radio(field, ["any", "true", "false"], horizontal=True, key=key)
        if choice == "any":
            return pd.Series(True, index=series.index)
        return series == (choice == "true")

    if pd.api.types.is_numeric_dtype(series):
        uniq = sorted(non_null.unique().tolist())
        if len(uniq) <= 12:
            sel = st.multiselect(field, uniq, default=uniq, key=key)
            return series.isin(sel)
        lo, hi = float(non_null.min()), float(non_null.max())
        if lo == hi:
            st.caption(f"{field}: single value = {lo}")
            return pd.Series(True, index=series.index)
        sel_range = st.slider(
            field, min_value=lo, max_value=hi, value=(lo, hi), key=key,
        )
        return series.between(sel_range[0], sel_range[1], inclusive="both")

    options = sorted(non_null.astype(str).unique().tolist())
    sel = st.multiselect(field, options, default=options, key=key)
    return series.astype(str).isin(sel)


# ---- Source filters (inference-api backed) ----


st.sidebar.header("Source")

try:
    hardware_options = list_distinct_hardware()
    model_options = list_distinct_models()
    engine_options = list_distinct_engines()
except Exception as exc:  # pragma: no cover - surfaced to user via UI
    st.error(f"Failed to query inference-api: {exc}")
    st.stop()

if not hardware_options:
    st.info("No experiments in the database yet. Run `inference-agent` to populate it.")
    st.stop()

# Primary selectors — model and context window first, hardware/engine after.
# The benchmarking story usually starts with "which model on which context",
# everything else is a refinement of that.
selected_models = st.sidebar.multiselect(
    "Model (primary filter)",
    model_options,
    default=model_options,
)
selected_engines_src = st.sidebar.multiselect(
    "Engine (source filter)",
    engine_options,
    default=engine_options,
)

hw_labels = {hw.label(): hw for hw in hardware_options}
hw_label = st.sidebar.selectbox("Hardware", list(hw_labels.keys()))
selected_hw: HardwareKey = hw_labels[hw_label]

filters = Filters(
    hardware=selected_hw,
    models=tuple(selected_models),
    engines=tuple(selected_engines_src),
)

df = list_experiment_summaries(filters)

if df.empty:
    st.warning("No experiments match the current filters.")
    st.stop()

# Context-window picker built from the LOADED rows (REST already filtered
# by hardware/model/engine). Sits right next to the model selector so the
# operator can lock both invariants before exploring other axes.
ctx_values_all = sorted(
    int(v) for v in df["max_model_len"].dropna().unique().tolist() if v > 0
) if "max_model_len" in df.columns else []
if ctx_values_all:
    selected_ctx = st.sidebar.multiselect(
        "Context window (max_model_len)",
        ctx_values_all,
        default=ctx_values_all,
        help=(
            "Across-experiment invariant in this project — fix it in config.yaml "
            "to keep the planner from varying it, then use this filter to "
            "compare runs at the same context."
        ),
    )
else:
    selected_ctx = []

st.success(f"Loaded {len(df)} experiments from inference-api")


# ---- Display filters (in-page narrowing of the loaded set) ----


st.sidebar.header("Display")

engines = st.sidebar.multiselect(
    "Engine",
    sorted(df["engine"].dropna().unique().tolist()),
    default=sorted(df["engine"].dropna().unique().tolist()),
)
statuses = st.sidebar.multiselect(
    "Status",
    sorted(df["status"].dropna().unique().tolist()),
    default=sorted(df["status"].dropna().unique().tolist()),
)
quants = st.sidebar.multiselect(
    "Quantization",
    sorted(df["quantization"].dropna().unique().tolist()),
    default=sorted(df["quantization"].dropna().unique().tolist()),
)
tp_values = st.sidebar.multiselect(
    "Tensor Parallel",
    sorted(df["tp"].dropna().unique().tolist()),
    default=sorted(df["tp"].dropna().unique().tolist()),
)
eligible_only = st.sidebar.checkbox(
    "Eligible only",
    value=False,
    help="status=success, correctness gate passed, and headline metrics present.",
)
latency_threshold_ms = st.sidebar.number_input(
    "Balanced latency threshold (ms)",
    min_value=1,
    value=500,
    step=25,
)

filtered = df[
    df["engine"].isin(engines)
    & df["status"].isin(statuses)
    & df["quantization"].isin(quants)
    & df["tp"].isin(tp_values)
].copy()
if selected_ctx and "max_model_len" in filtered.columns:
    filtered = filtered[filtered["max_model_len"].isin(selected_ctx)]

eligible_mask = (
    (filtered["status"] == "success")
    & filtered["correctness_gate_passed"]
    & (filtered["peak_throughput"] > 0)
    & (filtered["ttft_p95"] > 0)
)
if eligible_only:
    filtered = filtered[eligible_mask].copy()
    eligible_mask = pd.Series(True, index=filtered.index)

filtered_ids = tuple(sorted(filtered["experiment_id"].astype(str)))
phase_df = list_experiment_phases(filtered_ids)
eligible = filtered[
    (filtered["status"] == "success")
    & filtered["correctness_gate_passed"]
    & (filtered["peak_throughput"] > 0)
    & (filtered["ttft_p95"] > 0)
].copy()

if filtered.empty:
    st.warning("No experiments match the current display filters.")
    st.stop()


# ---- Dashboard tabs ----


tabs = st.tabs([
    "Overview",
    "Pareto",
    "Workloads",
    "Correctness",
    "Config Impact",
    "GPU Efficiency",
    "Reproduce",
    "Tables",
    "Search",
    "Compare",
    "Agentic",
    "Manage",
])


with tabs[0]:
    st.header("Executive Overview")
    st.caption(
        "**Primary goal: maximize parallel agents under SLO.** Raw-throughput "
        "panels below are informational only — the agent does NOT optimize for "
        "peak_throughput@c=256, p=512 anymore."
    )

    # Agentic eligibility: success + correctness + non-zero max_viable_c.
    agentic_eligible = (
        eligible[eligible.get("max_viable_agentic_concurrency", 0) > 0].copy()
        if "max_viable_agentic_concurrency" in eligible.columns
        else pd.DataFrame()
    )

    best_agentic_row = (
        agentic_eligible.sort_values(
            ["max_viable_agentic_concurrency", "agentic_tpot_p95"],
            ascending=[False, True],
        ).iloc[0]
        if not agentic_eligible.empty
        else None
    )
    best_lat_row = eligible.nsmallest(1, "ttft_p95")
    best_tp_row = eligible.nlargest(1, "peak_throughput")
    agentic_pareto_count = len(_agentic_pareto_ids(agentic_eligible))

    # ── Impact vs baseline ──────────────────────────────────────────────
    # Find the operator baseline in the FULL loaded set (df), not the
    # display-filtered one, so display filters can't hide the anchor. Compare
    # the best agent-found config against it on the primary axes.
    baseline_row = _baseline_row(df)
    if baseline_row is not None:
        st.subheader("⭐ Impact vs baseline")
        st.caption(
            f"Baseline experiment `{baseline_row['experiment_id']}` — the "
            "operator anchor the agent improves upon. Deltas are best-found vs "
            "baseline."
        )

        def _impact(label, best_val, base_val, suffix, prec, lower_better):
            base_val = _as_float(base_val)
            best_val = _as_float(best_val)
            if base_val <= 0 and best_val <= 0:
                return {"metric": label, "baseline": "n/a", "best agent": "n/a", "impact": "—"}
            delta = best_val - base_val
            if base_val > 0:
                pct = (delta / base_val) * 100.0
                improved = (delta < 0) if lower_better else (delta > 0)
                sign = "+" if delta > 0 else ""
                arrow = "✅" if improved else ("➖" if delta == 0 else "🔻")
                impact = f"{arrow} {sign}{delta:.{prec}f}{suffix} ({sign}{pct:.0f}%)"
            else:
                impact = "new"
            return {
                "metric": label,
                "baseline": _format_metric(base_val, suffix, prec),
                "best agent": _format_metric(best_val, suffix, prec),
                "impact": impact,
            }

        best_agents_val = (
            _as_float(best_agentic_row["max_viable_agentic_concurrency"])
            if best_agentic_row is not None else 0.0
        )
        best_agentic_tpot_val = (
            _as_float(best_agentic_row["agentic_tpot_p95"])
            if best_agentic_row is not None else 0.0
        )
        impact_rows = [
            _impact(
                "Max parallel agents (PRIMARY)",
                best_agents_val,
                baseline_row.get("max_viable_agentic_concurrency", 0),
                "", 0, False,
            ),
            _impact(
                "Agentic tpot p95",
                best_agentic_tpot_val,
                baseline_row.get("agentic_tpot_p95", 0),
                " ms", 1, True,
            ),
            _impact(
                "Cold-start TTFT p95",
                best_lat_row["ttft_p95"].iloc[0] if not best_lat_row.empty else 0,
                baseline_row.get("ttft_p95", 0),
                " ms", 1, True,
            ),
            _impact(
                "Raw throughput (info)",
                best_tp_row["peak_throughput"].iloc[0] if not best_tp_row.empty else 0,
                baseline_row.get("peak_throughput", 0),
                " tok/s", 0, False,
            ),
        ]
        st.dataframe(
            pd.DataFrame(impact_rows),
            use_container_width=True,
            hide_index=True,
        )

    # ── Primary KPI row: agentic-first ──────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        value = int(best_agentic_row["max_viable_agentic_concurrency"]) if best_agentic_row is not None else 0
        suffix = "+" if (best_agentic_row is not None and bool(best_agentic_row.get("agentic_ceiling_hit", False))) else ""
        st.metric(
            "Max parallel agents (primary)",
            f"{value}{suffix}" if value > 0 else "n/a",
            help=(
                "Headline metric of the agentic-first goal: largest concurrency "
                "where TTFT p95, tpot p95 AND session error_rate all meet SLO. "
                "'+' means the sweep ceiling was hit; the true value may be "
                "higher."
            ),
        )
    with col2:
        value = _as_float(best_agentic_row["agentic_tpot_p95"]) if best_agentic_row is not None else 0
        st.metric(
            "Best agentic tpot p95",
            _format_metric(value, " ms", precision=1),
            help="Per-token latency at the max-viable concurrency phase.",
        )
    with col3:
        value = best_lat_row["ttft_p95"].iloc[0] if not best_lat_row.empty else 0
        st.metric("Best cold-start latency", _format_metric(value, " ms", precision=1))
    with col4:
        st.metric("Agentic Pareto points", str(agentic_pareto_count))

    # ── Informational KPIs: throughput / counts ─────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        value = best_tp_row["peak_throughput"].iloc[0] if not best_tp_row.empty else 0
        st.metric(
            "Raw throughput (info only)",
            _format_metric(value, " tok/s"),
            help=(
                "Best peak_output_tokens_per_sec — typically a c=256/p=512 "
                "synthetic phase. NOT a leadership signal in the agentic-first "
                "regime."
            ),
        )
    with col2:
        st.metric("Displayed experiments", str(len(filtered)))
    with col3:
        st.metric("Eligible experiments", str(len(eligible)))
    with col4:
        failed = int((filtered["status"] != "success").sum())
        st.metric("Non-success runs", str(failed))

    # ── Leaderboards: agentic, latency, balanced (in that order) ────────
    st.subheader("Leaderboards")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("Agentic objective — max parallel agents under SLO (primary)")
        if not agentic_eligible.empty:
            cols = [
                "experiment_id",
                "engine",
                "quantization",
                "tp",
                "max_viable_agentic_concurrency",
                "agentic_tpot_p95",
                "agentic_ttft_p95",
                "agentic_peak_throughput",
                "is_agentic_pareto",
            ]
            present = [c for c in cols if c in agentic_eligible.columns]
            top = (
                agentic_eligible
                .sort_values(
                    ["max_viable_agentic_concurrency", "agentic_tpot_p95"],
                    ascending=[False, True],
                )
                .head(5)[present]
                .rename(columns={
                    "max_viable_agentic_concurrency": "max_agents",
                    "agentic_peak_throughput": "agentic_tok/s",
                })
            )
            st.dataframe(top, use_container_width=True, hide_index=True)
        else:
            st.info("No experiments passed the agentic SLO yet.")

    with col2:
        st.caption("Latency objective — lowest cold-start TTFT p95")
        latency_cols = [
            "experiment_id", "engine", "quantization", "tp",
            "peak_throughput", "ttft_p95",
            "peak_throughput_e2e_cv", "low_concurrency_ttft_cv",
        ]
        st.dataframe(
            eligible.nsmallest(5, "ttft_p95")[latency_cols],
            use_container_width=True,
            hide_index=True,
        )

    with col3:
        st.caption(f"Balanced — max agents with TTFT p95 < {latency_threshold_ms}ms")
        if not agentic_eligible.empty:
            bal_pool = agentic_eligible[
                (agentic_eligible["ttft_p95"] > 0)
                & (agentic_eligible["ttft_p95"] < latency_threshold_ms)
            ]
            if not bal_pool.empty:
                bal_cols = [
                    "experiment_id", "engine", "quantization", "tp",
                    "max_viable_agentic_concurrency",
                    "ttft_p95",
                    "agentic_tpot_p95",
                ]
                present = [c for c in bal_cols if c in bal_pool.columns]
                st.dataframe(
                    bal_pool.nlargest(5, "max_viable_agentic_concurrency")[present]
                    .rename(columns={"max_viable_agentic_concurrency": "max_agents"}),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No agentic-eligible config under the latency threshold yet.")
        else:
            st.info("No agentic-eligible runs yet.")

    # ── Timeline: max agents over time (was throughput) ─────────────────
    st.subheader("Best-So-Far Timeline")
    timeline = filtered.dropna(subset=["timestamp"]).sort_values("timestamp").copy()
    timeline = timeline[
        (timeline["status"] == "success")
        & timeline["correctness_gate_passed"]
    ]
    if "max_viable_agentic_concurrency" in timeline.columns and not timeline.empty:
        timeline["best_agents_so_far"] = (
            timeline["max_viable_agentic_concurrency"].cummax()
        )
        ttft_valid = timeline["ttft_p95"].where(timeline["ttft_p95"] > 0)
        timeline["best_latency_so_far"] = ttft_valid.cummin()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timeline["timestamp"],
            y=timeline["best_agents_so_far"],
            mode="lines+markers",
            name="Best max-parallel agents so far (primary)",
            yaxis="y",
        ))
        fig.add_trace(go.Scatter(
            x=timeline["timestamp"],
            y=timeline["best_latency_so_far"],
            mode="lines+markers",
            name="Best TTFT p95 so far",
            yaxis="y2",
        ))
        fig.update_layout(
            height=420,
            yaxis=dict(title="Max parallel agents"),
            yaxis2=dict(title="TTFT p95 ms", overlaying="y", side="right"),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No timestamped eligible runs available for the timeline.")


with tabs[1]:
    st.header("Pareto Explorer")

    pareto_source = eligible if eligible_only else filtered

    # ── Agentic Pareto (primary) ────────────────────────────────────────
    st.subheader("Agentic — max parallel agents vs tpot p95 (PRIMARY front)")
    st.caption(
        "Configurations on this front cannot be beaten on both axes "
        "simultaneously — more parallel agents AND lower per-token latency. "
        "Up-and-left is better."
    )
    agentic_valid = (
        pareto_source[
            (pareto_source.get("max_viable_agentic_concurrency", 0) > 0)
            & (pareto_source.get("agentic_tpot_p95", 0) > 0)
        ].copy()
        if "max_viable_agentic_concurrency" in pareto_source.columns
        else pd.DataFrame()
    )
    if not agentic_valid.empty:
        agentic_valid["marker_size"] = agentic_valid["tp"].clip(lower=1)
        fig = px.scatter(
            agentic_valid,
            x="agentic_tpot_p95",
            y="max_viable_agentic_concurrency",
            color="engine",
            symbol="quantization",
            size="marker_size",
            hover_data=[
                "experiment_id", "status", "correctness_gate_passed",
                "agentic_ttft_p95", "agentic_peak_throughput",
                "prefix_caching", "scheduling_policy",
            ],
            labels={
                "agentic_tpot_p95": "Agentic tpot p95 (ms) — lower is better",
                "max_viable_agentic_concurrency": "Max parallel agents (under SLO)",
            },
        )
        pareto_ids = _agentic_pareto_ids(agentic_valid)
        pareto_pts = agentic_valid[agentic_valid["experiment_id"].isin(pareto_ids)]
        if not pareto_pts.empty:
            pareto_sorted = pareto_pts.sort_values("agentic_tpot_p95")
            fig.add_trace(go.Scatter(
                x=pareto_sorted["agentic_tpot_p95"],
                y=pareto_sorted["max_viable_agentic_concurrency"],
                mode="lines+markers",
                name="Agentic Pareto front",
                line=dict(color="red", dash="dash", width=2),
                marker=dict(size=10, symbol="star"),
            ))
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Agentic Pareto points**")
        st.dataframe(
            pareto_pts.sort_values(
                ["max_viable_agentic_concurrency", "agentic_tpot_p95"],
                ascending=[False, True],
            )[
                [c for c in [
                    "experiment_id", "engine", "quantization", "tp",
                    "max_viable_agentic_concurrency",
                    "agentic_tpot_p95",
                    "agentic_ttft_p95",
                    "agentic_peak_throughput",
                    "prefix_caching",
                ] if c in pareto_pts.columns]
            ].rename(columns={"max_viable_agentic_concurrency": "max_agents"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No experiments cleared the agentic SLO with a measured tpot p95 "
            "yet. Run a config with `enable_agentic_long_context: true`."
        )

    # ── Throughput vs Latency Pareto (informational / historical) ──────
    st.subheader("Throughput vs Latency (informational)")
    st.caption(
        "Kept for backward compatibility and the synthetic-throughput view. "
        "Not the active leadership criterion — best raw throughput often comes "
        "from c=256, p=512 phases that fail the agentic SLO."
    )
    valid = pareto_source[
        (pareto_source["peak_throughput"] > 0)
        & (pareto_source["ttft_p95"] > 0)
    ].copy()

    if not valid.empty:
        valid["marker_size"] = valid["tp"].clip(lower=1)
        fig = px.scatter(
            valid,
            x="ttft_p95",
            y="peak_throughput",
            color="engine",
            symbol="quantization",
            size="marker_size",
            hover_data=[
                "experiment_id",
                "status",
                "correctness_gate_passed",
                "scheduling_policy",
                "chunked_prefill",
                "prefix_caching",
                "peak_throughput_e2e_cv",
                "low_concurrency_ttft_cv",
            ],
            labels={
                "ttft_p95": "TTFT p95 (ms) - lower is better",
                "peak_throughput": "Peak output throughput (tok/s)",
            },
        )

        pareto_pts = valid[valid["experiment_id"].isin(_pareto_ids(valid))]
        if not pareto_pts.empty:
            pareto_sorted = pareto_pts.sort_values("ttft_p95")
            fig.add_trace(go.Scatter(
                x=pareto_sorted["ttft_p95"],
                y=pareto_sorted["peak_throughput"],
                mode="lines+markers",
                name="Pareto front",
                line=dict(color="red", dash="dash", width=2),
                marker=dict(size=10, symbol="star"),
            ))

        fig.add_vline(
            x=latency_threshold_ms,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Balanced threshold: {latency_threshold_ms} ms",
        )
        fig.update_layout(height=560)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Pareto Points")
        st.dataframe(
            pareto_pts.sort_values(["ttft_p95", "peak_throughput"], ascending=[True, False])[
                [
                    "experiment_id",
                    "engine",
                    "quantization",
                    "tp",
                    "peak_throughput",
                    "ttft_p95",
                    "balanced_score",
                    "peak_throughput_e2e_cv",
                    "low_concurrency_ttft_cv",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No valid data points for Pareto chart.")


with tabs[2]:
    st.header("Workload Breakdown")

    if phase_df.empty:
        st.info("No per-phase benchmark data available.")
    else:
        max_error_rate = st.slider(
            "Max phase error rate",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
        )
        phases = phase_df[phase_df["error_rate"] <= max_error_rate].copy()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(
                phases,
                x="workload_id",
                y="output_tokens_per_sec",
                color="engine",
                title="Output Throughput by Workload",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(
                phases,
                x="workload_id",
                y="ttft_p95",
                color="engine",
                title="TTFT p95 by Workload",
            )
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            heat = phases.pivot_table(
                index="concurrency",
                columns="prompt_length",
                values="output_tokens_per_sec",
                aggfunc="max",
            )
            if not heat.empty:
                fig = px.imshow(
                    heat,
                    aspect="auto",
                    title="Max Output tok/s by Concurrency and Prompt Length",
                    labels=dict(x="Prompt length", y="Concurrency", color="tok/s"),
                )
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            heat = phases.pivot_table(
                index="concurrency",
                columns="prompt_length",
                values="ttft_p95",
                aggfunc="median",
            )
            if not heat.empty:
                fig = px.imshow(
                    heat,
                    aspect="auto",
                    title="Median TTFT p95 by Concurrency and Prompt Length",
                    labels=dict(x="Prompt length", y="Concurrency", color="ms"),
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Phase Error Rates")
        error_phases = phases[phases["error_rate"] > 0].sort_values(
            "error_rate",
            ascending=False,
        )
        if error_phases.empty:
            st.success("No phase errors in the current selection.")
        else:
            st.dataframe(
                error_phases[
                    [
                        "experiment_id",
                        "engine",
                        "workload_id",
                        "concurrency",
                        "prompt_length",
                        "error_rate",
                        "errors",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )


with tabs[3]:
    st.header("Correctness and Failures")

    col1, col2 = st.columns(2)
    with col1:
        status_counts = filtered["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig = px.bar(status_counts, x="status", y="count", title="Experiment Status")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        failure_counts = filtered["failure_classification"].value_counts().reset_index()
        failure_counts.columns = ["failure_classification", "count"]
        fig = px.bar(
            failure_counts,
            x="failure_classification",
            y="count",
            title="Failure Classification",
        )
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    smoke_cols = [
        "smoke_basic",
        "smoke_tool",
        "smoke_tool_required",
        "smoke_schema",
    ]
    smoke_summary = pd.DataFrame({
        "test": smoke_cols,
        "passed": [int(filtered[col].sum()) for col in smoke_cols],
        "failed": [int((~filtered[col]).sum()) for col in smoke_cols],
    })
    smoke_long = smoke_summary.melt(
        id_vars="test",
        value_vars=["passed", "failed"],
        var_name="result",
        value_name="count",
    )
    fig = px.bar(
        smoke_long,
        x="test",
        y="count",
        color="result",
        barmode="group",
        title="Smoke Test Matrix",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Non-Success Runs")
    failed = filtered[
        (filtered["status"] != "success")
        | (~filtered["correctness_gate_passed"])
        | (filtered["failure_classification"] != "none")
    ]
    if failed.empty:
        st.success("No failures in the current selection.")
    else:
        st.dataframe(
            failed[
                [
                    "experiment_id",
                    "engine",
                    "status",
                    "failure_classification",
                    "correctness_gate_passed",
                    "smoke_basic",
                    "smoke_tool",
                    "smoke_tool_required",
                    "smoke_schema",
                    "time_to_healthy_sec",
                    "duration_s",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


with tabs[4]:
    st.header("Config Impact Analysis")

    metric = st.selectbox(
        "Metric",
        [
            "peak_throughput",
            "ttft_p95",
            "balanced_score",
            "throughput_per_gpu",
            "throughput_per_watt",
            "kv_cache_usage",
        ],
    )
    analysis_source = eligible if not eligible.empty else filtered

    config_dimensions = [
        "engine",
        "tp",
        "parallelism",
        "max_model_len",
        "quantization",
        "dtype",
        "kv_cache_dtype",
        "chunked_prefill",
        "prefix_caching",
        "enforce_eager",
        "scheduling_policy",
        "attention_backend",
        "gpu_memory_utilization",
        "mem_fraction_static",
        "max_num_seqs",
        "max_running_requests",
        "max_num_batched_tokens",
        "max_prefill_tokens",
        "speculative_algorithm",
    ]
    dimension = st.selectbox("Config dimension", config_dimensions)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            analysis_source,
            x=dimension,
            y=metric,
            color="engine",
            points="outliers",
            title=f"{metric} by {dimension}",
            hover_data=["experiment_id"],
        )
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if pd.api.types.is_numeric_dtype(analysis_source[dimension]):
            fig = px.scatter(
                analysis_source,
                x=dimension,
                y=metric,
                color="engine",
                symbol="quantization",
                hover_data=["experiment_id", "tp", "max_model_len"],
                title=f"{metric} vs {dimension}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            grouped = (
                analysis_source.groupby(dimension, dropna=False)[metric]
                .agg(["count", "median", "max"])
                .reset_index()
                .sort_values("max", ascending=False)
            )
            st.dataframe(grouped, use_container_width=True, hide_index=True)

    st.subheader("Top Configs")
    ascending = metric in {"ttft_p95"}
    top_config_cols = [
        "experiment_id",
        "engine",
        "tp",
        "pp",
        "dp",
        "max_model_len",
        "quantization",
        "dtype",
        "kv_cache_dtype",
        "chunked_prefill",
        "prefix_caching",
        "scheduling_policy",
        metric,
        "peak_throughput",
        "ttft_p95",
        "balanced_score",
    ]
    top_config_cols = list(dict.fromkeys(top_config_cols))
    st.dataframe(
        analysis_source.sort_values(metric, ascending=ascending)[top_config_cols].head(20),
        use_container_width=True,
        hide_index=True,
    )


with tabs[5]:
    st.header("GPU Efficiency")

    gpu_source = eligible if not eligible.empty else filtered
    has_gpu = gpu_source[
        (gpu_source["gpu_util_avg"] > 0)
        | (gpu_source["gpu_memory_peak_mb"] > 0)
        | (gpu_source["gpu_power_total_w"] > 0)
    ]

    if has_gpu.empty:
        st.info("No GPU metrics available.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(
                has_gpu,
                x="gpu_util_avg",
                y="peak_throughput",
                color="engine",
                symbol="quantization",
                hover_data=["experiment_id", "tp", "gpu_memory_headroom_mb"],
                title="Throughput vs GPU Utilization",
                labels={"gpu_util_avg": "Average GPU utilization (%)"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(
                has_gpu,
                x="gpu_power_total_w",
                y="throughput_per_watt",
                color="engine",
                symbol="quantization",
                hover_data=["experiment_id", "peak_throughput"],
                title="Throughput per Watt",
            )
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(
                has_gpu,
                x="gpu_memory_peak_mb",
                y="peak_throughput",
                color="engine",
                symbol="quantization",
                hover_data=["experiment_id", "gpu_memory_headroom_mb"],
                title="Throughput vs Peak GPU Memory",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(
                has_gpu.sort_values("throughput_per_gpu", ascending=False).head(20),
                x="experiment_id",
                y="throughput_per_gpu",
                color="engine",
                title="Top Throughput per GPU",
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            has_gpu.sort_values("throughput_per_watt", ascending=False)[
                [
                    "experiment_id",
                    "engine",
                    "tp",
                    "peak_throughput",
                    "throughput_per_gpu",
                    "throughput_per_watt",
                    "gpu_util_avg",
                    "gpu_memory_peak_mb",
                    "gpu_memory_headroom_mb",
                    "gpu_power_total_w",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


with tabs[6]:
    st.header("Reproducibility")
    st.caption("Copy-paste ready runtime details for reproducing experiments.")

    label_map = {row["experiment_id"]: _baseline_badge(row) + _summary_label(row) for _, row in filtered.iterrows()}
    selected_exp = st.selectbox(
        "Select experiment",
        list(label_map.keys()),
        format_func=lambda x: label_map.get(x, x),
    )

    exp_data = get_experiment_payload(selected_exp) if selected_exp else None
    if exp_data:
        config = exp_data.get("config", {})
        container_cmd = exp_data.get("container_command", "")
        extra_env = config.get("extra_env") or {}

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Engine", str(exp_data.get("engine", "")))
        with col2:
            st.metric("Status", str(exp_data.get("status", "")))
        with col3:
            st.metric("Time to healthy", _format_metric(_as_float(exp_data.get("time_to_healthy_sec")), " s", 1))
        with col4:
            st.metric("Duration", _format_metric(_as_float(exp_data.get("duration_seconds")), " s", 1))

        if container_cmd:
            st.markdown("**Container command:**")
            st.code(container_cmd, language="bash")
        else:
            st.info("Container command not recorded.")

        if extra_env:
            st.markdown("**Extra env:**")
            st.code("\n".join(f"export {k}={v}" for k, v in extra_env.items()), language="bash")

        st.markdown("**Immutable runtime identifiers:**")
        st.dataframe(
            pd.DataFrame([
                {
                    "engine_version": exp_data.get("engine_version", ""),
                    "container_image_digest": exp_data.get("container_image_digest", ""),
                    "benchmark_seed": exp_data.get("benchmark_seed"),
                }
            ]),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Config JSON:**")
        st.json(config)

        commentary = exp_data.get("llm_commentary", "")
        rationale = config.get("rationale", "")
        if commentary or rationale:
            st.markdown("**LLM Analysis:**")
            if commentary:
                st.write(commentary)
            if rationale:
                st.markdown("**Planner Rationale:**")
                st.write(rationale)

        conc_results = exp_data.get("benchmark", {}).get("concurrency_results", [])
        if conc_results:
            st.subheader("Selected Experiment Curves")

            conc_df = pd.DataFrame(conc_results)
            # Extract ttft_p95 from the nested dict so we can plot it as a
            # flat column.
            conc_df["ttft_p95"] = conc_df.get("ttft_ms", pd.Series([{}] * len(conc_df))).apply(
                lambda d: (d or {}).get("p95", 0) if isinstance(d, dict) else 0
            )
            conc_df["workload_id"] = conc_df.get("workload_id", "").fillna("").replace("", "unknown")

            # Workload filter: mixing agent_short (short prompts) with
            # long_context (32k) and agentic_long_context (~48k prefix) on the
            # same concurrency axis makes the line plot unreadable. Filter to
            # let the user inspect one workload at a time.
            available_workloads = sorted(
                w for w in conc_df["workload_id"].unique().tolist()
                if w and w != "warmup"
            )
            selected_workloads = st.multiselect(
                "Filter by workload",
                available_workloads,
                default=available_workloads,
                key=f"reproduce_workload_filter_{selected_exp}",
                help=(
                    "agent_short = short prompts, low concurrency. "
                    "throughput = short prompts, mid/high concurrency. "
                    "stress = saturation probe. "
                    "long_context = 32k+ prompts. "
                    "agentic_long_context = multi-turn code-agent simulation."
                ),
            )
            view_df = conc_df[conc_df["workload_id"].isin(selected_workloads)].copy()

            if view_df.empty:
                st.info("No phases match the selected workloads.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.line(
                        view_df.sort_values(["workload_id", "prompt_length", "concurrency"]),
                        x="concurrency",
                        y="output_tokens_per_sec",
                        color="workload_id",
                        line_dash="prompt_length",
                        markers=True,
                        title="Output Throughput by Concurrency",
                        hover_data=["phase_id", "prompt_length"],
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.line(
                        view_df.sort_values(["workload_id", "prompt_length", "concurrency"]),
                        x="concurrency",
                        y="ttft_p95",
                        color="workload_id",
                        line_dash="prompt_length",
                        markers=True,
                        title="TTFT p95 by Concurrency",
                        hover_data=["phase_id", "prompt_length"],
                    )
                    st.plotly_chart(fig, use_container_width=True)


with tabs[7]:
    st.header("Full Comparison")
    st.dataframe(
        filtered.sort_values("peak_throughput", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    if not phase_df.empty:
        st.subheader("Per-Phase Results")
        st.dataframe(
            phase_df.sort_values(
                ["experiment_id", "concurrency", "prompt_length"],
                ascending=[True, True, True],
            ),
            use_container_width=True,
            hide_index=True,
        )


with tabs[8]:
    st.header("Configuration Search")
    st.caption(
        "Find experiments by any combination of config or metric values, then "
        "drill into one to see its standard + agentic benchmark results inline. "
        "Scope: source-level filters from the sidebar (hardware/model/engine) "
        "apply; display-level narrowing (status/quant/tp) does NOT — the full "
        "loaded set is searchable here."
    )

    search_pool = df.copy()

    id_query = st.text_input(
        "Experiment ID contains",
        value="",
        help="Substring match against experiment_id (case-insensitive).",
        key="search_id_query",
    )
    if id_query:
        search_pool = search_pool[
            search_pool["experiment_id"]
            .astype(str)
            .str.contains(id_query, case=False, na=False)
        ]

    candidate_fields = [c for c in [
        # Identity & status
        "engine", "engine_version", "model", "status", "correctness_gate_passed",
        # Precision
        "quantization", "dtype", "kv_cache_dtype",
        # Parallelism & memory
        "tp", "pp", "dp", "parallelism",
        "max_model_len", "gpu_count",
        "gpu_memory_utilization", "mem_fraction_static",
        "max_num_seqs", "max_running_requests",
        "max_num_batched_tokens", "max_prefill_tokens",
        # Engine knobs
        "chunked_prefill", "prefix_caching", "enforce_eager",
        "scheduling_policy", "attention_backend", "speculative_algorithm",
        # Standard metrics
        "peak_throughput", "peak_total_throughput", "peak_requests_per_sec",
        "ttft_p95", "tpot_p95",
        "peak_throughput_e2e_cv", "low_concurrency_ttft_cv",
        "prefix_hit_rate", "kv_cache_usage",
        # Agentic metrics
        "max_viable_agentic_concurrency", "agentic_ceiling_hit",
        "agentic_saturation_concurrency", "agentic_peak_throughput",
        "agentic_tpot_p95", "agentic_ttft_p95",
        "is_pareto", "is_agentic_pareto",
        # GPU
        "gpu_util_avg", "gpu_power_total_w", "gpu_memory_peak_mb",
    ] if c in search_pool.columns]

    chosen_fields = st.multiselect(
        "Parameters to filter on (AND across selections)",
        candidate_fields,
        default=[],
        help=(
            "Pick any combination of fields. Bool → any/true/false; "
            "low-cardinality numeric (tp, dp, …) → multiselect of values; "
            "high-cardinality numeric → range slider; string → multiselect."
        ),
        key="search_fields",
    )

    if chosen_fields:
        widget_cols = st.columns(min(3, len(chosen_fields)))
        for i, field in enumerate(chosen_fields):
            with widget_cols[i % len(widget_cols)]:
                mask = _build_search_widget(
                    field, search_pool[field], "search",
                )
                search_pool = search_pool[mask.reindex(search_pool.index, fill_value=True)]

    st.divider()
    st.subheader(f"Matched {len(search_pool)} experiments")

    if search_pool.empty:
        st.info("No experiments match the current filters.")
    else:
        result_cols = [c for c in [
            "experiment_id", "is_baseline", "timestamp", "engine", "model",
            "status", "correctness_gate_passed",
            "quantization", "tp", "pp", "dp",
            "prefix_caching", "chunked_prefill",
            "peak_throughput", "ttft_p95", "tpot_p95",
            "max_viable_agentic_concurrency",
            "agentic_tpot_p95", "agentic_peak_throughput",
        ] if c in search_pool.columns]
        sort_col = "timestamp" if "timestamp" in search_pool.columns else "experiment_id"
        st.dataframe(
            search_pool.sort_values(sort_col, ascending=False)[result_cols],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Inspect experiment")
        pick_label_map = {
            row["experiment_id"]: _baseline_badge(row) + _summary_label(row)
            for _, row in search_pool.iterrows()
        }
        pick_options = list(pick_label_map.keys())
        selected_search_id = st.selectbox(
            "Pick an experiment to view its benchmark results",
            pick_options,
            format_func=lambda x: pick_label_map.get(x, x),
            key="search_pick_id",
        )

        if selected_search_id:
            row = search_pool[
                search_pool["experiment_id"] == selected_search_id
            ].iloc[0]

            mcols = st.columns(4)
            mcols[0].metric(
                "Peak throughput",
                _format_metric(_as_float(row.get("peak_throughput")), " tok/s"),
            )
            mcols[1].metric(
                "TTFT p95",
                _format_metric(_as_float(row.get("ttft_p95")), " ms", 1),
            )
            agents_val = int(_as_float(row.get("max_viable_agentic_concurrency", 0)))
            mcols[2].metric(
                "Max parallel agents",
                f"{agents_val}{'+' if bool(row.get('agentic_ceiling_hit', False)) else ''}"
                if agents_val > 0
                else "n/a",
            )
            mcols[3].metric(
                "Agentic tpot p95",
                _format_metric(_as_float(row.get("agentic_tpot_p95")), " ms", 1),
            )

            with st.expander("Config & runtime identifiers", expanded=False):
                cfg_keys = [
                    "engine", "engine_version", "model",
                    "quantization", "dtype", "kv_cache_dtype",
                    "tp", "pp", "dp",
                    "max_model_len", "max_num_seqs", "max_num_batched_tokens",
                    "max_running_requests", "max_prefill_tokens",
                    "gpu_memory_utilization", "mem_fraction_static",
                    "prefix_caching", "chunked_prefill", "enforce_eager",
                    "scheduling_policy", "attention_backend",
                    "speculative_algorithm",
                    "container_image_digest", "benchmark_seed",
                ]
                cfg_view = {
                    k: row.get(k) for k in cfg_keys if k in row.index
                }
                st.dataframe(
                    pd.DataFrame(
                        [(k, v) for k, v in cfg_view.items()],
                        columns=["field", "value"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                container_cmd = row.get("container_command", "")
                if container_cmd:
                    st.markdown("**Container command:**")
                    st.code(container_cmd, language="bash")

            phases_single = list_experiment_phases((selected_search_id,))
            if phases_single.empty:
                st.info("No phase-level data recorded for this experiment.")
            else:
                agentic_mask = phases_single["workload_id"] == "agentic_long_context"
                std_phases = phases_single[~agentic_mask].copy()
                agt_phases = phases_single[agentic_mask].copy()

                st.markdown("### Standard benchmark phases")
                if std_phases.empty:
                    st.info("No standard phases (agentic-only run).")
                else:
                    std_cols = [c for c in [
                        "workload_id", "phase_id", "concurrency", "prompt_length",
                        "num_requests", "output_tokens_per_sec",
                        "requests_per_sec",
                        "ttft_p50", "ttft_p95", "ttft_p99",
                        "tpot_p95", "e2e_p95",
                        "errors", "error_rate",
                    ] if c in std_phases.columns]
                    st.dataframe(
                        std_phases.sort_values(
                            ["workload_id", "concurrency", "prompt_length"]
                        )[std_cols],
                        use_container_width=True,
                        hide_index=True,
                    )
                    chart_df = std_phases.sort_values(["workload_id", "concurrency"])
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.plotly_chart(
                            px.line(
                                chart_df,
                                x="concurrency",
                                y="output_tokens_per_sec",
                                color="workload_id",
                                line_dash="prompt_length",
                                markers=True,
                                title="Output throughput by concurrency",
                                hover_data=["phase_id", "ttft_p95", "errors"],
                            ),
                            use_container_width=True,
                        )
                    with col_b:
                        st.plotly_chart(
                            px.line(
                                chart_df,
                                x="concurrency",
                                y="ttft_p95",
                                color="workload_id",
                                line_dash="prompt_length",
                                markers=True,
                                title="TTFT p95 by concurrency",
                                hover_data=["phase_id", "errors"],
                            ),
                            use_container_width=True,
                        )

                st.markdown("### Agentic long-context phases")
                if agt_phases.empty:
                    st.info(
                        "No agentic phases. Enable "
                        "`benchmark.enable_agentic_long_context: true` in "
                        "`config.yaml` and re-run the agent."
                    )
                else:
                    agt_cols = [c for c in [
                        "phase_id", "concurrency", "num_requests",
                        "output_tokens_per_sec",
                        "ttft_p50", "ttft_p95", "tpot_p95", "e2e_p95",
                        "errors", "error_rate",
                    ] if c in agt_phases.columns]
                    st.dataframe(
                        agt_phases.sort_values("concurrency")[agt_cols],
                        use_container_width=True,
                        hide_index=True,
                    )

                    turns_single = list_agentic_turn_metrics((selected_search_id,))
                    if not turns_single.empty:
                        ok_turns = turns_single[
                            turns_single["error"].isna()
                            & (turns_single["ttft_ms"] > 0)
                        ]
                        if not ok_turns.empty:
                            st.plotly_chart(
                                px.box(
                                    ok_turns,
                                    x="turn_idx",
                                    y="ttft_ms",
                                    color="concurrency",
                                    points=False,
                                    title=(
                                        "TTFT distribution per turn "
                                        "(0 = cold prefill, 1+ = warm cache)"
                                    ),
                                    labels={
                                        "turn_idx": "Turn index",
                                        "ttft_ms": "TTFT, ms",
                                    },
                                ),
                                use_container_width=True,
                            )


with tabs[9]:
    st.header("Compare Two Configs")
    st.caption(
        "Подбираешь два эксперимента, видишь команды запуска, headline-метрики "
        "с дельтой, diff только различающихся параметров конфига, оверлеи "
        "throughput/TTFT по concurrency, агентные кривые по turn-индексу (cold/warm "
        "разделены), correctness gate, стабильность и LLM-комментарии."
    )

    if df.empty:
        st.info("No experiments loaded.")
    else:
        compare_pool = df.copy()
        labels = {
            row["experiment_id"]: _baseline_badge(row) + _summary_label(row)
            for _, row in compare_pool.iterrows()
        }
        opts = list(labels.keys())

        # If a baseline exists, default A to it so the natural comparison is
        # "baseline vs <something>" — directly answers "what did the agent add?".
        baseline_compare = _baseline_row(compare_pool)
        baseline_id = (
            baseline_compare["experiment_id"] if baseline_compare is not None else None
        )
        default_a_idx = opts.index(baseline_id) if baseline_id in opts else 0

        sel_a, sel_b = st.columns(2)
        with sel_a:
            id_a = st.selectbox(
                "Config A (⭐ = baseline)",
                opts,
                index=default_a_idx,
                format_func=lambda x: labels.get(x, x),
                key="compare_a",
            )
        with sel_b:
            # Default B to the best agent config (highest max parallel agents)
            # that isn't A, so the page opens on baseline-vs-best out of the box.
            default_b_idx = 1 if len(opts) > 1 else 0
            if "max_viable_agentic_concurrency" in compare_pool.columns:
                ranked = compare_pool.sort_values(
                    "max_viable_agentic_concurrency", ascending=False,
                )
                for cand in ranked["experiment_id"]:
                    if cand != id_a and cand in opts:
                        default_b_idx = opts.index(cand)
                        break
            id_b = st.selectbox(
                "Config B",
                opts,
                index=default_b_idx,
                format_func=lambda x: labels.get(x, x),
                key="compare_b",
            )

        if id_a == id_b:
            st.warning(
                "Выбраны одинаковые эксперименты — diff и дельты будут пустыми."
            )

        payload_a = get_experiment_payload(id_a) or {}
        payload_b = get_experiment_payload(id_b) or {}
        row_a = compare_pool[compare_pool["experiment_id"] == id_a].iloc[0]
        row_b = compare_pool[compare_pool["experiment_id"] == id_b].iloc[0]

        # Загружаем phase / turn данные один раз для обоих
        pair_ids = tuple(sorted({id_a, id_b}))
        pair_phases = list_experiment_phases(pair_ids)
        pair_turns = list_agentic_turn_metrics(pair_ids)

        # ── 1. Headline metrics ───────────────────────────────────────────
        st.subheader("Headline metrics")
        metric_specs: list[tuple[str, str, str, int, bool]] = [
            # label, field, suffix, precision, lower_is_better
            ("Peak throughput", "peak_throughput", " tok/s", 0, False),
            ("TTFT p95", "ttft_p95", " ms", 1, True),
            ("Tpot p95", "tpot_p95", " ms", 1, True),
            ("Max parallel agents", "max_viable_agentic_concurrency", "", 0, False),
            ("Agentic peak", "agentic_peak_throughput", " tok/s", 0, False),
            ("Agentic tpot p95", "agentic_tpot_p95", " ms", 1, True),
            ("Agentic TTFT p95", "agentic_ttft_p95", " ms", 1, True),
            ("Prefix hit rate", "prefix_hit_rate", "", 3, False),
        ]
        metric_rows = []
        for label, field, suffix, prec, lower_better in metric_specs:
            va = _as_float(row_a.get(field, 0))
            vb = _as_float(row_b.get(field, 0))
            delta = vb - va if (va > 0 or vb > 0) else 0.0
            if abs(delta) < 10 ** (-prec):
                delta_str = "—"
                winner = "="
            else:
                arrow = "↑" if delta > 0 else "↓"
                delta_str = f"{arrow} {abs(delta):.{prec}f}{suffix}"
                if lower_better:
                    winner = "A" if delta > 0 else "B"
                else:
                    winner = "B" if delta > 0 else "A"
            metric_rows.append({
                "metric": label,
                "A": _format_metric(va, suffix, prec),
                "B": _format_metric(vb, suffix, prec),
                "Δ (B−A)": delta_str,
                "winner": winner,
            })
        st.dataframe(
            pd.DataFrame(metric_rows),
            use_container_width=True,
            hide_index=True,
        )

        # ── 2. Container commands ─────────────────────────────────────────
        st.subheader("Container commands (launch lines)")
        ccol_a, ccol_b = st.columns(2)
        with ccol_a:
            st.markdown(f"**A** — `{id_a}`")
            cmd_a = payload_a.get("container_command", "") if payload_a else ""
            st.code(cmd_a or "(not recorded)", language="bash")
        with ccol_b:
            st.markdown(f"**B** — `{id_b}`")
            cmd_b = payload_b.get("container_command", "") if payload_b else ""
            st.code(cmd_b or "(not recorded)", language="bash")

        # ── 3. Config diff ────────────────────────────────────────────────
        st.subheader("Config diff — only differing keys")
        diff_rows = _diff_runtime_inputs(payload_a, payload_b)
        if not diff_rows:
            st.success("Конфиги идентичны — расхождений нет (включая engine_version и digest).")
        else:
            view_mode = st.radio(
                "Diff view",
                ["table", "code-block (A vs B)"],
                horizontal=True,
                key="compare_diff_view",
            )
            if view_mode == "table":
                diff_df = pd.DataFrame(diff_rows)
                diff_df["A"] = diff_df["A"].astype(str)
                diff_df["B"] = diff_df["B"].astype(str)
                st.dataframe(diff_df, use_container_width=True, hide_index=True)
            else:
                lines = []
                for r in diff_rows:
                    lines.append(f"- {r['field']}: {r['A']}")
                    lines.append(f"+ {r['field']}: {r['B']}")
                    lines.append("")
                st.code("\n".join(lines), language="diff")

        # ── 4. Standard throughput / TTFT overlay vs concurrency ──────────
        st.subheader("Standard benchmark — overlay vs concurrency")
        std_pair = pair_phases[
            pair_phases["workload_id"] != "agentic_long_context"
        ].copy() if not pair_phases.empty else pd.DataFrame()
        if std_pair.empty:
            st.info("No standard-workload phases for the selected pair.")
        else:
            std_pair["config"] = std_pair["experiment_id"].map({id_a: "A", id_b: "B"})
            ovcol1, ovcol2 = st.columns(2)
            with ovcol1:
                fig = px.line(
                    std_pair.sort_values(["workload_id", "concurrency"]),
                    x="concurrency",
                    y="output_tokens_per_sec",
                    color="config",
                    line_dash="workload_id",
                    markers=True,
                    title="Throughput by concurrency",
                    hover_data=["phase_id", "prompt_length", "ttft_p95"],
                )
                st.plotly_chart(fig, use_container_width=True)
            with ovcol2:
                fig = px.line(
                    std_pair.sort_values(["workload_id", "concurrency"]),
                    x="concurrency",
                    y="ttft_p95",
                    color="config",
                    line_dash="workload_id",
                    markers=True,
                    title="TTFT p95 by concurrency",
                    hover_data=["phase_id", "prompt_length"],
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── 5. Per-phase delta table (inner join on common cells) ─────
            with st.expander("Per-phase delta table (intersected phases)", expanded=False):
                a_phases = std_pair[std_pair["experiment_id"] == id_a]
                b_phases = std_pair[std_pair["experiment_id"] == id_b]
                merge_keys = ["workload_id", "concurrency", "prompt_length"]
                merged = a_phases.merge(
                    b_phases,
                    on=merge_keys,
                    suffixes=("_A", "_B"),
                )
                if merged.empty:
                    st.info("Нет совпадающих (workload, concurrency, prompt_length) ячеек.")
                else:
                    merged["Δ_throughput"] = (
                        merged["output_tokens_per_sec_B"]
                        - merged["output_tokens_per_sec_A"]
                    )
                    merged["Δ_ttft_p95"] = merged["ttft_p95_B"] - merged["ttft_p95_A"]
                    show_cols = [
                        "workload_id", "concurrency", "prompt_length",
                        "output_tokens_per_sec_A", "output_tokens_per_sec_B", "Δ_throughput",
                        "ttft_p95_A", "ttft_p95_B", "Δ_ttft_p95",
                        "error_rate_A", "error_rate_B",
                    ]
                    show_cols = [c for c in show_cols if c in merged.columns]
                    st.dataframe(
                        merged[show_cols].sort_values(
                            ["workload_id", "concurrency", "prompt_length"]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

        # ── 6. Agentic side-by-side: per-c metrics ────────────────────────
        st.subheader("Agentic — side-by-side by concurrency")
        ag_pair = pair_phases[
            pair_phases["workload_id"] == "agentic_long_context"
        ].copy() if not pair_phases.empty else pd.DataFrame()
        if ag_pair.empty:
            st.info(
                "Нет agentic-фаз для пары. Включи `benchmark.enable_agentic_long_context: true` "
                "в обеих конфигурациях, чтобы появилась эта сводка."
            )
        else:
            ag_pair["config"] = ag_pair["experiment_id"].map({id_a: "A", id_b: "B"})
            ag_cols = [
                "config", "concurrency", "num_requests",
                "output_tokens_per_sec", "ttft_p95", "tpot_p95",
                "e2e_p95", "errors", "error_rate",
            ]
            ag_cols = [c for c in ag_cols if c in ag_pair.columns]
            st.dataframe(
                ag_pair[ag_cols].sort_values(["concurrency", "config"]),
                use_container_width=True,
                hide_index=True,
            )

            # ── 7. Agentic turn-by-turn overlay, cold/warm split ──────────
            if not pair_turns.empty:
                pair_turns_view = pair_turns.copy()
                pair_turns_view["config"] = pair_turns_view["experiment_id"].map(
                    {id_a: "A", id_b: "B"}
                )
                ok = pair_turns_view[
                    pair_turns_view["error"].isna() & (pair_turns_view["ttft_ms"] > 0)
                ]
                cold = ok[ok["turn_idx"] == 0]
                warm = ok[ok["turn_idx"] >= 1]
                st.markdown("**TTFT per turn — A vs B (cold left, warm right)**")
                tcol_cold, tcol_warm = st.columns(2)
                with tcol_cold:
                    if cold.empty:
                        st.info("No cold (turn 0) samples.")
                    else:
                        fig = px.box(
                            cold,
                            x="config",
                            y="ttft_ms",
                            color="config",
                            points="outliers",
                            title="Cold (turn 0)",
                            labels={"ttft_ms": "TTFT cold, ms"},
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                with tcol_warm:
                    if warm.empty:
                        st.info("No warm (turn 1+) samples.")
                    else:
                        fig = px.box(
                            warm,
                            x="turn_idx",
                            y="ttft_ms",
                            color="config",
                            points=False,
                            title="Warm (turn 1+)",
                            labels={
                                "turn_idx": "Turn index",
                                "ttft_ms": "TTFT warm, ms",
                            },
                        )
                        st.plotly_chart(fig, use_container_width=True)

        # ── 8. Correctness gate breakdown ─────────────────────────────────
        st.subheader("Correctness gate")
        gate_fields = [
            ("Smoke: basic chat", "smoke_basic"),
            ("Smoke: tool calling", "smoke_tool"),
            ("Smoke: tool_required", "smoke_tool_required"),
            ("Smoke: json_schema", "smoke_schema"),
            ("Post-bench basic chat", "post_basic_chat"),
            ("Gate passed (overall)", "correctness_gate_passed"),
        ]
        gate_rows = []
        for label, field in gate_fields:
            va = bool(row_a.get(field, False))
            vb = bool(row_b.get(field, False))
            gate_rows.append({
                "check": label,
                "A": "✓" if va else "✗",
                "B": "✓" if vb else "✗",
                "diff": "" if va == vb else ("A→✗ / B→✓" if not va else "A→✓ / B→✗"),
            })
        st.dataframe(
            pd.DataFrame(gate_rows),
            use_container_width=True,
            hide_index=True,
        )

        # ── 9. Stability / CV / cache ─────────────────────────────────────
        with st.expander("Stability, dispersion, cache effectiveness", expanded=False):
            stab_fields = [
                ("Peak throughput e2e CV", "peak_throughput_e2e_cv", "", 3, True),
                ("Low-concurrency TTFT CV", "low_concurrency_ttft_cv", "", 3, True),
                ("Prefix hit rate", "prefix_hit_rate", "", 3, False),
                ("KV cache usage", "kv_cache_usage", "", 3, False),
            ]
            stab_rows = []
            for label, field, suffix, prec, lower_better in stab_fields:
                va = _as_float(row_a.get(field, 0))
                vb = _as_float(row_b.get(field, 0))
                delta = vb - va
                stab_rows.append({
                    "metric": label,
                    "A": _format_metric(va, suffix, prec),
                    "B": _format_metric(vb, suffix, prec),
                    "Δ (B−A)": f"{'+' if delta > 0 else ''}{delta:.{prec}f}{suffix}",
                    "lower_better": "yes" if lower_better else "no",
                })
            st.dataframe(
                pd.DataFrame(stab_rows),
                use_container_width=True,
                hide_index=True,
            )

        # ── 10. Runtime cost ──────────────────────────────────────────────
        with st.expander("Runtime cost (startup + benchmark duration)", expanded=False):
            cost_rows = []
            for label, field, suffix, prec in [
                ("Time to healthy", "time_to_healthy_sec", " s", 1),
                ("Total duration", "duration_s", " s", 1),
            ]:
                va = _as_float(row_a.get(field, 0))
                vb = _as_float(row_b.get(field, 0))
                delta = vb - va
                cost_rows.append({
                    "metric": label,
                    "A": _format_metric(va, suffix, prec),
                    "B": _format_metric(vb, suffix, prec),
                    "Δ (B−A)": f"{'+' if delta > 0 else ''}{delta:.{prec}f}{suffix}",
                })
            st.dataframe(
                pd.DataFrame(cost_rows),
                use_container_width=True,
                hide_index=True,
            )

        # ── 11. LLM rationale + analyzer commentary ───────────────────────
        with st.expander("LLM rationale + analyzer commentary", expanded=False):
            rcol_a, rcol_b = st.columns(2)
            with rcol_a:
                st.markdown(f"**A** — `{id_a}`")
                rationale_a = (payload_a.get("config") or {}).get("rationale", "")
                commentary_a = payload_a.get("llm_commentary", "")
                if rationale_a:
                    st.markdown("_Planner rationale:_")
                    st.write(rationale_a)
                if commentary_a:
                    st.markdown("_Analyzer commentary:_")
                    st.write(commentary_a)
                if not rationale_a and not commentary_a:
                    st.info("(no commentary)")
            with rcol_b:
                st.markdown(f"**B** — `{id_b}`")
                rationale_b = (payload_b.get("config") or {}).get("rationale", "")
                commentary_b = payload_b.get("llm_commentary", "")
                if rationale_b:
                    st.markdown("_Planner rationale:_")
                    st.write(rationale_b)
                if commentary_b:
                    st.markdown("_Analyzer commentary:_")
                    st.write(commentary_b)
                if not rationale_b and not commentary_b:
                    st.info("(no commentary)")


with tabs[10]:
    st.header("Agentic Long-Context Analytics")
    st.caption(
        "Multi-turn code-agent simulation: фиксированный префикс ~64k + N ходов, "
        "после каждого ответа модели в историю докидывается synthetic tool-result "
        "1–5k токенов. Главная производная метрика — **max parallel agents** "
        "(сколько code-агентов одновременно выдержит конфиг)."
    )

    agentic_phases_df = phase_df[phase_df["workload_id"] == "agentic_long_context"].copy() if not phase_df.empty else pd.DataFrame()
    agentic_summary = filtered[filtered["max_viable_agentic_concurrency"] > 0].copy() if "max_viable_agentic_concurrency" in filtered.columns else pd.DataFrame()

    if agentic_phases_df.empty:
        st.info(
            "No agentic_long_context phases found. Enable in `config.yaml` under "
            "`benchmark.enable_agentic_long_context: true` and re-run the agent."
        )
    else:
        # Pull per-turn rows once for the active selection.
        agentic_ids = tuple(sorted(agentic_phases_df["experiment_id"].astype(str).unique()))
        turns_df = list_agentic_turn_metrics(agentic_ids)

        # ── KPI row ──
        col1, col2, col3, col4 = st.columns(4)

        if not agentic_summary.empty:
            best_row = agentic_summary.sort_values(
                "max_viable_agentic_concurrency", ascending=False,
            ).iloc[0]
            best_max = int(best_row["max_viable_agentic_concurrency"])
            best_ceiling_hit = bool(best_row.get("agentic_ceiling_hit", False))
            col1.metric(
                "Max parallel agents (best config)",
                f"{best_max:d}{'+' if best_ceiling_hit else ''}",
                help=(
                    "Highest concurrency where TTFT p95 ≤ SLO and error_rate ≤ threshold. "
                    "'+' means we hit the sweep ceiling — the real number may be higher "
                    "(enable agentic_concurrency_ceiling_search)."
                ),
            )
        else:
            col1.metric("Max parallel agents (best config)", "n/a")

        peak_tp = agentic_phases_df["output_tokens_per_sec"].max() if not agentic_phases_df.empty else 0.0
        col2.metric("Peak agentic throughput", _format_metric(peak_tp, " tok/s", precision=1))

        if not turns_df.empty:
            turn0 = turns_df[(turns_df["turn_idx"] == 0) & (turns_df["error"].isna()) & (turns_df["ttft_ms"] > 0)]
            turn1plus = turns_df[(turns_df["turn_idx"] >= 1) & (turns_df["error"].isna()) & (turns_df["ttft_ms"] > 0)]
            col3.metric(
                "Median TTFT, turn 1 (cold prefill)",
                _format_metric(turn0["ttft_ms"].median(), " ms", precision=0) if not turn0.empty else "n/a",
            )
            col4.metric(
                "Median TTFT, turn 2+ (warm cache)",
                _format_metric(turn1plus["ttft_ms"].median(), " ms", precision=0) if not turn1plus.empty else "n/a",
            )
        else:
            col3.metric("Median TTFT, turn 1", "n/a")
            col4.metric("Median TTFT, turn 2+", "n/a")

        # ── Leaderboard: max parallel agents per engine/config ──
        st.subheader("Max parallel agents per experiment")
        if not agentic_summary.empty:
            leaderboard = agentic_summary[[
                "experiment_id", "engine", "quantization", "tp",
                "prefix_caching", "chunked_prefill",
                "max_viable_agentic_concurrency",
                "agentic_ceiling_hit",
                "agentic_saturation_concurrency",
                "agentic_peak_throughput",
                "prefix_hit_rate",
            ]].sort_values(
                ["max_viable_agentic_concurrency", "agentic_peak_throughput"],
                ascending=[False, False],
            ).rename(columns={
                "max_viable_agentic_concurrency": "max_agents",
                "agentic_ceiling_hit": "ceiling_hit",
                "agentic_saturation_concurrency": "saturation_c",
                "agentic_peak_throughput": "peak_tok/s",
                "prefix_hit_rate": "prefix_cache_hit",
            })
            st.dataframe(leaderboard, use_container_width=True, hide_index=True)
        else:
            st.info("No experiments passed the agentic SLO gates yet.")

        # ── Throughput by concurrency (per engine) ──
        # Each row in agentic_phases_df is one (experiment, concurrency) sample,
        # so at a given concurrency there can be many rows from different
        # experiments. px.bar without aggregation stacks those rows' y values,
        # producing bars that are the SUM of all matching runs — misleading.
        # Pick the winning row (max output_tokens_per_sec) per (concurrency,
        # engine) so the bar represents the best config at that concurrency.
        st.subheader("Agentic throughput by concurrency — best config per engine")
        winners_idx = agentic_phases_df.groupby(["concurrency", "engine"])[
            "output_tokens_per_sec"
        ].idxmax()
        agentic_best = agentic_phases_df.loc[winners_idx].copy()
        fig_tp = px.bar(
            agentic_best,
            x="concurrency",
            y="output_tokens_per_sec",
            color="engine",
            barmode="group",
            hover_data=["experiment_id", "ttft_p95", "errors"],
            labels={
                "output_tokens_per_sec": "Output tokens/sec (best run, aggregate over sessions)",
                "concurrency": "Concurrent agentic sessions",
            },
        )
        st.plotly_chart(fig_tp, use_container_width=True)
        with st.expander("Show all runs (not just best per concurrency)"):
            st.caption(
                "Scatter view: every (experiment, concurrency) point is one dot. "
                "Use this to see spread / regressions across configs at the same concurrency."
            )
            fig_all = px.strip(
                agentic_phases_df,
                x="concurrency",
                y="output_tokens_per_sec",
                color="engine",
                hover_data=["experiment_id", "ttft_p95", "errors"],
                labels={"output_tokens_per_sec": "Output tokens/sec (aggregate)"},
            )
            st.plotly_chart(fig_all, use_container_width=True)

        # ── TTFT vs turn_idx — split into cold (turn 0) and warm (turn 1+) ──
        # Объединять их на одном Y нельзя: cold prefill 83k токенов даёт
        # 20-60s, а warm cache-hit на ~5k новых токенов — 0.2-1s. Боксы warm
        # турнов схлопываются в полосу около нуля и шкала становится мёртвой.
        # Разделение на два графика даёт каждой популяции свою Y-ось.
        if not turns_df.empty:
            st.subheader("TTFT vs turn index — does prefix-cache work?")
            st.caption(
                "Cold (turn 0) — полный prefill ~83k unique-токенов. Warm (turn 1+) — "
                "должен лететь на cache-hit, TTFT падает в ~50× раз. Разделили в два "
                "графика, потому что общая Y-ось делает warm-turn'ы нечитаемыми."
            )
            ok_turns = turns_df[(turns_df["error"].isna()) & (turns_df["ttft_ms"] > 0)].copy()
            if not ok_turns.empty:
                cold_turns = ok_turns[ok_turns["turn_idx"] == 0]
                warm_turns = ok_turns[ok_turns["turn_idx"] >= 1]

                col_cold, col_warm = st.columns(2)
                with col_cold:
                    st.markdown("**Cold: turn 0 (full prefill)**")
                    if cold_turns.empty:
                        st.info("No turn-0 samples.")
                    else:
                        fig_cold = px.box(
                            cold_turns,
                            x="engine",
                            y="ttft_ms",
                            color="engine",
                            points="outliers",
                            labels={"ttft_ms": "TTFT, ms (cold prefill)"},
                        )
                        fig_cold.update_layout(showlegend=False)
                        st.plotly_chart(fig_cold, use_container_width=True)
                with col_warm:
                    st.markdown("**Warm: turn 1+ (cache-hit expected)**")
                    if warm_turns.empty:
                        st.info("No turn-1+ samples.")
                    else:
                        fig_warm = px.box(
                            warm_turns,
                            x="turn_idx",
                            y="ttft_ms",
                            color="engine",
                            points=False,
                            labels={
                                "turn_idx": "Turn index (warm)",
                                "ttft_ms": "TTFT, ms (warm)",
                            },
                        )
                        st.plotly_chart(fig_warm, use_container_width=True)

                # Scatter: input_tokens (per turn) vs ttft_ms — также split.
                # На общей шкале turn 0 (50s, 83k tokens) живёт высоко слева,
                # а warm turns тянутся горизонтальной линией у нуля при росте
                # input_tokens — невозможно понять, есть ли наклон.
                st.subheader("Per-turn input length vs TTFT")
                st.caption(
                    "На warm-графике: линейный (растущий) тренд ⇒ engine делает "
                    "full prefill каждый turn (cache miss). Плоский тренд ⇒ кеш "
                    "работает. Cold-график показывает разброс самого тяжёлого "
                    "prefill'а по разным конфигам."
                )
                col_cold2, col_warm2 = st.columns(2)
                with col_cold2:
                    st.markdown("**Cold: turn 0 only**")
                    if cold_turns.empty:
                        st.info("No turn-0 samples.")
                    else:
                        fig_cold_sc = px.scatter(
                            cold_turns,
                            x="input_tokens",
                            y="ttft_ms",
                            color="engine",
                            hover_data=["experiment_id", "concurrency", "session_idx"],
                            labels={
                                "input_tokens": "Cold prefill tokens",
                                "ttft_ms": "TTFT, ms (cold)",
                            },
                        )
                        st.plotly_chart(fig_cold_sc, use_container_width=True)
                with col_warm2:
                    st.markdown("**Warm: turn 1+**")
                    if warm_turns.empty:
                        st.info("No turn-1+ samples.")
                    else:
                        fig_warm_sc = px.scatter(
                            warm_turns,
                            x="input_tokens",
                            y="ttft_ms",
                            color="engine",
                            symbol="turn_idx",
                            hover_data=["experiment_id", "concurrency", "session_idx", "turn_idx"],
                            labels={
                                "input_tokens": "Per-turn input tokens (warm)",
                                "ttft_ms": "TTFT, ms (warm)",
                            },
                        )
                        st.plotly_chart(fig_warm_sc, use_container_width=True)
                        # Linear OLS на dataframe (без statsmodels): простой
                        # numeric slope как подпись — даёт ответ «работает ли
                        # кеш» в одной цифре без визуального fitting'а.
                        if len(warm_turns) >= 2:
                            x = warm_turns["input_tokens"].astype(float).to_numpy()
                            y = warm_turns["ttft_ms"].astype(float).to_numpy()
                            if x.std() > 0:
                                slope = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
                                st.caption(
                                    f"OLS slope: **{slope:.3f} ms/token** — близко к 0 ⇒ "
                                    "cache работает; заметный положительный ⇒ full prefill "
                                    "каждый turn."
                                )
            else:
                st.info("No successful turns to plot.")

        # ── Per-phase table ──
        st.subheader("Per-agentic-phase breakdown")
        st.dataframe(
            agentic_phases_df[[
                "experiment_id", "engine", "phase_id", "concurrency",
                "num_requests", "output_tokens_per_sec",
                "ttft_p50", "ttft_p95", "tpot_p95", "e2e_p95",
                "errors", "error_rate",
            ]].sort_values(
                ["experiment_id", "concurrency"],
            ),
            use_container_width=True,
            hide_index=True,
        )


with tabs[11]:
    st.header("Manage Experiments")
    st.caption(
        "Permanently delete experiments from the database. This cannot be undone; "
        "the row, JSONB payload, and any per-phase data are removed."
    )

    with st.expander("Delete experiments", expanded=False):
        id_to_label = {
            row["experiment_id"]: _baseline_badge(row) + _summary_label(row)
            for _, row in filtered.iterrows()
            if row["experiment_id"]
        }

        selected_ids = st.multiselect(
            "Experiments to delete",
            options=list(id_to_label.keys()),
            format_func=lambda x: id_to_label.get(x, x),
            key="delete_selection",
        )

        confirm = st.checkbox(
            f"I understand this will permanently delete {len(selected_ids)} "
            f"experiment(s).",
            key="delete_confirm",
            disabled=not selected_ids,
        )

        if st.button(
            "Delete selected",
            type="primary",
            disabled=not (selected_ids and confirm),
        ):
            try:
                deleted = delete_experiments(selected_ids)
            except Exception as exc:
                st.error(f"Delete failed: {exc}")
            else:
                st.success(f"Deleted {deleted} experiment(s).")
                st.rerun()
