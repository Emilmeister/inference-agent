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


def _summary_label(row: pd.Series) -> str:
    return (
        f"{row['experiment_id']} ({row['engine']} "
        f"TP={row['tp']} q={row['quantization']}) - "
        f"{row['peak_throughput']:.0f} tok/s"
    )


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

hw_labels = {hw.label(): hw for hw in hardware_options}
hw_label = st.sidebar.selectbox("Hardware", list(hw_labels.keys()))
selected_hw: HardwareKey = hw_labels[hw_label]

selected_models = st.sidebar.multiselect(
    "Model",
    model_options,
    default=model_options,
)
selected_engines_src = st.sidebar.multiselect(
    "Engine (source filter)",
    engine_options,
    default=engine_options,
)

filters = Filters(
    hardware=selected_hw,
    models=tuple(selected_models),
    engines=tuple(selected_engines_src),
)

df = list_experiment_summaries(filters)

if df.empty:
    st.warning("No experiments match the current filters.")
    st.stop()

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
    "Agentic",
    "Manage",
])


with tabs[0]:
    st.header("Executive Overview")

    best_tp = eligible.nlargest(1, "peak_throughput")
    best_lat = eligible.nsmallest(1, "ttft_p95")
    best_bal = eligible.nlargest(1, "balanced_score")
    pareto_count = len(_pareto_ids(eligible))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        value = best_tp["peak_throughput"].iloc[0] if not best_tp.empty else 0
        st.metric("Best throughput", _format_metric(value, " tok/s"))
    with col2:
        value = best_lat["ttft_p95"].iloc[0] if not best_lat.empty else 0
        st.metric("Best latency", _format_metric(value, " ms", precision=1))
    with col3:
        value = best_bal["balanced_score"].iloc[0] if not best_bal.empty else 0
        st.metric("Best balanced score", f"{value:.3f}" if value > 0 else "n/a")
    with col4:
        st.metric("Pareto points", str(pareto_count))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Displayed experiments", str(len(filtered)))
    with col2:
        st.metric("Eligible experiments", str(len(eligible)))
    with col3:
        st.metric("Correctness gate pass", str(int(filtered["correctness_gate_passed"].sum())))
    with col4:
        failed = int((filtered["status"] != "success").sum())
        st.metric("Non-success runs", str(failed))

    st.subheader("Noise-Aware Leaderboards")
    col1, col2, col3 = st.columns(3)

    leaderboard_cols = [
        "experiment_id",
        "engine",
        "quantization",
        "tp",
        "peak_throughput",
        "ttft_p95",
        "peak_throughput_e2e_cv",
        "low_concurrency_ttft_cv",
    ]

    with col1:
        st.caption("Throughput objective")
        st.dataframe(
            eligible.nlargest(5, "peak_throughput")[leaderboard_cols],
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.caption("Latency objective")
        st.dataframe(
            eligible.nsmallest(5, "ttft_p95")[leaderboard_cols],
            use_container_width=True,
            hide_index=True,
        )

    with col3:
        st.caption("Balanced objective")
        st.dataframe(
            eligible.nlargest(5, "balanced_score")[
                [
                    "experiment_id",
                    "engine",
                    "quantization",
                    "tp",
                    "balanced_score",
                    "peak_throughput",
                    "ttft_p95",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Best-So-Far Timeline")
    timeline = filtered.dropna(subset=["timestamp"]).sort_values("timestamp").copy()
    timeline = timeline[
        (timeline["status"] == "success")
        & timeline["correctness_gate_passed"]
        & (timeline["peak_throughput"] > 0)
        & (timeline["ttft_p95"] > 0)
    ]
    if not timeline.empty:
        timeline["best_throughput_so_far"] = timeline["peak_throughput"].cummax()
        timeline["best_latency_so_far"] = timeline["ttft_p95"].cummin()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timeline["timestamp"],
            y=timeline["best_throughput_so_far"],
            mode="lines+markers",
            name="Best throughput so far",
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
            yaxis=dict(title="Output tok/s"),
            yaxis2=dict(title="TTFT p95 ms", overlaying="y", side="right"),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No timestamped eligible runs available for the timeline.")


with tabs[1]:
    st.header("Throughput vs Latency Pareto Explorer")

    pareto_source = eligible if eligible_only else filtered
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
        "smoke_json",
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
                    "smoke_json",
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

    label_map = {row["experiment_id"]: _summary_label(row) for _, row in filtered.iterrows()}
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

        # ── TTFT vs turn_idx (the killer chart) ──
        if not turns_df.empty:
            st.subheader("TTFT vs turn index — does prefix-cache work?")
            st.caption(
                "Turn 0 = cold prefill of the ~64k prefix. Turn 1+ should be a "
                "cache-hit and TTFT should drop dramatically. A flat line means "
                "prefix-caching is disabled or not effective for this config."
            )
            ok_turns = turns_df[(turns_df["error"].isna()) & (turns_df["ttft_ms"] > 0)].copy()
            if not ok_turns.empty:
                fig_ttft = px.box(
                    ok_turns,
                    x="turn_idx",
                    y="ttft_ms",
                    color="engine",
                    points=False,
                    labels={"turn_idx": "Turn index (0 = cold)", "ttft_ms": "TTFT, ms"},
                )
                st.plotly_chart(fig_ttft, use_container_width=True)

                # Scatter: input_tokens (per turn) vs ttft_ms
                st.subheader("Per-turn input length vs TTFT")
                st.caption(
                    "Linear (rising) trend ⇒ engine is doing full prefill every turn "
                    "(prefix-cache miss). Flat trend ⇒ cache works."
                )
                fig_scatter = px.scatter(
                    ok_turns,
                    x="input_tokens",
                    y="ttft_ms",
                    color="engine",
                    symbol="turn_idx",
                    hover_data=["experiment_id", "concurrency", "session_idx", "turn_idx"],
                    labels={"input_tokens": "Per-turn input tokens", "ttft_ms": "TTFT, ms"},
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
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


with tabs[9]:
    st.header("Manage Experiments")
    st.caption(
        "Permanently delete experiments from the database. This cannot be undone; "
        "the row, JSONB payload, and any per-phase data are removed."
    )

    with st.expander("Delete experiments", expanded=False):
        id_to_label = {
            row["experiment_id"]: _summary_label(row)
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
