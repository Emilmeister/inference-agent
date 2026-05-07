"""Streamlit dashboard for visualizing inference benchmark experiments.

Source: Postgres (configured via DATABASE_* env vars + DB_PASSWORD).
Run: `streamlit run streamlit_app/app.py`
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from db import (
    Filters,
    HardwareKey,
    delete_experiments,
    get_experiment_payload,
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


# ---- Source filters (Postgres-backed) ----


st.sidebar.header("Source")

try:
    hardware_options = list_distinct_hardware()
    model_options = list_distinct_models()
    engine_options = list_distinct_engines()
except Exception as exc:  # pragma: no cover - surfaced to user via UI
    st.error(f"Failed to connect to Postgres: {exc}")
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

st.success(f"Loaded {len(df)} experiments from Postgres")


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
            points="all",
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
        docker_cmd = exp_data.get("docker_command", "")
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

        if docker_cmd:
            st.markdown("**Docker command:**")
            st.code(docker_cmd, language="bash")
        else:
            st.info("Docker command not recorded.")

        if extra_env:
            st.markdown("**Extra env:**")
            st.code("\n".join(f"export {k}={v}" for k, v in extra_env.items()), language="bash")

        st.markdown("**Immutable runtime identifiers:**")
        st.dataframe(
            pd.DataFrame([
                {
                    "engine_version": exp_data.get("engine_version", ""),
                    "docker_image_digest": exp_data.get("docker_image_digest", ""),
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
            col1, col2 = st.columns(2)
            with col1:
                fig = px.line(
                    conc_df,
                    x="concurrency",
                    y="output_tokens_per_sec",
                    color="prompt_length",
                    markers=True,
                    title="Output Throughput by Concurrency",
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                ttft_data = [
                    {
                        "concurrency": cr.get("concurrency"),
                        "prompt_length": cr.get("prompt_length"),
                        "ttft_p95": cr.get("ttft_ms", {}).get("p95", 0),
                    }
                    for cr in conc_results
                ]
                fig = px.line(
                    pd.DataFrame(ttft_data),
                    x="concurrency",
                    y="ttft_p95",
                    color="prompt_length",
                    markers=True,
                    title="TTFT p95 by Concurrency",
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
