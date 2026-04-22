"""Streamlit dashboard for visualizing inference benchmark experiments."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Inference Benchmark Dashboard",
    layout="wide",
)

st.title("LLM Inference Benchmark Dashboard")


# ── File upload ────────────────────────────────────────────────────────────

uploaded_files = st.file_uploader(
    "Upload experiment JSON files",
    type=["json"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more experiment JSON files to get started.")
    st.stop()


# ── Parse experiments ───────���──────────────────────────────────────────────


@st.cache_data
def parse_experiments(files_data: list[tuple[str, bytes]]) -> list[dict]:
    experiments = []
    for name, data in files_data:
        try:
            exp = json.loads(data)
            experiments.append(exp)
        except json.JSONDecodeError:
            st.warning(f"Failed to parse {name}")
    return experiments


files_data = [(f.name, f.read()) for f in uploaded_files]
experiments = parse_experiments(files_data)

if not experiments:
    st.error("No valid experiments found.")
    st.stop()

st.success(f"Loaded {len(experiments)} experiments")


# ── Build dataframe ─���──────────────────────────────────────────────────────


def build_summary_df(experiments: list[dict]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        config = exp.get("config", {})
        bench = exp.get("benchmark", {})
        smoke = exp.get("smoke_tests", {})
        scores = exp.get("scores", {})

        rows.append({
            "experiment_id": exp.get("experiment_id", ""),
            "engine": exp.get("engine", ""),
            "status": exp.get("status", ""),
            "tp": config.get("tensor_parallel_size", 1),
            "pp": config.get("pipeline_parallel_size", 1),
            "dp": config.get("data_parallel_size", 1),
            "max_model_len": config.get("max_model_len"),
            "quantization": config.get("quantization", "none"),
            "dtype": config.get("dtype", "auto"),
            "kv_cache_dtype": config.get("kv_cache_dtype", "auto"),
            "chunked_prefill": config.get("enable_chunked_prefill", False),
            "prefix_caching": config.get("enable_prefix_caching", False),
            "enforce_eager": config.get("enforce_eager", False),
            "scheduling_policy": config.get("scheduling_policy", "fcfs"),
            "peak_throughput": bench.get("peak_output_tokens_per_sec", 0),
            "peak_total_throughput": bench.get("peak_total_tokens_per_sec", 0),
            "ttft_p95": bench.get("low_concurrency_ttft_p95_ms", 0),
            "tpot_p95": bench.get("low_concurrency_tpot_p95_ms", 0),
            "kv_cache_usage": bench.get("kv_cache_usage_percent", 0),
            "prefix_hit_rate": bench.get("prefix_cache_hit_rate", 0),
            "smoke_tool": smoke.get("tool_calling", False),
            "smoke_json": smoke.get("json_mode", False),
            "smoke_schema": smoke.get("json_schema", False),
            "throughput_score": scores.get("throughput_score", 0),
            "latency_score": scores.get("latency_score", 0),
            "balanced_score": scores.get("balanced_score", 0),
            "is_pareto": scores.get("is_pareto_optimal", False),
            "classification": exp.get("optimization_classification", "none"),
            "commentary": exp.get("llm_commentary", ""),
            "duration_s": exp.get("duration_seconds", 0),
        })

    return pd.DataFrame(rows)


df = build_summary_df(experiments)


# ── Filters ────────────────────────────────────────────────────────────────

st.sidebar.header("Filters")

engines = st.sidebar.multiselect(
    "Engine", df["engine"].unique().tolist(), default=df["engine"].unique().tolist()
)
quants = st.sidebar.multiselect(
    "Quantization",
    df["quantization"].unique().tolist(),
    default=df["quantization"].unique().tolist(),
)
tp_values = st.sidebar.multiselect(
    "Tensor Parallel",
    sorted(df["tp"].unique().tolist()),
    default=sorted(df["tp"].unique().tolist()),
)

filtered = df[
    df["engine"].isin(engines)
    & df["quantization"].isin(quants)
    & df["tp"].isin(tp_values)
]


# ── Leaderboards ───────────────────────────────────────────────────────────

st.header("Leaderboards")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Best Throughput")
    top_tp = filtered.nlargest(5, "peak_throughput")[
        ["experiment_id", "engine", "quantization", "tp", "peak_throughput", "ttft_p95"]
    ]
    st.dataframe(top_tp, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Best Latency")
    valid_lat = filtered[filtered["ttft_p95"] > 0]
    top_lat = valid_lat.nsmallest(5, "ttft_p95")[
        ["experiment_id", "engine", "quantization", "tp", "ttft_p95", "peak_throughput"]
    ]
    st.dataframe(top_lat, use_container_width=True, hide_index=True)

with col3:
    st.subheader("Best Balanced")
    top_bal = filtered.nlargest(5, "balanced_score")[
        [
            "experiment_id", "engine", "quantization", "tp",
            "balanced_score", "peak_throughput", "ttft_p95",
        ]
    ]
    st.dataframe(top_bal, use_container_width=True, hide_index=True)


# ── Pareto Chart ─────────��─────────────────────────────────────────────────

st.header("Throughput vs Latency (Pareto Front)")

valid = filtered[(filtered["peak_throughput"] > 0) & (filtered["ttft_p95"] > 0)]

if not valid.empty:
    fig = px.scatter(
        valid,
        x="ttft_p95",
        y="peak_throughput",
        color="engine",
        symbol="quantization",
        size="tp",
        hover_data=["experiment_id", "scheduling_policy", "chunked_prefill"],
        labels={
            "ttft_p95": "TTFT p95 (ms) ← lower is better",
            "peak_throughput": "Peak Throughput (tok/s) ↑",
        },
    )

    # Highlight Pareto-optimal points
    pareto_pts = valid[valid["is_pareto"]]
    if not pareto_pts.empty:
        pareto_sorted = pareto_pts.sort_values("ttft_p95")
        fig.add_trace(go.Scatter(
            x=pareto_sorted["ttft_p95"],
            y=pareto_sorted["peak_throughput"],
            mode="lines",
            name="Pareto Front",
            line=dict(color="red", dash="dash", width=2),
        ))

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No valid data points for Pareto chart.")


# ── Concurrency Curves ────────────────────────────────────────────────────

st.header("Throughput vs Concurrency")

selected_exp = st.selectbox(
    "Select experiment",
    [e.get("experiment_id", "") for e in experiments],
    format_func=lambda x: next(
        (
            f"{e.get('experiment_id', '')} ({e.get('engine', '')} "
            f"TP={e.get('config', {}).get('tensor_parallel_size', '?')} "
            f"q={e.get('config', {}).get('quantization', 'none')})"
            for e in experiments
            if e.get("experiment_id") == x
        ),
        x,
    ),
)

if selected_exp:
    exp_data = next((e for e in experiments if e.get("experiment_id") == selected_exp), None)
    if exp_data:
        conc_results = exp_data.get("benchmark", {}).get("concurrency_results", [])
        if conc_results:
            conc_df = pd.DataFrame(conc_results)

            col1, col2 = st.columns(2)

            with col1:
                fig_tp = px.line(
                    conc_df,
                    x="concurrency",
                    y="output_tokens_per_sec",
                    color="prompt_length",
                    markers=True,
                    title="Output Throughput by Concurrency",
                    labels={
                        "output_tokens_per_sec": "Output tok/s",
                        "concurrency": "Concurrency",
                    },
                )
                st.plotly_chart(fig_tp, use_container_width=True)

            with col2:
                # TTFT p95 from nested stats
                ttft_data = []
                for cr in conc_results:
                    ttft = cr.get("ttft_ms", {})
                    ttft_data.append({
                        "concurrency": cr["concurrency"],
                        "prompt_length": cr["prompt_length"],
                        "ttft_p50": ttft.get("median", 0),
                        "ttft_p95": ttft.get("p95", 0),
                        "ttft_p99": ttft.get("p99", 0),
                    })
                ttft_df = pd.DataFrame(ttft_data)

                fig_lat = px.line(
                    ttft_df,
                    x="concurrency",
                    y="ttft_p95",
                    color="prompt_length",
                    markers=True,
                    title="TTFT p95 by Concurrency",
                    labels={
                        "ttft_p95": "TTFT p95 (ms)",
                        "concurrency": "Concurrency",
                    },
                )
                st.plotly_chart(fig_lat, use_container_width=True)


# ── Long Context Analysis ─────────────────────────────────────────────────

st.header("Long Context Performance")

long_ctx_data = []
for exp in experiments:
    conc_results = exp.get("benchmark", {}).get("concurrency_results", [])
    for cr in conc_results:
        if cr.get("prompt_length", 0) >= 4096:
            long_ctx_data.append({
                "experiment_id": exp.get("experiment_id", ""),
                "engine": exp.get("engine", ""),
                "prompt_length": cr["prompt_length"],
                "concurrency": cr["concurrency"],
                "output_tokens_per_sec": cr.get("output_tokens_per_sec", 0),
                "ttft_p95": cr.get("ttft_ms", {}).get("p95", 0),
                "quantization": exp.get("config", {}).get("quantization", "none"),
            })

if long_ctx_data:
    lc_df = pd.DataFrame(long_ctx_data)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            lc_df,
            x="prompt_length",
            y="output_tokens_per_sec",
            color="engine",
            title="Throughput by Context Length",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            lc_df,
            x="prompt_length",
            y="ttft_p95",
            color="engine",
            title="TTFT p95 by Context Length",
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No long context data available.")


# ── GPU Utilization ────────────────────────────────────────────────────────

st.header("GPU Metrics")

gpu_data = []
for exp in experiments:
    bench = exp.get("benchmark", {})
    gpu_util = bench.get("gpu_utilization_percent", [])
    gpu_mem = bench.get("gpu_memory_used_mb", [])
    gpu_power = bench.get("gpu_power_draw_watts", [])
    for i, (u, m, p) in enumerate(
        zip(gpu_util, gpu_mem, gpu_power, strict=False)
    ):
        gpu_data.append({
            "experiment_id": exp.get("experiment_id", ""),
            "engine": exp.get("engine", ""),
            "gpu_index": i,
            "utilization": u,
            "memory_mb": m,
            "power_w": p,
        })

if gpu_data:
    gpu_df = pd.DataFrame(gpu_data)
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            gpu_df,
            x="experiment_id",
            y="utilization",
            color="gpu_index",
            barmode="group",
            title="GPU Utilization (%)",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            gpu_df,
            x="experiment_id",
            y="memory_mb",
            color="gpu_index",
            barmode="group",
            title="GPU Memory Usage (MB)",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No GPU metrics available.")


# ── LLM Commentary ────────────────────────────────────────────────────────

st.header("LLM Analysis")

for exp in experiments:
    commentary = exp.get("llm_commentary", "")
    if commentary:
        eid = exp.get("experiment_id", "")
        engine = exp.get("engine", "")
        classification = exp.get("optimization_classification", "none")
        with st.expander(f"{eid} ({engine}) — {classification}"):
            st.write(commentary)
            smoke = exp.get("smoke_tests", {})
            st.write(
                f"Smoke tests: tool_calling={'✓' if smoke.get('tool_calling') else '✗'}, "
                f"json_mode={'✓' if smoke.get('json_mode') else '✗'}, "
                f"json_schema={'✓' if smoke.get('json_schema') else '✗'}"
            )


# ── Full comparison table ─────────────────────────────────────────────────

st.header("Full Comparison")
st.dataframe(
    filtered.sort_values("peak_throughput", ascending=False),
    use_container_width=True,
    hide_index=True,
)
