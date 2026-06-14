"""LangGraph agent — main graph assembly."""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from inference_agent.api_client import ExperimentApiClient
from inference_agent.nodes.analyzer import analyzer_node
from inference_agent.nodes.baseline_injector import baseline_injector_node
from inference_agent.nodes.crash_diagnostician import crash_diagnostician_node
from inference_agent.nodes.discovery import discovery_node
from inference_agent.nodes.executor import executor_node
from inference_agent.nodes.history_loader import make_history_loader_node
from inference_agent.nodes.planner import planner_node
from inference_agent.nodes.reporter import make_reporter_node
from inference_agent.nodes.validator import validator_node
from inference_agent.state import AgentState

logger = logging.getLogger(__name__)


def _should_continue(state: AgentState) -> str:
    """Decide whether to continue experimenting or stop."""
    if state.get("status") == "completed":
        return "end"
    if state.get("status") == "failed":
        return "end"
    return "continue"


def _after_validator(state: AgentState) -> str:
    """Route after validation: run executor or skip to analyzer."""
    if state.get("skip_executor"):
        return "skip"
    return "run"


def _after_history_loader(state: AgentState) -> str:
    """Route after history load: inject the baseline anchor or plan normally.

    The baseline runs deterministically as the first experiment (no LLM) when
    one is configured and not yet measured for this hardware/model. Every other
    iteration goes through the planner.
    """
    if state.get("baseline_pending"):
        return "baseline"
    return "plan"


def build_graph(client: ExperimentApiClient) -> StateGraph:
    """Build the LangGraph agent graph.

    `history_loader` and `reporter` are client-bound (closure DI); other
    nodes are plain async functions.
    """
    graph = StateGraph(AgentState)

    graph.add_node("discovery", discovery_node)
    graph.add_node("history_loader", make_history_loader_node(client))
    graph.add_node("baseline_injector", baseline_injector_node)
    graph.add_node("planner", planner_node)
    graph.add_node("validator", validator_node)
    graph.add_node("executor", executor_node)
    graph.add_node("crash_diagnostician", crash_diagnostician_node)
    graph.add_node("reporter", make_reporter_node(client))
    graph.add_node("analyzer", analyzer_node)

    graph.set_entry_point("discovery")

    # Flow: discovery → history_loader → (baseline_injector | planner) →
    # validator → executor → crash_diagnostician → analyzer → reporter → loop.
    # The first iteration may inject the operator baseline (deterministic, no
    # LLM); every later iteration plans via the LLM. If validation fails, skip
    # executor and go directly to analyzer. crash_diagnostician is a no-op
    # unless the executor produced a container-crash result.
    graph.add_edge("discovery", "history_loader")
    graph.add_conditional_edges(
        "history_loader",
        _after_history_loader,
        {
            "baseline": "baseline_injector",
            "plan": "planner",
        },
    )
    graph.add_edge("baseline_injector", "validator")
    graph.add_edge("planner", "validator")

    graph.add_conditional_edges(
        "validator",
        _after_validator,
        {
            "run": "executor",
            "skip": "analyzer",
        },
    )

    graph.add_edge("executor", "crash_diagnostician")
    graph.add_edge("crash_diagnostician", "analyzer")
    graph.add_edge("analyzer", "reporter")

    graph.add_conditional_edges(
        "reporter",
        _should_continue,
        {
            "continue": "planner",
            "end": END,
        },
    )

    return graph


def compile_agent(client: ExperimentApiClient):
    """Compile and return the runnable agent."""
    graph = build_graph(client)
    return graph.compile()
