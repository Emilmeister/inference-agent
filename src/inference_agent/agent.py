"""LangGraph agent — main graph assembly."""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from inference_agent.nodes.analyzer import analyzer_node
from inference_agent.nodes.discovery import discovery_node
from inference_agent.nodes.executor import executor_node
from inference_agent.nodes.planner import planner_node
from inference_agent.nodes.reporter import reporter_node
from inference_agent.state import AgentState

logger = logging.getLogger(__name__)


def _should_continue(state: AgentState) -> str:
    """Decide whether to continue experimenting or stop."""
    if state.get("status") == "completed":
        return "end"
    if state.get("status") == "failed":
        return "end"
    return "continue"


def build_graph() -> StateGraph:
    """Build the LangGraph agent graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("discovery", discovery_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reporter", reporter_node)
    graph.add_node("analyzer", analyzer_node)

    # Set entry point
    graph.set_entry_point("discovery")

    # Define edges
    graph.add_edge("discovery", "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reporter")
    graph.add_edge("reporter", "analyzer")

    # Conditional edge from analyzer
    graph.add_conditional_edges(
        "analyzer",
        _should_continue,
        {
            "continue": "planner",
            "end": END,
        },
    )

    return graph


def compile_agent():
    """Compile and return the runnable agent."""
    graph = build_graph()
    return graph.compile()
