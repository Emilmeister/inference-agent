"""Tests for the agentic multi-turn session runner.

Spins up a minimal aiohttp SSE backend and verifies that _run_agent_session
correctly grows the message history across turns, captures per-turn metrics,
and breaks on error.
"""

from __future__ import annotations

import json
import random
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from inference_agent.benchmark.runner import (
    _run_agent_session,
    run_agentic_long_context_phase,
)


def _sse_bytes(events: list[dict]) -> bytes:
    """Encode a list of dict events as an SSE stream."""
    out: list[str] = []
    for ev in events:
        out.append(f"data: {json.dumps(ev)}\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out).encode("utf-8")


def _delta_event(content: str) -> dict:
    return {
        "choices": [{"index": 0, "delta": {"content": content}}],
    }


def _usage_event(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class _TurnRecorder:
    """Captures per-turn messages so tests can assert on history growth."""

    def __init__(self):
        self.turns: list[list[dict]] = []
        # Behavior knobs:
        self.error_on_turn: int | None = None  # return HTTP 500 on this turn (0-based)
        self.empty_on_turn: int | None = None  # stream zero tokens on this turn
        self.fixed_assistant_text = "Reply text from the model."

    async def handler(self, request: web.Request) -> web.StreamResponse:
        body: dict[str, Any] = await request.json()
        turn_idx = len(self.turns)
        self.turns.append(body["messages"])

        if self.error_on_turn == turn_idx:
            return web.Response(status=500, text="injected failure")

        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await resp.prepare(request)

        if self.empty_on_turn == turn_idx:
            # Zero-token response → triggers "Empty response" error in helper.
            await resp.write(_sse_bytes([_usage_event(prompt_tokens=10, completion_tokens=0)]))
            await resp.write_eof()
            return resp

        events = [
            _delta_event("Reply "),
            _delta_event("text "),
            _delta_event("from "),
            _delta_event("the model."),
            _usage_event(
                prompt_tokens=100 + turn_idx * 50,
                completion_tokens=4,
            ),
        ]
        await resp.write(_sse_bytes(events))
        await resp.write_eof()
        return resp


async def _start_server(handler):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}/v1"


@pytest.mark.asyncio
class TestRunAgentSession:
    async def test_history_grows_across_turns(self):
        rec = _TurnRecorder()
        runner, base = await _start_server(rec.handler)
        try:
            url = f"{base}/chat/completions"
            async with aiohttp.ClientSession() as http:
                rng = random.Random(42)
                results = await _run_agent_session(
                    session_idx=0,
                    http_session=http,
                    url=url,
                    model="test-model",
                    prefix_tokens=64,         # ~256 chars
                    turns=3,
                    max_output_tokens=64,
                    tool_result_min=16,
                    tool_result_max=32,
                    rng=rng,
                    per_turn_timeout_sec=10,
                )
        finally:
            await runner.cleanup()

        # 3 successful turns recorded.
        assert len(results) == 3
        for i, r in enumerate(results):
            assert r["turn_idx"] == i
            assert r["session_idx"] == 0
            assert r["error"] is None
            assert r["output_tokens"] == 4
            assert r["ttft_ms"] > 0

        # History on turn 0 = just the prefix [user]; turn 1 = [user, assistant, user];
        # turn 2 = [user, assistant, user, assistant, user]. We add 2 messages per turn.
        assert len(rec.turns[0]) == 1
        assert len(rec.turns[1]) == 3
        assert len(rec.turns[2]) == 5

        # Roles alternate user → assistant → user → assistant → user
        roles_t2 = [m["role"] for m in rec.turns[2]]
        assert roles_t2 == ["user", "assistant", "user", "assistant", "user"]

        # Tool result is wrapped with the [tool_result] sentinel.
        assert rec.turns[2][2]["content"].startswith("[tool_result]\n")
        assert rec.turns[2][4]["content"].startswith("[tool_result]\n")

        # input_tokens grows turn-over-turn (we report usage.prompt_tokens).
        assert results[0]["input_tokens"] == 100
        assert results[1]["input_tokens"] == 150
        assert results[2]["input_tokens"] == 200

    async def test_error_on_turn_breaks_session(self):
        """HTTP 500 on turn 1 → session stops, only 2 records returned (turn 0 ok + turn 1 err)."""
        rec = _TurnRecorder()
        rec.error_on_turn = 1
        runner, base = await _start_server(rec.handler)
        try:
            url = f"{base}/chat/completions"
            async with aiohttp.ClientSession() as http:
                rng = random.Random(0)
                results = await _run_agent_session(
                    session_idx=7,
                    http_session=http,
                    url=url,
                    model="test-model",
                    prefix_tokens=32,
                    turns=4,                   # would do 4 turns but should stop at 2
                    max_output_tokens=32,
                    tool_result_min=8,
                    tool_result_max=16,
                    rng=rng,
                    per_turn_timeout_sec=10,
                )
        finally:
            await runner.cleanup()

        assert len(results) == 2
        assert results[0]["error"] is None
        assert results[0]["turn_idx"] == 0
        assert results[1]["error"] is not None
        assert "HTTP 500" in results[1]["error"]
        assert results[1]["turn_idx"] == 1
        # Server should have been called exactly twice.
        assert len(rec.turns) == 2

    async def test_empty_response_treated_as_error_and_stops(self):
        """Zero-token response on turn 1 → marked error and session stops."""
        rec = _TurnRecorder()
        rec.empty_on_turn = 1
        runner, base = await _start_server(rec.handler)
        try:
            url = f"{base}/chat/completions"
            async with aiohttp.ClientSession() as http:
                rng = random.Random(0)
                results = await _run_agent_session(
                    session_idx=0,
                    http_session=http,
                    url=url,
                    model="test-model",
                    prefix_tokens=32,
                    turns=3,
                    max_output_tokens=32,
                    tool_result_min=8,
                    tool_result_max=16,
                    rng=rng,
                    per_turn_timeout_sec=10,
                )
        finally:
            await runner.cleanup()

        assert len(results) == 2
        assert results[1]["error"] is not None and "Empty response" in results[1]["error"]


@pytest.mark.asyncio
class TestRunAgenticLongContextPhase:
    async def test_aggregates_all_turns_across_sessions(self):
        rec = _TurnRecorder()
        runner, base = await _start_server(rec.handler)
        try:
            result = await run_agentic_long_context_phase(
                api_base_url=base,
                model_name="test-model",
                concurrency=2,
                prefix_tokens=32,
                max_output_tokens=32,
                turns=2,
                tool_result_min=8,
                tool_result_max=16,
                session_timeout_sec=30,
                per_turn_timeout_sec=10,
                seed=123,
                workload_id="agentic_long_context",
                phase_id="agentic_c2_p32",
            )
        finally:
            await runner.cleanup()

        # 2 sessions × 2 turns = 4 turn-requests.
        assert result.num_requests == 4
        assert len(result.agentic_turn_metrics) == 4
        assert result.workload_id == "agentic_long_context"
        assert result.phase_id == "agentic_c2_p32"
        assert result.errors == 0
        assert result.output_tokens_per_sec > 0

        # All turn_idx values present (0 and 1).
        turn_ids = sorted(m.turn_idx for m in result.agentic_turn_metrics)
        assert turn_ids == [0, 0, 1, 1]
