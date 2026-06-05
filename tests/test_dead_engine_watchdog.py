"""Tests for the dead-engine watchdogs in the benchmark runner and executor.

Two layers cooperate to stop the agent from burning a 30 s phase (or a dozen
of them) firing requests into a socket that nobody is listening on:

  * `_is_connection_failure` — classifies a single request error string as
    "the upstream is gone" vs "the upstream is unhappy but reachable".
  * `_looks_like_dead_engine_phase` — promotes a phase to "dead" only when
    error_rate is at saturation AND most samples look connection-shaped.

The executor uses the phase-level signal to break the sweep after a couple
of consecutive deaths; the runner uses the per-request signal to bail out
of the duration loop. We don't unit-test the full async loop here — that's
covered by integration runs — but we do nail down the classifiers since
they're load-bearing for both decisions.
"""

from inference_agent.benchmark.runner import _is_connection_failure
from inference_agent.models import ConcurrencyResult, PercentileStats

from inference_agent.nodes.executor import _looks_like_dead_engine_phase


class TestConnectionFailureClassifier:
    def test_none_and_empty(self):
        assert _is_connection_failure(None) is False
        assert _is_connection_failure("") is False

    def test_aiohttp_cannot_connect(self):
        assert _is_connection_failure(
            "Cannot connect to host localhost:8000 ssl:default "
            "[Connect call failed ('127.0.0.1', 8000)]"
        )

    def test_kernel_econnrefused(self):
        assert _is_connection_failure("OSError: [Errno 111] Connection refused")

    def test_connection_reset_mid_stream(self):
        assert _is_connection_failure(
            "ServerDisconnectedError: Server disconnected"
        )

    def test_case_insensitive(self):
        assert _is_connection_failure("CONNECTION REFUSED on attempt 3")

    def test_not_a_connection_failure(self):
        # Real engine-side errors that should NOT trip the watchdog: the
        # server is alive and answering, just refusing this specific request.
        assert _is_connection_failure("HTTP 500 Internal Server Error") is False
        assert _is_connection_failure(
            "json.decoder.JSONDecodeError: Expecting value"
        ) is False
        assert _is_connection_failure("Per-turn timeout after 60s") is False
        assert _is_connection_failure("Session timeout after 600s") is False


def _make_phase(
    *,
    error_rate: float,
    error_details: list[str],
    errors: int | None = None,
) -> ConcurrencyResult:
    """Minimal ConcurrencyResult — enough for the dead-phase classifier."""
    n = errors if errors is not None else max(1, len(error_details))
    return ConcurrencyResult(
        concurrency=8,
        prompt_length=512,
        max_output_tokens=256,
        num_requests=n,
        workload_id="agent_short",
        phase_id="c8_p512",
        ttft_ms=PercentileStats(),
        tpot_ms=PercentileStats(),
        itl_ms=PercentileStats(),
        e2e_latency_ms=PercentileStats(),
        errors=n if error_rate >= 1.0 else int(n * error_rate),
        error_rate=error_rate,
        error_details=error_details[:10],
    )


class TestDeadPhaseClassifier:
    def test_full_saturation_with_connection_errors(self):
        # Classic engine-died phase: every recorded sample is socket-shaped.
        phase = _make_phase(
            error_rate=1.0,
            error_details=[
                "Cannot connect to host localhost:8000",
                "Cannot connect to host localhost:8000",
                "Connection refused",
            ],
        )
        assert _looks_like_dead_engine_phase(phase) is True

    def test_partial_failures_not_dead(self):
        # Real load that hit timeouts at high concurrency — NOT a dead engine,
        # just an overloaded one. Must not trip the watchdog.
        phase = _make_phase(
            error_rate=0.5,
            error_details=[
                "Cannot connect to host localhost:8000",
                "Cannot connect to host localhost:8000",
            ],
        )
        assert _looks_like_dead_engine_phase(phase) is False

    def test_saturation_but_timeout_shaped(self):
        # 100% errors but they're per-turn timeouts (engine is up but slow on
        # this concurrency). This is the agentic-ceiling case — handled by a
        # different code path; the dead-engine watchdog must stay quiet.
        phase = _make_phase(
            error_rate=1.0,
            error_details=[
                "Per-turn timeout after 60s",
                "Session timeout after 600s",
                "Per-turn timeout after 60s",
            ],
        )
        assert _looks_like_dead_engine_phase(phase) is False

    def test_saturation_with_majority_connection_errors(self):
        # One straggler timeout among connection errors still counts as dead.
        phase = _make_phase(
            error_rate=1.0,
            error_details=[
                "Cannot connect to host localhost:8000",
                "Cannot connect to host localhost:8000",
                "Connection refused",
                "Per-turn timeout after 60s",
            ],
        )
        assert _looks_like_dead_engine_phase(phase) is True

    def test_no_samples_not_dead(self):
        # error_rate at saturation but no captured details — can't tell what
        # went wrong, don't blow up the experiment on a guess.
        phase = _make_phase(error_rate=1.0, error_details=[], errors=100)
        assert _looks_like_dead_engine_phase(phase) is False
