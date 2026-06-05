#!/usr/bin/env bash
# start.sh — launch inference-agent in the background with the right env.
#
# Why this script exists:
#   1. The agent needs INFERENCE_API_TOKEN, AGENT_API_BASE_URL and the LLM
#      keys in its environment. We keep them in `.env` next to the repo so
#      they aren't committed.
#   2. The agent ultimately runs under `sudo -E` (nerdctl needs root on most
#      setups). `sudo -E` only preserves env vars that are ALREADY in the
#      caller's shell — it cannot read .env on its own. So we source .env
#      first, THEN sudo -E.
#   3. nerdctl reaches HuggingFace and the inference-api endpoint through
#      HTTP_PROXY on locked-down VMs. That proxy lives in .env too, same
#      story: must be loaded into the shell before sudo.
#
# Usage:
#   ./scripts/start.sh                 # uses config.yaml in the repo root
#   ./scripts/start.sh path/to/config.yaml
#
# Output:
#   The previous `log` (if any) is rotated to `log.<timestamp>` and the new
#   run streams to `log`. Tail it with `tail -f log`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${1:-config.yaml}"
if [[ ! -f "$CONFIG" ]]; then
    echo "start.sh: config not found: $CONFIG" >&2
    exit 2
fi

# Load .env into the current shell so `sudo -E` can forward it.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    . .env
    set +a
else
    echo "start.sh: WARNING — no .env in $REPO_ROOT; relying on inherited env" >&2
fi

# Fail fast on missing required vars — better than dying inside a python
# stack trace 30s later because the agent couldn't reach the REST service.
: "${INFERENCE_API_TOKEN:?INFERENCE_API_TOKEN missing — set it in .env}"
: "${AGENT_API_BASE_URL:?AGENT_API_BASE_URL missing — set it in .env}"
: "${AGENT_LLM_BASE_URL:?AGENT_LLM_BASE_URL missing — set it in .env}"

# Rotate previous log so we never tail into stale output.
if [[ -f log ]]; then
    mv log "log.$(date +%Y%m%d-%H%M%S)"
fi

# Locate the agent entrypoint. `pip install -e .` puts it under
# ~/.local/bin on most distros; fall back to the system PATH.
AGENT_BIN="${AGENT_BIN:-${HOME}/.local/bin/inference-agent}"
if [[ ! -x "$AGENT_BIN" ]]; then
    if command -v inference-agent >/dev/null 2>&1; then
        AGENT_BIN="$(command -v inference-agent)"
    else
        echo "start.sh: inference-agent binary not found; pip install -e . first" >&2
        exit 3
    fi
fi

# PYTHONPATH covers the dev-install (src/) plus the user site-packages where
# pip installed our deps. Without the latter, sudo's reset of HOME would
# make the imports fail.
PYPATH="${REPO_ROOT}/src:${HOME}/.local/lib/python3.10/site-packages"

nohup sudo -E PYTHONPATH="$PYPATH" "$AGENT_BIN" -c "$CONFIG" -v > log 2>&1 &
NEW_PID=$!
disown "$NEW_PID" 2>/dev/null || true

# Give it 2 seconds, then sanity-check the process is still alive — if the
# python entrypoint died on a missing var, we want to know NOW, not when
# someone notices `log` is full of a single traceback.
sleep 2
if ! kill -0 "$NEW_PID" 2>/dev/null; then
    echo "start.sh: agent died immediately; tail of log:" >&2
    tail -20 log >&2 || true
    exit 1
fi

echo "started: shell-PID=$NEW_PID, logs: tail -f $REPO_ROOT/log"
