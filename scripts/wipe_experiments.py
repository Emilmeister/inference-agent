"""Wipe ALL experiments from inference-api.

DANGER: this irreversibly deletes every experiment row, JSONB payload, and
per-phase/per-turn artefact in the database. There is no soft-delete and no
backup. Run only when you intend to start a fresh leaderboard.

Usage:
  export INFERENCE_API_URL=http://localhost:8080
  export INFERENCE_API_TOKEN=...
  python scripts/wipe_experiments.py [--yes]

Without `--yes` the script lists what it would delete and exits.
"""

from __future__ import annotations

import argparse
import os
import sys

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion. Without this flag the script is a dry-run.",
    )
    args = parser.parse_args()

    url = os.environ.get("INFERENCE_API_URL")
    token = os.environ.get("INFERENCE_API_TOKEN")
    if not url or not token:
        print("ERROR: INFERENCE_API_URL and INFERENCE_API_TOKEN must be set.", file=sys.stderr)
        return 2

    base = url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    print(f"Fetching experiment list from {base}/experiments ...")
    resp = requests.get(f"{base}/experiments", headers=headers, timeout=120)
    resp.raise_for_status()
    rows = resp.json()
    ids = [r["experiment_id"] for r in rows if r.get("experiment_id")]
    print(f"Found {len(ids)} experiments.")
    if not ids:
        return 0

    if not args.yes:
        print("Dry-run. Re-run with --yes to permanently delete:")
        for eid in ids[:10]:
            print(f"  - {eid}")
        if len(ids) > 10:
            print(f"  ... and {len(ids) - 10} more.")
        return 0

    print(f"Deleting {len(ids)} experiments ...")
    resp = requests.delete(
        f"{base}/experiments",
        headers={**headers, "Content-Type": "application/json"},
        json={"experiment_ids": ids},
        timeout=300,
    )
    resp.raise_for_status()
    body = resp.json()
    print(f"Deleted: {body.get('deleted', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
