"""Frozen evidence for the semantic relation-alignment construction route.

This artifact deliberately does not rerun the search.  The exploratory run was
completed in ``/tmp`` and froze its JSON evidence before packaging.  Keeping
the executable as a verifier makes the failure reproducible without silently
changing lexical pools, budgets, or the novelty baseline.

The route's construction state is a directed relation edge between two event
frames.  Each endpoint is independently lexicalized, while a boundary-aware
ledger consumes one character from the left edge and one from the reversed
right edge.  It is exhaustive recursion over lexical boundaries: no beam,
MCTS, chart, CSP, ILP, static palindrome, or catalogue text is used.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "runs" / "semantic-relation-alignment-20260916.json"
EXPERIMENT_ID = "semantic-relation-alignment-20260916"
SIGNATURE = (
    "semantic-relation-alignment|directed-event-edge-pairing|"
    "independent-role-lexicalization|boundary-synchronous-character-ledger|"
    "deterministic-exhaustive"
)

# These are the four semantic edge types actually explored.  They are kept in
# the artifact so a future repair cannot accidentally change the route while
# claiming to reproduce this run.
FRAME_EDGES = {
    "preserve_then_recover": ("preserves", "enables"),
    "signal_then_respond": ("signals", "elicits"),
    "plant_then_grow": ("plants", "causes"),
    "measure_then_adjust": ("measures", "guides"),
}


def _tape(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.casefold()))


def replay_ledger(text: str) -> bool:
    """Recheck every mirrored character in a rendered candidate."""
    tape = _tape(text)
    return bool(tape) and all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2))


def load_evidence() -> dict:
    """Load and fail closed on the packaged, non-reader evidence."""
    payload = json.loads(EVIDENCE.read_text())
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("evidence experiment id does not match artifact")
    if payload.get("signature") != SIGNATURE:
        raise ValueError("evidence signature does not match artifact")
    if payload.get("stats", {}).get("exact", 0) != 0:
        raise ValueError("packaged failure evidence unexpectedly contains an exact row")
    if payload.get("stats", {}).get("admitted", 0) != 0:
        raise ValueError("packaged failure evidence unexpectedly contains an admitted row")
    return payload


def run() -> dict:
    """Return frozen evidence; the failed exploratory search is not repeated."""
    return load_evidence()


def main() -> None:
    payload = run()
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "stats": payload["stats"]}, sort_keys=True))


if __name__ == "__main__":
    main()
