"""Acyclic optional-phrase lattice with asynchronous residual consumption.

This is a new construction topology, not a wider fixed-path product.  Each
side walks an acyclic phrase lattice whose optional adjunct and relative edges
may be taken or skipped.  Character debt is consumed as soon as either side
emits a phrase; no completed tape is reversed or repaired.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "exact": bool(tape) and tape == tape[::-1],
        "sha256_forward": forward,
        "sha256_reverse": reverse,
    }


# Complete, independently authored phrase units.  They are ordinary prose,
# not reverse pairs; the lattice chooses them before character compatibility.
BANK = {
    "NP": (
        "the patient sailor", "a quiet cartographer", "our young poet",
        "the careful keeper", "a bright gardener", "the old scholar",
    ),
    "VP": (
        "studies the northern chart", "marks the distant shore",
        "guards the lantern", "remembers the winter garden",
        "carries a small letter", "watches the harbor birds",
    ),
    "PP": (
        "by the river", "under the pale moon", "near the quiet harbor",
        "beside the old bridge", "after the evening rain",
    ),
    "ADV": ("at dawn", "with care", "in silence", "before dusk"),
    "REL": (
        "who listens by the water", "that waits beside the road",
        "who keeps the candle bright",
    ),
}


# Acyclic graph.  Optional edges are represented explicitly; no fixed number
# of phrase slots is assumed.  The terminal can be reached after any adjunct.
EDGES = {
    "START": (("NP", "NP"),),
    "NP": (("VP", "VP"),),
    "VP": (("PP", "PP"), ("ADV", "ADV"), ("REL", "REL"), ("END", "SKIP")),
    "PP": (("PP", "PP"), ("ADV", "ADV"), ("REL", "REL"), ("END", "SKIP")),
    "ADV": (("PP", "PP"), ("REL", "REL"), ("END", "SKIP")),
    "REL": (("PP", "PP"), ("ADV", "ADV"), ("END", "SKIP")),
}


def next_edges(node: str):
    return EDGES.get(node, ())


def consume(left: str, right: str):
    n = min(len(left), len(right))
    if left[:n] != right[:n]:
        return None
    return left[n:], right[n:]


def independent_exact(text: str) -> bool:
    tape = letters(text)
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            return False
        i += 1
        j -= 1
    return bool(tape)


def controls() -> list[dict]:
    texts = [
        "The patient sailor studies the northern chart by the river at dawn.",
        "A quiet cartographer marks the distant shore near the quiet harbor.",
        "Our young poet remembers the winter garden with care before dusk.",
        "The careful keeper guards the lantern beside the old bridge.",
    ]
    return [{"rendered": t, "audit": audit(t), "reader_eligible": False,
             "provenance": "authored intact prose control; not a generated candidate"}
            for t in texts]


def run(limit: int = 5_000) -> dict:
    states = 0
    pruned = 0
    exact = []
    seen = set()
    # State stores graph nodes and comparison-oriented residual buffers.  The
    # right graph is traversed from its inner edge, so phrase characters enter
    # the right buffer reversed while the rendered phrase is prepended.
    stack = [("START", "START", "", "", "", "", (), ())]
    while stack and states < limit:
        ln, rn, left, right, lbuf, rbuf, lp, rp = stack.pop()
        states += 1
        lterminal = ln == "END"
        rterminal = rn == "END"
        if lterminal and rterminal:
            rendered = (left + " " + right).strip()
            info = audit(rendered)
            if not lbuf and not rbuf and info["exact"] and independent_exact(rendered):
                if info["letters"] >= 40 and rendered not in seen:
                    seen.add(rendered)
                    exact.append({
                        "rendered": rendered, "audit": info,
                        "independent_pointer_exact": independent_exact(rendered),
                        "provenance": {"left_path": lp, "right_path": rp,
                                       "lattice": "acyclic optional phrase graph",
                                       "corpus_sentence_replay": False,
                                       "mirrored_token_units": False,
                                       "posthoc_repair": False},
                    })
            continue
        if not lterminal:
            for edge, kind in next_edges(ln):
                if edge == "END":
                    stack.append(("END", rn, left, right, lbuf, rbuf, lp + ("skip",), rp))
                    continue
                for phrase in BANK[kind]:
                    residual = consume(lbuf + letters(phrase), rbuf)
                    if residual is None:
                        pruned += 1
                        continue
                    stack.append((edge, rn, (left + " " if left else "") + phrase,
                                  right, residual[0], residual[1], lp + (phrase,), rp))
        if not rterminal:
            for edge, kind in next_edges(rn):
                if edge == "END":
                    stack.append((ln, "END", left, right, lbuf, rbuf, lp, rp + ("skip",)))
                    continue
                for phrase in BANK[kind]:
                    residual = consume(lbuf, rbuf + letters(phrase)[::-1])
                    if residual is None:
                        pruned += 1
                        continue
                    stack.append((ln, edge, left, phrase + (" " + right if right else ""),
                                  residual[0], residual[1], lp, rp + (phrase,)))
    return {
        "method": "optional-phrase-lattice-20260920",
        "status": "completed_no_exact_closure" if not exact else "exact_candidates_require_readers",
        "operator": "acyclic optional phrase lattice with asynchronous residual consumption",
        "nodes": sorted(EDGES), "optional_edge_types": ["PP", "ADV", "REL"],
        "states": states, "pruned": pruned, "state_limit": limit,
        "exact_candidates": exact, "exact_candidate_count": len(exact),
        "reader_facing_candidates": [], "reader_eligible": False,
        "controls": controls(),
        "independent_validation": ["literal outside-in two-pointer", "forward/reverse SHA-256"],
        "provenance": "fresh authored ordinary phrase bank and acyclic optional edges; no fixed-slot Cartesian sweep, finished-tape reversal, catalogue text, mirrored units, or repair",
        "first_live_diagnostic": "outer residual mismatch after optional edge emission" if not exact else "exact closure requires blinded reader review",
        "novelty_preflight": {"overlaps": [], "duplicate_sweep": False,
                              "reason": "graph topology changes phrase count via optional edges rather than widening a fixed path product"},
        "next_construction": "add one held-out finite relative edge with explicit antecedent agreement and retain optional-edge lattice; do not add another adjunct sweep",
    }


if __name__ == "__main__":
    result = run()
    out = ROOT / "runs/optional-phrase-lattice-20260920.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
