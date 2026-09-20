"""Exact-by-construction typed two-event/relative-clause grammar.

This is a new grammar and held-out phrase bank, not a sweep of the earlier
seed.  Both sides choose role-labelled phrases independently while carrying
comparison-oriented residual buffers.  No completed tape is reversed.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Held out from the earlier seed/control bank.  Chunks are ordinary prose,
# grouped by syntactic/semantic role; none is constructed as a mirror mate.
BANK = {
    "NP": ("the baker", "a singer", "the young captain", "a patient gardener",
           "the small fox", "a thoughtful teacher", "the winter poet"),
    "VP": ("greets the crowd", "keeps warm bread", "writes a letter",
           "guides the child", "opens the window", "carries fresh water",
           "hears the bell"),
    "PP": ("beside the harbor", "under the old tree", "near a quiet house",
           "with a red ribbon", "by the northern road"),
    "REL": ("who sings softly", "that remembers home", "who watches the rain",
            "that carries hope", "who walks at dusk"),
}


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    return {"letters": len(tape), "exact": bool(tape) and tape == tape[::-1],
            "sha256": hashlib.sha256(tape.encode()).hexdigest()}


def consume(left: str, right: str):
    """Consume equal outer-to-inner prefixes of residual buffers."""
    n = min(len(left), len(right))
    if left[:n] != right[:n]:
        return None
    return left[n:], right[n:]


def paths(max_units: int = 6):
    base = [("NP", "VP", "NP", "VP")]
    for n in range(1, max_units - 3):
        # Relative clauses and adjuncts are inserted only after both events.
        base.append(("NP", "VP", "NP", "VP") + ("REL",) * n)
        base.append(("NP", "VP", "NP", "VP") + ("PP",) * n)
    return tuple(base)


def run(limit: int = 180_000, max_units: int = 6) -> dict:
    grammar = paths(max_units)
    states = pruned = 0
    exact = []
    seen = set()
    for left_roles in grammar:
        for right_rendered_roles in grammar:
            right_roles = tuple(reversed(right_rendered_roles))
            stack = [(0, 0, "", "", "", "", ())]
            while stack and states < limit:
                li, ri, left, right, lbuf, rbuf, prov = stack.pop()
                states += 1
                if li == len(left_roles) and ri == len(right_roles):
                    if lbuf or rbuf:
                        pruned += 1
                        continue
                    rendered = (left + " " + right).strip()
                    info = audit(rendered)
                    if info["exact"] and info["letters"] > 38 and rendered not in seen:
                        seen.add(rendered)
                        exact.append({"rendered": rendered, "audit": info,
                                      "provenance": {"left_roles": left_roles,
                                                     "right_roles": right_roles,
                                                     "phrase_units": len(prov),
                                                     "bank": "held-out-authored-semantic",
                                                     "corpus_sentence_replay": False,
                                                     "mirrored_token_units": False}})
                    continue
                if li < len(left_roles):
                    role = left_roles[li]
                    for phrase in reversed(BANK[role]):
                        residual = consume(lbuf + letters(phrase), rbuf)
                        if residual is None:
                            pruned += 1
                            continue
                        stack.append((li + 1, ri, (left + " " if left else "") + phrase,
                                      right, residual[0], residual[1],
                                      prov + (("L", role, phrase),)))
                if ri < len(right_roles):
                    role = right_roles[ri]
                    for phrase in reversed(BANK[role]):
                        residual = consume(lbuf, rbuf + letters(phrase)[::-1])
                        if residual is None:
                            pruned += 1
                            continue
                        stack.append((li, ri + 1, left,
                                      phrase + (" " + right if right else ""),
                                      residual[0], residual[1],
                                      prov + (("R", role, phrase),)))
            if states >= limit:
                break
        if states >= limit:
            break
    return {"method": "typed-two-event-relative-grammar-20260920",
            "grammar_paths": len(grammar), "bank_sizes": {k: len(v) for k, v in BANK.items()},
            "states": states, "pruned": pruned, "exact_candidates": exact,
            "candidate_count": len(exact),
            "status": "reader gate required" if exact else "construction frontier empty"}


if __name__ == "__main__":
    result = run()
    out = ROOT / "runs/typed-two-event-relative-grammar-20260920.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
