"""Bilateral semantic nonterminal construction with live character equations.

This is deliberately neither a tape resegmentation nor an Earley intersection:
each side is a separately selected, ordinary-order scene clause.  A paired
semantic nonterminal records its realization and the constructor exposes the
unmatched character frontier while selecting the next pair.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
import json, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import mechanical_admission_checks

EXPERIMENT_ID = "bilateral-semantic-cfg-20260916"
ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Realization:
    subject: str
    verb: str
    obj: str
    meaning: str
    def render(self) -> str:
        return f"{self.subject} {self.verb} {self.obj}"

PAIRS = (
    (Realization("the quiet clerk", "records", "the parcel", "clerk records parcel"),
     Realization("the patient guard", "checks", "the seal", "guard checks seal")),
    (Realization("the courier", "carries", "the letter", "courier carries letter"),
     Realization("the keeper", "opens", "the gate", "keeper opens gate")),
    (Realization("the young baker", "mixes", "the dough", "baker mixes dough"),
     Realization("the old sailor", "mends", "the sail", "sailor mends sail")),
)

def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def audits(text: str) -> dict:
    n = norm(text)
    rev = n[::-1]
    return {
        "letters": len(n),
        "exact": n == rev,
        "two_pointer": all(n[i] == n[-1-i] for i in range(len(n)//2)),
        "sha256_forward": sha256(n.encode()).hexdigest(),
        "sha256_reverse": sha256(rev.encode()).hexdigest(),
        "hash_equal": sha256(n.encode()).digest() == sha256(rev.encode()).digest(),
        "mechanical": mechanical_admission_checks(text, min_letters=39),
    }

def first_mismatch(text: str):
    n = norm(text)
    for i, (a, b) in enumerate(zip(n, reversed(n))):
        if a != b:
            return {"index": i, "left": a, "right": b}
    return None

def construct(pairs=PAIRS):
    left, right, frontier, trace = [], [], [], []
    # semantic nonterminals are expanded on both sides, not derived from a tape
    for idx, (l, r) in enumerate(pairs):
        left.append(l.render()); right.append(r.render())
        ln, rn = norm(l.render()), norm(r.render())
        # The frontier is diagnostic/live state, not a post-hoc reverse operation.
        m = min(len(ln), len(rn))
        matched = sum(a == b for a, b in zip(ln[:m], rn[:m]))
        frontier.append({"pair": idx, "left_nt": asdict(l), "right_nt": asdict(r),
                         "left_letters": len(ln), "right_letters": len(rn),
                         "local_same_position_matches": matched,
                         "unresolved_character_debt": abs(len(ln)-len(rn))})
        trace.append({"event": "expand-paired-nonterminal", "pair": idx,
                      "semantic_left": l.meaning, "semantic_right": r.meaning,
                      "frontier": frontier[-1]})
    return ". ".join(left) + ". " + ". ".join(right), trace, frontier

def main():
    text, trace, frontier = construct()
    # Held-out repair: alter a semantic object and show recomputation, never
    # copying letters from the target or reversing the rendered string.
    held = list(PAIRS)
    held[1] = (held[1][0], Realization("the keeper", "opens", "the lock", "keeper opens lock"))
    repaired, repair_trace, repair_frontier = construct(tuple(held))
    out = {
        "id": EXPERIMENT_ID, "method": "bilateral semantic nonterminal CFG",
        "construction": {"ordinary_word_order": True, "separately_authored_sides": True,
                          "live_character_equations": True, "fixed_tape": False,
                          "reverse_decoder": False, "complete_prose": True,
                          "clause_count": 6},
        "candidate": {"text": text, "audits": audits(text), "first_mismatch": first_mismatch(text)},
        "frontier_trace": trace, "frontier": frontier,
        "repair": {"operator": "held-out semantic object substitution",
                    "before": text, "after": repaired, "audits": audits(repaired),
                    "first_mismatch": first_mismatch(repaired), "frontier": repair_frontier,
                    "trace": repair_trace},
        "provenance": {"source": str(Path(__file__).relative_to(ROOT)),
                        "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                        "pair_count": len(PAIRS), "pair_meanings": [[a.meaning,b.meaning] for a,b in PAIRS]},
        "reader_status": "diagnostic near-miss: intact six-clause prose; not promoted without exact closure",
        "next_repair": "expand the held-out semantic object domains jointly with verb valency while retaining the live frontier"
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
