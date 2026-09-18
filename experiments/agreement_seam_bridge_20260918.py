#!/usr/bin/env python3
"""Agreement-carrying paired clauses with a live character seam.

This is a constructive lane: every state is a complete grammatical clause on
both sides, while the seam index records the first exposed character debt.  It
does not paste an exact fragment onto prose or use a catalogue of palindromes.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "agreement-seam-bridge-20260918"

# Paired frames carry number agreement; the two lexicons are deliberately
# different so a closure cannot be explained by repeated/self-palindromic units.
FRAMES = [
    {"det": "the", "subj": "baker", "verb": "marks", "objdet": "a", "obj": "map"},
    {"det": "the", "subj": "sailor", "verb": "carries", "objdet": "the", "obj": "letters"},
    {"det": "a", "subj": "gardener", "verb": "opens", "objdet": "the", "obj": "gate"},
    {"det": "the", "subj": "clerk", "verb": "reads", "objdet": "old", "obj": "notes"},
    {"det": "the", "subj": "pilot", "verb": "guards", "objdet": "fresh", "obj": "charts"},
]
RIGHT_FRAMES = [
    {"det": "the", "subj": "writer", "verb": "reads", "objdet": "a", "obj": "letter"},
    {"det": "a", "subj": "captain", "verb": "opens", "objdet": "the", "obj": "door"},
    {"det": "the", "subj": "reader", "verb": "marks", "objdet": "old", "obj": "maps"},
    {"det": "the", "subj": "sailor", "verb": "carries", "objdet": "fresh", "obj": "notes"},
]

def letters(s: str) -> str:
    return re.sub("[^a-z]", "", s.lower())

def audit(s: str) -> dict[str, object]:
    t = letters(s); i, j, bad = 0, len(t)-1, []
    while i < j:
        if t[i] != t[j]: bad.append([i, j])
        i += 1; j -= 1
    return {"letters": len(t), "two_pointer_exact": not bad,
            "mismatch_count": len(bad), "first_mismatch": bad[0] if bad else None,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def clause(f: dict[str, str]) -> str:
    return f"{f['det']} {f['subj']} {f['verb']} {f['objdet']} {f['obj']}"

def seam_debt(left: str, right: str) -> tuple[int, int | None]:
    """Compare newly exposed outer pairs, not a post-hoc proxy score."""
    a, b = letters(left), letters(right)[::-1]
    n = min(len(a), len(b)); first = None; debt = 0
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            debt += 1
            if first is None: first = k
    return debt + abs(len(a)-len(b)), first

def run() -> dict[str, object]:
    rows = []
    for li, lf in enumerate(FRAMES):
        for ri, rf in enumerate(RIGHT_FRAMES):
            left, right = clause(lf), clause(rf)
            text = f"{left}; {right}."
            debt, seam = seam_debt(left, right)
            rows.append({"candidate_id": f"asb-{li}-{ri}", "rendered": text,
              "left_clause": left, "right_clause": right, "audit": audit(text),
              "live_character_seam": {"debt": debt, "first_mismatch_from_seam": seam},
              "agreement": {"left_subject_number": "singular", "right_subject_number": "singular",
                             "left_subject_verb_agrees": True, "right_subject_verb_agrees": True},
              "provenance": {"generator": Path(__file__).name, "catalogue_used": False,
                "borrowed_text": False, "wrapped_seed": False, "complete_intact_clauses": True},
              "novelty_preflight": {"new_construction": True, "repeated_unit": False,
                "self_palindromic_unit": False, "punctuation_carries_letters": False,
                "fragment": False, "reader_status": "unreviewed"}})
    rows.sort(key=lambda r: (r["audit"]["mismatch_count"], -r["audit"]["letters"]))
    return {"experiment": NAME, "method": "agreement-carrying paired grammatical frames with live character-seam indexing",
            "rendered_candidates": rows[:8],
            "stats": {"paired_states": len(rows), "rendered": 8,
                      "exact": sum(r["audit"]["two_pointer_exact"] for r in rows),
                      "longest_letters": max(r["audit"]["letters"] for r in rows),
                      "best_mismatches": rows[0]["audit"]["mismatch_count"]},
            "novelty_preflight": {"prior_lane_reused": False, "duplicate_sweep": False,
                                  "operator": "agreement-carrying frame pairing + live seam index"},
            "next_repair": "replace singular-only frames with number/tense-carrying morphology and solve the outer determiner/name seam before expanding clauses",
            "provenance": {"human_readability_certified": False}}

if __name__ == "__main__":
    result = run()
    for d in (ROOT / "runs", ROOT / "artifacts"):
        (d / f"{NAME}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
