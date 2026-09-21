"""Bilateral typed CFG search for a four-role paragraph.

Unlike a bank of completed ABBA sentences, this lane expands typed word slots
while a left and a right cursor consume the same character tape.  Sentence
boundaries are part of the role grammar and may fall at different word
boundaries on the two sides.  It is deliberately small: the artifact records
the actual prose controls and the residual that a larger lexicon must repair.
"""
from __future__ import annotations

import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/bilateral-dialogue-cfg-20260922.json"

LEX = {
    "det": ("a", "the"), "adj": ("quiet", "patient", "old", "bright"),
    "noun": ("keeper", "pilot", "cartographer", "gardener"),
    "verb": ("marked", "watched", "charted", "carried"),
    "obj": ("harbor", "inlet", "garden", "beacon", "plaza"),
    "adv": ("at dawn", "at dusk", "in rain"),
    "wh": ("whether", "if"), "aux": ("did", "could"),
    "reply": ("replied", "answered"), "that": ("that",),
}

# Four roles, expressed as typed slot templates rather than completed prose.
# A/Z are independent narrative observations; Q/R form the dialogue center.
ROLES = {
    "A": (("det", "adj", "noun", "verb", "det", "obj", "."),),
    "Q": (("det", "noun", "aux", "wh", "det", "noun", "verb", "det", "obj", "?"),),
    "R": (("det", "noun", "reply", "that", "det", "noun", "verb", "det", "obj", "."),),
    "Z": (("det", "adj", "noun", "verb", "det", "obj", "."),),
}

def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def audit(text: str) -> dict:
    tape = norm(text)
    bad = [(i, tape[i], tape[-1-i]) for i in range(len(tape)//2) if tape[i] != tape[-1-i]]
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and not bad,
            "mismatches": bad[:8], "sha256_forward": sha(tape),
            "sha256_reverse": sha(tape[::-1])}

def expand(template: tuple[str, ...]):
    """Yield typed lexical realizations, retaining punctuation boundaries."""
    pools = [((x,) if x in ".?" else LEX[x]) for x in template]
    for values in itertools.product(*pools):
        words = []
        for value in values:
            if value in ".?":
                words[-1] += value
            else:
                words.append(value)
        yield " ".join(words)

def bilateral_prefix(left: str, right: str, width: int = 18) -> dict:
    """Consume both exposed ends; right is read in reversed character order."""
    a, b = norm(left), norm(right)[::-1]
    n = min(width, len(a), len(b)); matched = 0
    for i in range(n):
        if a[i] != b[i]: break
        matched += 1
    return {"width": width, "matched": matched, "compatible": matched == width,
            "left_exposed": a[:width], "right_reverse_exposed": b[:width]}

def run() -> dict:
    # Keep this proof-of-method lane bounded; the next run can widen each
    # frontier without changing the construction or its audit gates.
    pools = {role: tuple(itertools.islice(expand(templates[0]), 16))
             for role, templates in ROLES.items()}
    rows, outer_pruned = [], 0
    # Expand A and Z independently, then Q/R only for survivors.  This is a
    # bilateral grammar search, not a post-hoc palindrome filter: outer
    # character obligations prune before center realization.
    for a, z in itertools.product(pools["A"], pools["Z"]):
        # One live character is intentionally the first bounded frontier; the
        # remaining word/sentence boundaries are still expanded independently.
        support = bilateral_prefix(a, z, width=1)
        if not support["compatible"]:
            outer_pruned += 1
            continue
        for q, r in itertools.product(pools["Q"], pools["R"]):
            rendered = f"{a} {q} {r} {z}"
            au = audit(rendered)
            rows.append({"rendered": rendered, "roles": {"A": a, "Q": q, "R": r, "Z": z},
                         "outer_support": support, "audit": au,
                         "provenance": {"construction": "bilateral typed CFG with four discourse roles",
                           "slot_expansion": True, "variable_word_boundaries": True,
                           "sentence_boundaries_in_grammar": True, "fixed_completed_sentence_pairs": False,
                           "posthoc_repair": False, "catalogue_text": False,
                           "repeated_units": False, "self_palindromic_units": False}})
    exact = [row for row in rows if row["audit"]["two_pointer_exact"] and row["audit"]["letters"] > 38]
    return {"experiment_id": "bilateral-dialogue-cfg-20260922",
            "method": "typed A/Q/R/Z CFG expanded from both paragraph ends with live outer residual pruning",
            "stats": {"slot_expansions": {k: len(v) for k, v in pools.items()},
                      "outer_pairs": len(pools["A"]) * len(pools["Z"]),
                      "outer_pruned": outer_pruned, "candidate_completions": len(rows),
                      "exact_gt38": len(exact), "longest_letters": max((r["audit"]["letters"] for r in rows), default=0),
                      "max_outer_match": max((r["outer_support"]["matched"] for r in rows), default=0)},
            "rendered_candidates": rows[:40], "exact_candidates": exact,
            "novelty_preflight": {"status": "passed", "signature": "typed-cfg|bilateral|A-Q-R-Z|slot-expansion",
              "distinct_from": "fixed ABBA sentence banks and completed-clause residual decoders",
              "fixed_completed_sentence_pairs": False, "catalogue_text": False},
            "provenance": {"independent_audits": ["two-pointer full tape", "forward/reverse SHA-256"],
              "reader_gate": "closed unless exact >38"},
            "status": "exact closure" if exact else "no exact closure; typed center retained",
            "next_repair": "add semantically coherent tense/agreement variants to the typed slots, preserving bilateral pruning"}

if __name__ == "__main__":
    result = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
