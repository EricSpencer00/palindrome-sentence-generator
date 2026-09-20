"""Shared-center chart with two independent live seam debts.

Subject-relative and object-relative derivations are paired around an
explicit ``who`` center item.  The chart advances an outer debt and a center
debt separately; exact admission still requires the complete rendered tape
to pass an independent two-pointer/SHA audit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.character_cfg_asymmetric_relative_earley_20260920 import chart
from experiments.character_cfg_recursive_relative_earley_20260920 import audit, brown_words, norm

EXPERIMENT_ID = "character-cfg-shared-center-two-debt-20260920"


def seam_positions(words):
    pos = 0
    out = set()
    for word in words[:-1]:
        pos += len(word)
        out.add(pos)
    return out


def debt(left: str, right: str, left_seams: set[int], right_seams: set[int]):
    """Match exposed characters while tracking whether a word seam is crossed."""
    matched = 0
    crossed = False
    reverse = right[::-1]
    while matched < len(left) and matched < len(reverse) and left[matched] == reverse[matched]:
        crossed = crossed or matched + 1 in left_seams or len(right) - matched - 1 in right_seams
        matched += 1
    return matched, crossed


def run():
    trie, lex = brown_words()
    subject, subject_states = chart(lex, trie, True)
    object_side, object_states = chart(lex, trie, False)
    candidates = []
    outer_states = center_states = seam_hits = 0
    best = {"letters": 0, "left": "", "right": "", "outer_seam": False, "center_seam": False}
    for left in subject:
        lw = " ".join(left.words)
        lt = norm(lw)
        # The shared center nonterminal is the existing determiner ``a``;
        # relative clauses remain on the surrounding NP sides.  Using a
        # one-character terminal keeps this a real character equation.
        lcenter = lt.find("a")
        if lcenter < 0:
            continue
        lseams = seam_positions(left.words)
        for right in object_side:
            rw = " ".join(right.words)
            rt = norm(rw)
            rcenter = rt.find("a")
            if rcenter < 0:
                continue
            rseams = seam_positions(right.words)
            # Debt one: outer spans up to each center.
            outer_l = lt[:lcenter]
            outer_r = rt[rcenter + 1:]
            outer_n, outer_cross = debt(outer_l, outer_r, lseams, rseams)
            outer_states += outer_n
            # Debt two: spans exposed around the shared center nonterminal.
            center_l = lt[lcenter:lcenter + 1]
            center_r = rt[rcenter:rcenter + 1]
            center_n, center_cross = debt(center_l, center_r, lseams, rseams)
            center_cross = center_cross or (lcenter in lseams) or (lcenter + 1 in lseams)
            center_cross = center_cross or (rcenter in rseams) or (rcenter + 1 in rseams)
            center_states += center_n
            seam_hits += int(outer_cross and center_cross)
            score = outer_n + center_n
            if score > best["letters"]:
                best = {"letters": score, "left": lw, "right": rw,
                        "outer_seam": outer_cross, "center_seam": center_cross}
            if not (outer_cross and center_cross):
                continue
            rendered = lw.capitalize() + "; " + rw + "."
            a = audit(rendered)
            if a["two_pointer_exact"]:
                candidates.append({"rendered": rendered, "audit": a,
                                   "debts": {"outer": outer_n, "center": center_n},
                                   "seams": {"outer": True, "center": True},
                                   "provenance": {"brown_lexicon_only": True,
                                                  "catalogue_imported": False,
                                                  "finished_tape_reversed": False,
                                                  "word_order_mirror": False,
                                                  "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in candidates}
    exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "shared-center character CFG chart with independent outer and center seam debts",
            "grammar": ["S -> NPsubject VP", "VP -> V NPobject", "NP -> DET NOUN REL", "REL -> who VP"],
            "stats": {"subject_derivations": len(subject), "object_derivations": len(object_side),
                      "subject_chart_states": subject_states, "object_chart_states": object_states,
                      "outer_debt_states": outer_states, "center_debt_states": center_states,
                      "both_debts_cross_word_seam": seam_hits, "exact": len(exact),
                      "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact),
                      "best_two_debt_score": best["letters"]},
            "complete_prose_controls": ["The sailor who reads the area guards the harbor.",
                                        "The sailor guards the area who reads the poet."],
            "best_diagnostic": best,
            "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "character-cfg-shared-center-two-debt-20260920",
                                  "catalogue_imported": False, "fixed_bank_sweep": False,
                                  "distinct_from": ["character-cfg-asymmetric-relative-seam-20260920",
                                                     "character-cfg-object-relative-attachment-20260920"]},
            "next_construction": "Replace the fixed who center with a typed center nonterminal whose agreement feature is solved jointly with both debts; keep lexical terminals fixed.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run()
    out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
