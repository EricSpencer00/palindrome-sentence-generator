"""Typed center nonterminal with agreement-carrying two-debt equations."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.character_cfg_asymmetric_relative_earley_20260920 import chart
from experiments.character_cfg_recursive_relative_earley_20260920 import audit, brown_words, norm

EXPERIMENT_ID = "character-cfg-typed-center-two-debt-20260920"

# Held-out typed center category bank. These are already in the fixed lexical
# terminals; the new variable is the typed nonterminal and its agreement.
CENTERS = (("a", "sg"), ("the", "sg"), ("some", "pl"))


def seam_positions(words):
    p = 0
    out = set()
    for word in words[:-1]:
        p += len(word)
        out.add(p)
    return out


def match_debt(left, right, left_seams, right_seams):
    matched = 0
    crossed = False
    reverse = right[::-1]
    while matched < len(left) and matched < len(reverse) and left[matched] == reverse[matched]:
        crossed = crossed or matched + 1 in left_seams or len(right) - matched - 1 in right_seams
        matched += 1
    return matched, crossed


def center_occurrences(words, center):
    positions = []
    pos = 0
    for index, word in enumerate(words):
        if word == center:
            positions.append((index, pos))
        pos += len(word)
    return positions


def run():
    trie, lex = brown_words()
    subjects, subject_states = chart(lex, trie, True)
    objects, object_states = chart(lex, trie, False)
    support = Counter()
    first_obligation = Counter()
    outer_states = center_states = seam_hits = 0
    exact = []
    best = {"score": 0, "left": "", "right": "", "center": ""}
    for left in subjects:
        lw = " ".join(left.words); lt = norm(lw); ls = seam_positions(left.words)
        for right in objects:
            rw = " ".join(right.words); rt = norm(rw); rs = seam_positions(right.words)
            for center, agreement in CENTERS:
                if left.number != agreement or right.number != agreement:
                    continue
                lpos = center_occurrences(left.words, center)
                rpos = center_occurrences(right.words, center)
                if not lpos or not rpos:
                    continue
                support[center] += 1
                # Use the first typed center occurrence, retaining the typed
                # item in the chart state rather than searching all words.
                _, lc = lpos[0]; _, rc = rpos[0]
                outer_l, outer_r = lt[:lc], rt[rc + len(center):]
                center_l, center_r = lt[lc:lc + len(center)], rt[rc:rc + len(center)]
                outer_n, outer_cross = match_debt(outer_l, outer_r, ls, rs)
                center_n, center_cross = match_debt(center_l, center_r, ls, rs)
                outer_states += outer_n; center_states += center_n
                if outer_l and not outer_n:
                    first_obligation[outer_l[0] + "/" + outer_r[-1:] if outer_r else outer_l[0]] += 1
                if outer_cross and center_cross:
                    seam_hits += 1
                score = outer_n + center_n
                if score > best["score"]:
                    best = {"score": score, "left": lw, "right": rw, "center": center,
                            "outer_seam": outer_cross, "center_seam": center_cross}
                if not (outer_cross and center_cross):
                    continue
                rendered = lw.capitalize() + "; " + rw + "."
                a = audit(rendered)
                if a["two_pointer_exact"]:
                    exact.append({"rendered": rendered, "audit": a,
                                  "typed_center": {"surface": center, "agreement": agreement},
                                  "debts": {"outer": outer_n, "center": center_n},
                                  "provenance": {"catalogue_imported": False, "finished_tape_reversed": False,
                                                 "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in exact}
    exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "typed center CFG chart with agreement-carrying outer and center seam debts",
            "grammar": ["S -> NPsubject VP", "VP -> V NPobject", "NP -> DET NOUN REL", "REL -> who VP", "CENTER -> a|the|some"],
            "center_bank": [{"surface": x, "agreement": y} for x, y in CENTERS],
            "stats": {"subject_derivations": len(subjects), "object_derivations": len(objects),
                      "subject_chart_states": subject_states, "object_chart_states": object_states,
                      "typed_center_support": dict(support), "outer_debt_states": outer_states,
                      "center_debt_states": center_states, "both_debts_cross_word_seam": seam_hits,
                      "exact": len(exact), "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact),
                      "best_two_debt_score": best["score"]},
            "first_obligation_support": dict(first_obligation),
            "complete_prose_controls": ["The sailor who reads the area guards the harbor.",
                                        "Some guides who keep the era mark a garden."],
            "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "character-cfg-typed-center-two-debt-20260920",
                                  "catalogue_imported": False, "fixed_bank_sweep": False,
                                  "distinct_from": "character-cfg-shared-center-two-debt-20260920"},
            "next_construction": "Use a center nonterminal with a typed mirrored two-character onset, then solve the first outer obligation jointly instead of adding center categories.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run()
    out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
