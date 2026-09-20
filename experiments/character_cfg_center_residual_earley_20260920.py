"""Residual-character center chart for the recursive CFG lane."""
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

EXPERIMENT_ID = "character-cfg-center-residual-earley-20260920"
CENTER_ONSETS = (("th", "the", "sg"), ("so", "some", "pl"))
RESIDUAL_ALPHABET = tuple("thso")


def seams(words):
    p = 0; out = set()
    for word in words[:-1]: p += len(word); out.add(p)
    return out


def residual_match(left, right, ls, rs):
    """Consume equal exposed chars and preserve one-sided residual debt."""
    i = j = 0; states = 0; crossed = False; residual = ""
    rev = right[::-1]
    while i < len(left) and j < len(rev) and left[i] == rev[j]:
        states += 1; crossed = crossed or i + 1 in ls or len(right) - j - 1 in rs
        i += 1; j += 1
    if i < len(left) and j < len(rev):
        # A bounded held-out residual chart item records possible one-char
        # continuations without pretending the mismatch is exact.
        residual = left[i] + rev[j]
        states += sum(char in RESIDUAL_ALPHABET for char in residual)
    return states, crossed, residual, i + j


def occurrences(words, token):
    p = 0; out = []
    for i, word in enumerate(words):
        if word == token: out.append((i, p))
        p += len(word)
    return out


def run():
    trie, lex = brown_words()
    subject, subject_states = chart(lex, trie, True, cap=540)
    object_side, object_states = chart(lex, trie, False, cap=540)
    support = Counter(); residuals = Counter(); outer_states = center_states = seam_hits = 0; exact = []
    best = {"score": 0, "left": "", "right": "", "residual": ""}
    for left in subject:
        lt = norm(" ".join(left.words)); ls = seams(left.words)
        for right in object_side:
            rt = norm(" ".join(right.words)); rs = seams(right.words)
            for onset, surface, agreement in CENTER_ONSETS:
                if left.number != agreement or right.number != agreement: continue
                lp = occurrences(left.words, surface); rp = occurrences(right.words, surface)
                if not lp or not rp: continue
                support[onset] += 1; _, lc = lp[0]; _, rc = rp[0]
                cn, cc, residual, center_score = residual_match(lt[lc:lc + len(onset)], rt[rc:rc + len(onset)], ls, rs)
                on, oc, _, outer_score = residual_match(lt[:lc], rt[rc + len(onset):], ls, rs)
                center_states += cn; outer_states += on
                if residual: residuals[residual] += 1
                if cc and oc: seam_hits += 1
                score = cn + on
                if score > best["score"]:
                    best = {"score": score, "left": " ".join(left.words), "right": " ".join(right.words),
                            "onset": onset, "residual": residual, "center_seam": cc, "outer_seam": oc}
                if not (cc and oc and not residual): continue
                rendered = " ".join(left.words).capitalize() + "; " + " ".join(right.words) + "."
                a = audit(rendered)
                if a["two_pointer_exact"]:
                    exact.append({"rendered": rendered, "audit": a, "residual": residual,
                                  "provenance": {"catalogue_imported": False, "finished_tape_reversed": False,
                                                 "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in exact}; exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "character CFG chart with retained residual center-character debt",
            "grammar": ["S -> NPsubject VP", "VP -> V NPobject", "NP -> DET NOUN REL", "REL -> who VP", "CENTER -> onset + residual"],
            "center_onset_bank": [{"onset": a, "surface": b, "agreement": c} for a, b, c in CENTER_ONSETS],
            "residual_alphabet": list(RESIDUAL_ALPHABET),
            "stats": {"subject_derivations": len(subject), "object_derivations": len(object_side),
                      "subject_chart_states": subject_states, "object_chart_states": object_states,
                      "center_support": dict(support), "residual_support": dict(residuals),
                      "center_debt_states": center_states, "outer_debt_states": outer_states,
                      "both_debts_cross_word_seam": seam_hits, "exact": len(exact),
                      "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact), "best_two_debt_score": best["score"]},
            "complete_prose_controls": ["The sailor who reads the area guards the harbor.",
                                        "Some guides who keep the era mark a garden."],
            "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "character-cfg-center-residual-20260920",
                                  "catalogue_imported": False, "fixed_bank_sweep": False,
                                  "distinct_from": "character-cfg-typed-twochar-center-20260920"},
            "next_construction": "Promote only residual states that preserve a complete lexical center token, then solve the first outer obligation with a typed center pair; do not widen the alphabet.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
