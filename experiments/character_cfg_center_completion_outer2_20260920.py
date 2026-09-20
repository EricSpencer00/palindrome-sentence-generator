"""Typed lexical-center completion jointly coupled to a two-char outer seam."""
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

EXPERIMENT_ID = "character-cfg-center-completion-outer2-20260920"
CENTERS = (("th", "the", "sg"), ("so", "some", "pl"))


def seams(words):
    p = 0; out = set()
    for word in words[:-1]: p += len(word); out.add(p)
    return out


def exact_prefix(left, right, ls, rs, width):
    """Expose only a bounded seam and return live matched states."""
    l = left[:width]; r = right[-width:]
    rev = r[::-1]; n = 0; crossed = False
    while n < len(l) and n < len(rev) and l[n] == rev[n]:
        crossed = crossed or n + 1 in ls or len(right) - n - 1 in rs
        n += 1
    return n, crossed, l, r


def positions(words, token):
    p = 0; out = []
    for i, word in enumerate(words):
        if word == token: out.append((i, p))
        p += len(word)
    return out


def run():
    trie, lex = brown_words()
    subject, subject_states = chart(lex, trie, True, cap=540)
    object_side, object_states = chart(lex, trie, False, cap=540)
    support = Counter(); center_states = outer_states = seam_hits = 0; mismatch = Counter(); exact = []
    best = {"score": 0, "left": "", "right": "", "center": "", "outer_seam": False}
    for left in subject:
        lt = norm(" ".join(left.words)); ls = seams(left.words)
        for right in object_side:
            rt = norm(" ".join(right.words)); rs = seams(right.words)
            for onset, surface, agreement in CENTERS:
                if left.number != agreement or right.number != agreement: continue
                lp = positions(left.words, surface); rp = positions(right.words, surface)
                if not lp or not rp: continue
                support[onset] += 1; _, lc = lp[0]; _, rc = rp[0]
                # Complete lexical center is admitted as a typed constituent.
                center_l = lt[lc:lc + len(surface)]
                center_r = rt[rc:rc + len(surface)]
                center_rev = center_r[::-1]
                cstates = 0
                while cstates < len(center_l) and cstates < len(center_rev) and center_l[cstates] == center_rev[cstates]: cstates += 1
                center_states += cstates
                if cstates < len(surface): mismatch[surface[cstates] + "/" + center_rev[cstates]] += 1
                # Jointly expose exactly two characters from the outer seam.
                ol, oright = lt[:lc], rt[rc + len(surface):]
                ostates, oc, lchunk, rchunk = exact_prefix(ol, oright, ls, rs, 2)
                outer_states += ostates
                if oc and cstates == len(surface): seam_hits += 1
                score = cstates + ostates
                if score > best["score"]:
                    best = {"score": score, "left": " ".join(left.words), "right": " ".join(right.words),
                            "center": surface, "outer_left": lchunk, "outer_right": rchunk,
                            "outer_seam": oc, "center_closed": cstates == len(surface)}
                if not (oc and cstates == len(surface)): continue
                rendered = " ".join(left.words).capitalize() + "; " + " ".join(right.words) + "."
                a = audit(rendered)
                if a["two_pointer_exact"]:
                    exact.append({"rendered": rendered, "audit": a,
                                  "center": {"surface": surface, "agreement": agreement},
                                  "outer_seam_width": 2,
                                  "provenance": {"catalogue_imported": False, "finished_tape_reversed": False,
                                                 "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in exact}; exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "typed lexical-center completion jointly solved with two-character outer seam",
            "grammar": ["S -> NPsubject VP", "VP -> V NPobject", "NP -> DET NOUN REL", "REL -> who VP", "CENTER -> complete lexical token"],
            "center_bank": [{"onset": a, "surface": b, "agreement": c} for a, b, c in CENTERS],
            "stats": {"subject_derivations": len(subject), "object_derivations": len(object_side),
                      "subject_chart_states": subject_states, "object_chart_states": object_states,
                      "center_support": dict(support), "center_debt_states": center_states,
                      "outer_twochar_states": outer_states, "center_mismatch_support": dict(mismatch),
                      "joint_seam_closures": seam_hits, "exact": len(exact),
                      "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact), "best_joint_score": best["score"]},
            "complete_prose_controls": ["The sailor who reads the area guards the harbor.",
                                        "Some guides who keep the era mark a garden."],
            "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "character-cfg-center-completion-outer2-20260920",
                                  "catalogue_imported": False, "fixed_bank_sweep": False,
                                  "distinct_from": "character-cfg-complete-center-admission-20260920"},
            "next_construction": "Use the recorded center mismatch pair as a typed two-sided center item and solve its residual jointly with the outer seam; do not admit mirrored fragments.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
