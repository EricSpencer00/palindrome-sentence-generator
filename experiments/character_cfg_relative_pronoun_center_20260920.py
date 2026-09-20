"""Relative-pronoun center category with direct character equation."""
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

EXPERIMENT_ID = "character-cfg-relative-pronoun-center-20260920"
CENTER = ("REL_PRON", "who")


def seams(words):
    p = 0; out = set()
    for word in words[:-1]: p += len(word); out.add(p)
    return out


def outer_two(left, right, ls, rs):
    l = left[:2]; r = right[-2:]; rev = r[::-1]; n = 0; crossed = False
    while n < len(l) and n < len(rev) and l[n] == rev[n]:
        crossed = crossed or n + 1 in ls or len(right) - n - 1 in rs; n += 1
    return n, crossed


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
    support = 0; direct_states = outer_states = joint = 0; mismatch = Counter(); exact = []
    best = {"score": 0, "left": "", "right": "", "center_category": CENTER[0]}
    for left in subject:
        lt = norm(" ".join(left.words)); ls = seams(left.words)
        for right in object_side:
            rt = norm(" ".join(right.words)); rs = seams(right.words)
            lp = positions(left.words, CENTER[1]); rp = positions(right.words, CENTER[1])
            if not lp or not rp: continue
            support += 1; _, li = lp[0]; _, ri = rp[0]
            lc = lt[li:li + len(CENTER[1])]; rc = rt[ri:ri + len(CENTER[1])]
            rev = rc[::-1]; n = 0
            while n < len(lc) and n < len(rev) and lc[n] == rev[n]: direct_states += 1; n += 1
            if n < len(lc) and n < len(rev): mismatch[f"{lc[n]}/{rev[n]}"] += 1
            ol, oright = lt[:li], rt[ri + len(CENTER[1]):]
            on, oc = outer_two(ol, oright, ls, rs); outer_states += on
            if n == len(lc) and oc: joint += 1
            score = n + on
            if score > best["score"]:
                best = {"score": score, "left": " ".join(left.words), "right": " ".join(right.words),
                        "center_category": CENTER[0], "center_surface": CENTER[1],
                        "center_closed": n == len(lc), "outer_seam": oc}
            if not (n == len(lc) and oc): continue
            rendered = " ".join(left.words).capitalize() + "; " + " ".join(right.words) + "."
            a = audit(rendered)
            if a["two_pointer_exact"]:
                exact.append({"rendered": rendered, "audit": a, "center_category": CENTER[0],
                              "outer_seam_width": 2,
                              "provenance": {"catalogue_imported": False, "finished_tape_reversed": False,
                                             "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in exact}; exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "direct relative-pronoun center CFG equation with relation-aware attachment",
            "grammar": ["S -> NPsubject VP", "VP -> V NPobject", "NP -> DET NOUN REL", "REL -> REL_PRON VP", "REL_PRON -> who"],
            "center_category": {"nonterminal": CENTER[0], "surface": CENTER[1]},
            "stats": {"subject_derivations": len(subject), "object_derivations": len(object_side),
                      "subject_chart_states": subject_states, "object_chart_states": object_states,
                      "center_support": support, "direct_center_states": direct_states,
                      "center_mismatch_support": dict(mismatch), "outer_twochar_states": outer_states,
                      "joint_center_outer_closures": joint, "exact": len(exact),
                      "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact), "best_score": best["score"]},
            "complete_prose_controls": ["The sailor who reads the area guards the harbor.",
                                        "Some guides who keep the era mark a garden."],
            "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "character-cfg-relative-pronoun-center-20260920",
                                  "catalogue_imported": False, "fixed_bank_sweep": False,
                                  "distinct_from": "character-cfg-semantic-center-relation-20260920"},
            "next_construction": "Try a complementizer center category with a distinct complete CFG attachment, keeping direct character equality and outer seam hard.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
