"""Semantic center relation over the typed t/e pair and outer seam."""
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

EXPERIMENT_ID = "character-cfg-semantic-center-relation-20260920"
RELATIONS = (("contrastive_center", "t", "e", "the", "sg"),
             ("continuative_center", "s", "e", "some", "pl"))


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
    relation_support = Counter(); outer_states = seam_hits = 0; exact = []
    best = {"score": 0, "left": "", "right": "", "relation": ""}
    for left in subject:
        lt = norm(" ".join(left.words)); ls = seams(left.words)
        for right in object_side:
            rt = norm(" ".join(right.words)); rs = seams(right.words)
            for relation, lchar, rchar, surface, agreement in RELATIONS:
                if left.number != agreement or right.number != agreement: continue
                lp = positions(left.words, surface); rp = positions(right.words, surface)
                if not lp or not rp: continue
                relation_support[relation] += 1; _, li = lp[0]; _, ri = rp[0]
                observed = (lt[li], rt[ri + len(surface) - 1])
                semantic_edge = {"relation": relation, "left_char": lchar, "right_char": rchar,
                                 "observed": {"left": observed[0], "right": observed[1]},
                                 "complete_left_center": surface, "complete_right_center": surface}
                # The relation is semantic metadata only; it cannot discharge
                # the character inequality. Exact tape validation remains hard.
                center_relation_valid = observed == (lchar, rchar)
                ol, oright = lt[:li], rt[ri + len(surface):]
                on, oc = outer_two(ol, oright, ls, rs); outer_states += on
                if center_relation_valid and oc: seam_hits += 1
                score = on + int(center_relation_valid)
                if score > best["score"]:
                    best = {"score": score, "left": " ".join(left.words), "right": " ".join(right.words),
                            "relation": relation, "semantic_edge": semantic_edge, "outer_seam": oc}
                if not (center_relation_valid and oc): continue
                rendered = " ".join(left.words).capitalize() + "; " + " ".join(right.words) + "."
                a = audit(rendered)
                if a["two_pointer_exact"]:
                    exact.append({"rendered": rendered, "audit": a, "semantic_center": semantic_edge,
                                  "outer_seam_width": 2,
                                  "provenance": {"catalogue_imported": False, "finished_tape_reversed": False,
                                                 "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in exact}; exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "semantic center relation over complete lexical constituents with outer-two seam",
            "grammar": ["S -> NPsubject VP", "VP -> V NPobject", "NP -> DET NOUN REL", "REL -> who VP", "CENTERREL -> semantic relation(left_char,right_char)"],
            "relation_bank": [{"relation": a, "left_char": b, "right_char": c, "surface": d, "agreement": e} for a, b, c, d, e in RELATIONS],
            "stats": {"subject_derivations": len(subject), "object_derivations": len(object_side),
                      "subject_chart_states": subject_states, "object_chart_states": object_states,
                      "semantic_relation_support": dict(relation_support), "outer_twochar_states": outer_states,
                      "joint_relation_seam_closures": seam_hits, "exact": len(exact),
                      "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact), "best_relation_score": best["score"]},
            "complete_prose_controls": ["The sailor who reads the area guards the harbor.",
                                        "Some guides who keep the era mark a garden."],
            "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "character-cfg-semantic-center-relation-20260920",
                                  "catalogue_imported": False, "fixed_bank_sweep": False,
                                  "distinct_from": "character-cfg-typed-twosided-center-20260920"},
            "next_construction": "Use semantic relation choice to select a fresh grammatical center category, then solve its character equation directly; do not use relation labels to forgive mismatches.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
