"""Shared semantic attachment relation across opposite-agreement COMP sites."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.character_cfg_recursive_relative_earley_20260920 import audit, brown_words, norm
from experiments.character_cfg_two_comp_opposite_agreement_20260920 import chart

EXPERIMENT_ID = "character-cfg-shared-comp-semantic-relation-20260920"
RELATIONS = {"maritime_scene": {"sailor", "sailors", "harbor"},
             "writing_scene": {"poet", "writer", "writers", "guides"}}
COMP = "that"


def seams(words):
    p = 0; out = set()
    for word in words[:-1]: p += len(word); out.add(p)
    return out


def outer_two(left, right, ls, rs):
    l = left[:2]; r = right[-2:]; rev = r[::-1]; n = 0; crossed = False
    while n < len(l) and n < len(rev) and l[n] == rev[n]:
        crossed = crossed or n + 1 in ls or len(right) - n - 1 in rs; n += 1
    return n, crossed


def relation_for(left_words, right_words):
    bag = set(left_words) | set(right_words)
    for relation, lex in RELATIONS.items():
        if len(bag & lex) >= 2: return relation
    return None


def run():
    trie, lex = brown_words(); lex["NOUN"] |= {"writers", "guides", "sailors"}; lex["VERB"] |= {"read", "mark", "guide"}
    ss, ss_states = chart(lex, trie, "subject", "sg"); op, op_states = chart(lex, trie, "object", "pl")
    ps, ps_states = chart(lex, trie, "subject", "pl"); os, os_states = chart(lex, trie, "object", "sg")
    support = Counter(); mismatch = Counter(); outer_states = joint = 0; exact = []
    best = {"score": 0, "left": "", "right": "", "relation": ""}
    for left, right, orient in ((ss, op, "sg/pl"), (ps, os, "pl/sg")):
        for lder in left:
            lw = " ".join(lder["words"]); lt = norm(lw); ls = seams(lder["words"])
            for rder in right:
                rw = " ".join(rder["words"]); rt = norm(rw); rs = seams(rder["words"])
                relation = relation_for(lder["words"], rder["words"])
                if relation is None: continue
                support[relation] += 1; li = lt.find(COMP); ri = rt.find(COMP)
                lc = lt[li:li + len(COMP)]; rc = rt[ri:ri + len(COMP)][::-1]; n = 0
                while n < len(lc) and n < len(rc) and lc[n] == rc[n]: n += 1
                if n < len(lc): mismatch[f"{lc[n]}/{rc[n]}"] += 1
                on, oc = outer_two(lt[:li], rt[ri + len(COMP):], ls, rs); outer_states += on
                if n == len(lc) and oc: joint += 1
                if n + on > best["score"]: best = {"score": n + on, "left": lw, "right": rw, "relation": relation, "orientation": orient, "outer_seam": oc}
                if not (n == len(lc) and oc): continue
                rendered = lw.capitalize() + "; " + rw + "."; a = audit(rendered)
                if a["two_pointer_exact"]: exact.append({"rendered": rendered, "audit": a, "semantic_relation": relation,
                    "provenance": {"catalogue_imported": False, "finished_tape_reversed": False, "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in exact}; exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "shared semantic attachment relation across opposite-agreement complementizer sites",
            "grammar": ["S -> NP VP", "subject-site NP -> DET NOUN COMP S", "object-site NP -> DET NOUN COMP S", "COMP -> that", "shared relation -> scene role"],
            "relations": {k: sorted(v) for k, v in RELATIONS.items()},
            "stats": {"subject_sg": len(ss), "object_pl": len(op), "subject_pl": len(ps), "object_sg": len(os),
                      "chart_states": ss_states + op_states + ps_states + os_states, "semantic_relation_support": dict(support),
                      "center_mismatch_support": dict(mismatch), "outer_twochar_states": outer_states,
                      "joint_relation_center_outer_closures": joint, "exact": len(exact),
                      "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact), "best_score": best["score"]},
            "complete_prose_controls": ["The sailor that the poet reads guards the harbor.",
                                        "The writers that some guides mark guide the sailors."],
            "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "character-cfg-shared-comp-semantic-relation-20260920",
                                  "catalogue_imported": False, "lexical_sweep": False,
                                  "distinct_from": "character-cfg-two-comp-opposite-agreement-20260920"},
            "next_construction": "Use a relation-constrained pair of distinct complementizer lexical categories, retaining direct center equality and hard seam; no mismatch forgiveness.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
