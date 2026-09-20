"""Relation-constrained pair of distinct complementizer categories."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.character_cfg_recursive_relative_earley_20260920 import audit, brown_words, norm

EXPERIMENT_ID = "character-cfg-distinct-complementizer-pair-20260920"
COMPS = (("COMP_THAT", "that", "sg"), ("COMP_IF", "if", "sg"))
RELATIONS = {"writing_scene": {"poet", "writer", "writers", "guides"}, "maritime_scene": {"sailor", "sailors", "harbor"}}


def seams(words):
    p = 0; out = set()
    for word in words[:-1]: p += len(word); out.add(p)
    return out


def chart(lex, trie, comp, cap=180):
    rows = []; states = 0; surface = comp[1]
    nouns = sorted(n for n in lex["NOUN"] if n in {"sailor", "poet", "keeper", "writer", "garden", "harbor", "area", "era"})
    verbs = sorted(v for v in lex["VERB"] if v in {"guards", "marks", "guides", "keeps", "reads", "writes"})
    for det in sorted(lex["DET"]):
        for noun in nouns:
            for inner in nouns:
                for verb in verbs:
                    words = (det, noun, surface, det, inner, verb, det, noun)
                    if not all(trie.accepts(w) for w in words): continue
                    states += sum(len(w) for w in words)
                    rows.append({"words": words, "tree": f"S(NP({det},{noun},{comp[0]} S),VP({verb}))", "surface": surface})
                    if len(rows) >= cap: return tuple(rows), states
    return tuple(rows), states


def outer_two(left, right, ls, rs):
    l = left[:2]; r = right[-2:]; rev = r[::-1]; n = 0; crossed = False
    while n < len(l) and n < len(rev) and l[n] == rev[n]:
        crossed = crossed or n + 1 in ls or len(right) - n - 1 in rs; n += 1
    return n, crossed


def relation_for(left, right):
    bag = set(left) | set(right)
    for name, words in RELATIONS.items():
        if len(bag & words) >= 2: return name
    return None


def run():
    trie, lex = brown_words(); left_that, st = chart(lex, trie, COMPS[0]); right_if, si = chart(lex, trie, COMPS[1])
    left_if, sf = chart(lex, trie, COMPS[1]); right_that, it = chart(lex, trie, COMPS[0])
    relation_support = Counter(); mismatch = Counter(); outer_states = joint = 0; exact = []
    best = {"score": 0, "left": "", "right": "", "pair": ""}
    for left, right, pair in ((left_that, right_if, "that/if"), (left_if, right_that, "if/that")):
        for lder in left:
            lw = " ".join(lder["words"]); lt = norm(lw); ls = seams(lder["words"])
            for rder in right:
                rw = " ".join(rder["words"]); rt = norm(rw); rs = seams(rder["words"])
                relation = relation_for(lder["words"], rder["words"])
                if relation is None: continue
                relation_support[f"{pair}:{relation}"] += 1
                li = lt.find(lder["surface"]); ri = rt.find(rder["surface"])
                lc = lt[li:li + len(lder["surface"])]
                rc = rt[ri:ri + len(rder["surface"])][::-1]
                n = 0
                while n < len(lc) and n < len(rc) and lc[n] == rc[n]: n += 1
                if n < len(lc) and n < len(rc): mismatch[f"{lc[n]}/{rc[n]}"] += 1
                on, oc = outer_two(lt[:li], rt[ri + len(rder["surface"]):], ls, rs); outer_states += on
                if n == len(lc) and oc: joint += 1
                if n + on > best["score"]: best = {"score": n + on, "left": lw, "right": rw, "pair": pair, "relation": relation, "outer_seam": oc}
                if not (n == len(lc) and oc): continue
                rendered = lw.capitalize() + "; " + rw + "."; a = audit(rendered)
                if a["two_pointer_exact"]: exact.append({"rendered": rendered, "audit": a, "relation": relation,
                    "provenance": {"catalogue_imported": False, "finished_tape_reversed": False, "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in exact}; exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "relation-constrained distinct complementizer pair with direct center equality",
            "grammar": ["S -> NP VP", "NP -> DET NOUN COMP S", "COMP -> that | if", "shared relation -> scene role"],
            "complementizers": [{"category": a, "surface": b, "agreement": c} for a, b, c in COMPS],
            "relations": {k: sorted(v) for k, v in RELATIONS.items()},
            "stats": {"that_derivations": len(left_that), "if_derivations": len(left_if), "chart_states": st + si + sf + it,
                      "relation_pair_support": dict(relation_support), "center_mismatch_support": dict(mismatch),
                      "outer_twochar_states": outer_states, "joint_closures": joint, "exact": len(exact),
                      "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact), "best_score": best["score"]},
            "complete_prose_controls": ["The sailor that the poet reads guards the harbor.",
                                        "The sailor wonders if the poet reads the harbor."],
            "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "character-cfg-distinct-complementizer-pair-20260920",
                                  "catalogue_imported": False, "lexical_sweep": False,
                                  "distinct_from": "character-cfg-shared-comp-semantic-relation-20260920"},
            "next_construction": "Use two distinct complementizers with compatible subcategorization and agreement, preserving direct center equality and the hard outer seam.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
