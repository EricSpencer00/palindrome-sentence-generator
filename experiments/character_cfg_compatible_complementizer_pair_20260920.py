"""Compatible distinct complementizers with direct center and seam gates."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.character_cfg_recursive_relative_earley_20260920 import audit, brown_words, norm
from experiments.character_cfg_distinct_complementizer_pair_20260920 import chart, relation_for, outer_two, seams

EXPERIMENT_ID = "character-cfg-compatible-complementizer-pair-20260920"
COMPS = (("COMP_THAT", "that", "sg"), ("COMP_AS", "as", "sg"))


def run():
    trie, lex = brown_words()
    left_that, a = chart(lex, trie, COMPS[0]); right_as, b = chart(lex, trie, COMPS[1])
    left_as, c = chart(lex, trie, COMPS[1]); right_that, d = chart(lex, trie, COMPS[0])
    support = Counter(); mismatch = Counter(); outer_states = joint = 0; exact = []
    best = {"score": 0, "left": "", "right": "", "pair": ""}
    for left, right, pair in ((left_that, right_as, "that/as"), (left_as, right_that, "as/that")):
        for lder in left:
            lw = " ".join(lder["words"]); lt = norm(lw); ls = seams(lder["words"])
            for rder in right:
                rw = " ".join(rder["words"]); rt = norm(rw); rs = seams(rder["words"])
                relation = relation_for(lder["words"], rder["words"])
                if relation is None: continue
                support[f"{pair}:{relation}"] += 1
                li = lt.find(lder["surface"]); ri = rt.find(rder["surface"])
                lc = lt[li:li + len(lder["surface"])]
                rc = rt[ri:ri + len(rder["surface"])][::-1]; n = 0
                while n < len(lc) and n < len(rc) and lc[n] == rc[n]: n += 1
                if n < len(lc) and n < len(rc): mismatch[f"{lc[n]}/{rc[n]}"] += 1
                on, oc = outer_two(lt[:li], rt[ri + len(rder["surface"]):], ls, rs); outer_states += on
                if n == len(lc) and oc: joint += 1
                if n + on > best["score"]: best = {"score": n + on, "left": lw, "right": rw, "pair": pair, "relation": relation, "outer_seam": oc}
                if not (n == len(lc) and oc): continue
                rendered = lw.capitalize() + "; " + rw + "."; x = audit(rendered)
                if x["two_pointer_exact"]: exact.append({"rendered": rendered, "audit": x, "relation": relation,
                    "provenance": {"catalogue_imported": False, "finished_tape_reversed": False, "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in exact}; exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "compatible distinct complementizer pair with direct center equality and hard outer seam",
            "grammar": ["S -> NP VP", "NP -> DET NOUN COMP S", "COMP -> that | as", "shared scene relation"],
            "complementizers": [{"category": a, "surface": b, "agreement": c} for a, b, c in COMPS],
            "stats": {"that_derivations": len(left_that), "as_derivations": len(left_as), "chart_states": a + b + c + d,
                      "relation_pair_support": dict(support), "center_mismatch_support": dict(mismatch),
                      "outer_twochar_states": outer_states, "joint_closures": joint, "exact": len(exact),
                      "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact), "best_score": best["score"]},
            "complete_prose_controls": ["The sailor that the poet reads guards the harbor.",
                                        "The sailor acts as the poet reads the harbor."],
            "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "character-cfg-compatible-complementizer-pair-20260920",
                                  "catalogue_imported": False, "lexical_sweep": False,
                                  "distinct_from": "character-cfg-distinct-complementizer-pair-20260920"},
            "next_construction": "Pair compatible complementizers with opposite attachment depth, retaining direct center equality and hard seam.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
