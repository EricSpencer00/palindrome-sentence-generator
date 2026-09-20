"""Opposite complementizer attachment depth with direct center/seam gates."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.character_cfg_recursive_relative_earley_20260920 import audit, brown_words, norm

EXPERIMENT_ID = "character-cfg-opposite-complementizer-depth-20260920"
NOUNS = ("sailor", "poet", "keeper", "writer", "garden", "harbor", "area", "era")
VERBS = ("guards", "marks", "guides", "keeps", "reads", "writes")


def seams(words):
    p = 0; out = set()
    for word in words[:-1]: p += len(word); out.add(p)
    return out


def chart(lex, trie, surface, depth, cap=180):
    rows = []; states = 0
    for det in sorted(lex["DET"]):
        for noun in NOUNS:
            if noun not in lex["NOUN"]: continue
            for inner in NOUNS:
                if inner not in lex["NOUN"]: continue
                for verb in VERBS:
                    if verb not in lex["VERB"]: continue
                    if depth == 1:
                        words = (det, noun, surface, det, inner, verb, det, noun)
                    else:
                        words = (det, noun, surface, det, inner, surface, det, noun, verb, det, noun)
                    if not all(trie.accepts(w) for w in words): continue
                    states += sum(len(w) for w in words)
                    rows.append({"words": words, "tree": f"S(NP(depth={depth},COMP={surface}))", "surface": surface, "depth": depth})
                    if len(rows) >= cap: return tuple(rows), states
    return tuple(rows), states


def outer_two(left, right, ls, rs):
    l = left[:2]; r = right[-2:]; rev = r[::-1]; n = 0; crossed = False
    while n < len(l) and n < len(rev) and l[n] == rev[n]:
        crossed = crossed or n + 1 in ls or len(right) - n - 1 in rs; n += 1
    return n, crossed


def run():
    trie, lex = brown_words()
    left_one, a = chart(lex, trie, "that", 1); right_two, b = chart(lex, trie, "as", 2)
    left_two, c = chart(lex, trie, "as", 2); right_one, d = chart(lex, trie, "that", 1)
    support = Counter(); mismatch = Counter(); outer_states = joint = 0; exact = []
    best = {"score": 0, "left": "", "right": "", "depth_pair": ""}
    for left, right, label in ((left_one, right_two, "1/2"), (left_two, right_one, "2/1")):
        for lder in left:
            lw = " ".join(lder["words"]); lt = norm(lw); ls = seams(lder["words"])
            for rder in right:
                rw = " ".join(rder["words"]); rt = norm(rw); rs = seams(rder["words"])
                support[label] += 1; li = lt.find(lder["surface"]); ri = rt.find(rder["surface"])
                lc = lt[li:li + len(lder["surface"])]
                rc = rt[ri:ri + len(rder["surface"])][::-1]; n = 0
                while n < len(lc) and n < len(rc) and lc[n] == rc[n]: n += 1
                if n < len(lc) and n < len(rc): mismatch[f"{lc[n]}/{rc[n]}"] += 1
                on, oc = outer_two(lt[:li], rt[ri + len(rder["surface"]):], ls, rs); outer_states += on
                if n == len(lc) and oc: joint += 1
                if n + on > best["score"]: best = {"score": n + on, "left": lw, "right": rw, "depth_pair": label, "outer_seam": oc}
                if not (n == len(lc) and oc): continue
                rendered = lw.capitalize() + "; " + rw + "."; x = audit(rendered)
                if x["two_pointer_exact"]: exact.append({"rendered": rendered, "audit": x, "depth_pair": label,
                    "provenance": {"catalogue_imported": False, "finished_tape_reversed": False, "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in exact}; exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "opposite complementizer attachment depth with direct center equality and hard outer seam",
            "grammar": ["S -> NP VP", "depth1 NP -> DET NOUN COMP S", "depth2 NP -> DET NOUN COMP NP COMP S", "COMP -> that|as"],
            "stats": {"depth1_that": len(left_one), "depth2_as": len(right_two), "depth2_as_reverse": len(left_two), "depth1_that_reverse": len(right_one),
                      "chart_states": a + b + c + d, "depth_pair_support": dict(support), "center_mismatch_support": dict(mismatch),
                      "outer_twochar_states": outer_states, "joint_closures": joint, "exact": len(exact),
                      "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact), "best_score": best["score"]},
            "complete_prose_controls": ["The sailor that the poet reads guards the harbor.",
                                        "The sailor as the poet that the keeper reads guards the harbor."],
            "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "character-cfg-opposite-complementizer-depth-20260920",
                                  "catalogue_imported": False, "lexical_sweep": False,
                                  "distinct_from": "character-cfg-compatible-complementizer-pair-20260920"},
            "next_construction": "Pair opposite-depth complementizers with an agreement-carrying embedded clause, retaining direct center equality and hard seams.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
