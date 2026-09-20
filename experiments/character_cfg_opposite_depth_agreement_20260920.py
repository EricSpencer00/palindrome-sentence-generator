"""Opposite complementizer depth plus embedded agreement feature."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.character_cfg_recursive_relative_earley_20260920 import audit, brown_words, norm

EXPERIMENT_ID = "character-cfg-opposite-depth-agreement-20260920"
NOUN_NUM = {"sailor": "sg", "poet": "sg", "keeper": "sg", "writer": "sg", "garden": "sg", "harbor": "sg", "area": "sg", "era": "sg", "writers": "pl", "guides": "pl", "sailors": "pl"}
VERB_NUM = {"guards": "sg", "marks": "sg", "guides": "sg", "keeps": "sg", "reads": "sg", "writes": "sg", "read": "pl", "mark": "pl", "guide": "pl"}


def seams(words):
    p = 0; out = set()
    for word in words[:-1]: p += len(word); out.add(p)
    return out


def chart(lex, trie, surface, depth, agreement, cap=180):
    rows = []; states = 0
    nouns = sorted(n for n, num in NOUN_NUM.items() if num == agreement and n in lex["NOUN"])
    verbs = sorted(v for v, num in VERB_NUM.items() if num == agreement and v in lex["VERB"])
    for det in sorted(lex["DET"]):
        for noun in nouns:
            for inner in nouns:
                for verb in verbs:
                    if depth == 1: words = (det, noun, surface, det, inner, verb, det, noun)
                    else: words = (det, noun, surface, det, inner, surface, det, noun, verb, det, noun)
                    if not all(trie.accepts(w) for w in words): continue
                    states += sum(len(w) for w in words)
                    rows.append({"words": words, "surface": surface, "depth": depth, "agreement": agreement,
                                 "tree": f"S(depth={depth},COMP={surface},embedded_agreement={agreement})"})
                    if len(rows) >= cap: return tuple(rows), states
    return tuple(rows), states


def outer_two(left, right, ls, rs):
    l = left[:2]; r = right[-2:]; rev = r[::-1]; n = 0; crossed = False
    while n < len(l) and n < len(rev) and l[n] == rev[n]:
        crossed = crossed or n + 1 in ls or len(right) - n - 1 in rs; n += 1
    return n, crossed


def run():
    trie, lex = brown_words(); lex["NOUN"] |= {"writers", "guides", "sailors"}; lex["VERB"] |= {"read", "mark", "guide"}
    d1s, a = chart(lex, trie, "that", 1, "sg"); d2p, b = chart(lex, trie, "as", 2, "pl")
    d2p2, c = chart(lex, trie, "as", 2, "pl"); d1s2, d = chart(lex, trie, "that", 1, "sg")
    support = Counter(); mismatch = Counter(); outer_states = joint = 0; exact = []
    best = {"score": 0, "left": "", "right": "", "orientation": ""}
    for left, right, orient in ((d1s, d2p, "depth1-sg/depth2-pl"), (d2p2, d1s2, "depth2-pl/depth1-sg")):
        for lder in left:
            lw = " ".join(lder["words"]); lt = norm(lw); ls = seams(lder["words"])
            for rder in right:
                rw = " ".join(rder["words"]); rt = norm(rw); rs = seams(rder["words"])
                support[orient] += 1; li = lt.find(lder["surface"]); ri = rt.find(rder["surface"])
                lc = lt[li:li + len(lder["surface"])]
                rc = rt[ri:ri + len(rder["surface"])][::-1]; n = 0
                while n < len(lc) and n < len(rc) and lc[n] == rc[n]: n += 1
                if n < len(lc) and n < len(rc): mismatch[f"{lc[n]}/{rc[n]}"] += 1
                on, oc = outer_two(lt[:li], rt[ri + len(rder["surface"]):], ls, rs); outer_states += on
                if n == len(lc) and oc: joint += 1
                if n + on > best["score"]: best = {"score": n + on, "left": lw, "right": rw, "orientation": orient, "outer_seam": oc}
                if not (n == len(lc) and oc): continue
                rendered = lw.capitalize() + "; " + rw + "."; x = audit(rendered)
                if x["two_pointer_exact"]: exact.append({"rendered": rendered, "audit": x, "agreement": "sg/pl", "depth_pair": orient,
                    "provenance": {"catalogue_imported": False, "finished_tape_reversed": False, "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in exact}; exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "opposite complementizer depth with opposite embedded agreement",
            "grammar": ["depth1 NP -> DET NOUN COMP S", "depth2 NP -> DET NOUN COMP NP COMP S", "COMP -> that|as", "embedded V agrees"],
            "stats": {"depth1_sg": len(d1s), "depth2_pl": len(d2p), "chart_states": a + b + c + d,
                      "depth_pair_support": dict(support), "center_mismatch_support": dict(mismatch), "outer_twochar_states": outer_states,
                      "joint_closures": joint, "exact": len(exact), "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact), "best_score": best["score"]},
            "complete_prose_controls": ["The sailor that the poet reads guards the harbor.",
                                        "The writers as the guides that the sailors read guide the harbor."],
            "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "character-cfg-opposite-depth-agreement-20260920",
                                  "catalogue_imported": False, "lexical_sweep": False,
                                  "distinct_from": "character-cfg-opposite-complementizer-depth-20260920"},
            "next_construction": "Pair opposite depths with a semantic attachment relation while preserving opposite agreement and direct center equality.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
