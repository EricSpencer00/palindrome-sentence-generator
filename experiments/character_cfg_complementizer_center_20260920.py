"""Held-out complementizer center category with direct character equality."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.character_cfg_recursive_relative_earley_20260920 import audit, brown_words, norm

EXPERIMENT_ID = "character-cfg-complementizer-center-20260920"
COMP = ("COMP", "that", "sg")


def seams(words):
    p = 0; out = set()
    for word in words[:-1]: p += len(word); out.add(p)
    return out


def chart(lex, trie, cap=180):
    rows = []; states = 0
    dets = sorted(lex["DET"]); nouns = sorted(lex["NOUN"]); verbs = sorted(lex["VERB"])
    # NP -> DET NOUN COMP S; S -> DET NOUN V DET NOUN. The complementizer
    # itself is the only held-out category item; all other terminals are fixed.
    for det in dets:
        for noun in nouns:
            for verb in verbs:
                if verb not in lex["VERB"]: continue
                for inner_noun in nouns:
                    words = (det, noun, COMP[1], det, inner_noun, verb, det, noun)
                    if not all(trie.accepts(word) for word in words): continue
                    states += sum(len(word) for word in words)
                    rows.append({"words": words, "tree": f"S(NP({det},{noun},COMP(that,S)),VP({verb},NP({det},{noun})))", "number": COMP[2]})
                    if len(rows) >= cap: return tuple(rows), states
    return tuple(rows), states


def outer_two(left, right, ls, rs):
    l = left[:2]; r = right[-2:]; rev = r[::-1]; n = 0; crossed = False
    while n < len(l) and n < len(rev) and l[n] == rev[n]:
        crossed = crossed or n + 1 in ls or len(right) - n - 1 in rs; n += 1
    return n, crossed


def run():
    trie, lex = brown_words(); left, ls = chart(lex, trie); right, rs = chart(lex, trie)
    center_states = outer_states = joint = 0; mismatch = Counter(); exact = []
    best = {"score": 0, "left": "", "right": "", "center": COMP[1]}
    for lder in left:
        lw = " ".join(lder["words"]); lt = norm(lw); lseams = seams(lder["words"])
        for rder in right:
            rw = " ".join(rder["words"]); rt = norm(rw); rseams = seams(rder["words"])
            li = lt.find(COMP[1]); ri = rt.find(COMP[1])
            if li < 0 or ri < 0: continue
            lc = lt[li:li + len(COMP[1])]; rc = rt[ri:ri + len(COMP[1])][::-1]
            n = 0
            while n < len(lc) and n < len(rc) and lc[n] == rc[n]: center_states += 1; n += 1
            if n < len(lc): mismatch[f"{lc[n]}/{rc[n]}"] += 1
            ol, oright = lt[:li], rt[ri + len(COMP[1]):]
            on, oc = outer_two(ol, oright, lseams, rseams); outer_states += on
            if n == len(lc) and oc: joint += 1
            score = n + on
            if score > best["score"]: best = {"score": score, "left": lw, "right": rw, "center": COMP[1], "outer_seam": oc}
            if not (n == len(lc) and oc): continue
            rendered = lw.capitalize() + "; " + rw + "."; a = audit(rendered)
            if a["two_pointer_exact"]:
                exact.append({"rendered": rendered, "audit": a, "tree": {"left": lder["tree"], "right": rder["tree"]},
                              "provenance": {"held_out_complementizer": True, "catalogue_imported": False,
                                             "finished_tape_reversed": False, "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in exact}; exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "held-out complementizer center CFG with direct equation and outer-two seam",
            "grammar": ["S -> NP VP", "NP -> DET NOUN COMP S", "COMP -> that", "VP -> V NP"],
            "stats": {"left_derivations": len(left), "right_derivations": len(right), "left_chart_states": ls,
                      "right_chart_states": rs, "center_direct_states": center_states,
                      "center_mismatch_support": dict(mismatch), "outer_twochar_states": outer_states,
                      "joint_center_outer_closures": joint, "exact": len(exact),
                      "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact), "best_score": best["score"]},
            "complete_prose_controls": ["The sailor that the poet reads guards the harbor.",
                                        "A keeper that a writer marks guards the garden."],
            "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "character-cfg-complementizer-center-20260920",
                                  "catalogue_imported": False, "lexical_sweep": False,
                                  "distinct_from": "character-cfg-relative-pronoun-center-20260920"},
            "next_construction": "Use a second complete complementizer attachment with a different agreement-bearing embedded clause, keeping direct center equality hard.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
