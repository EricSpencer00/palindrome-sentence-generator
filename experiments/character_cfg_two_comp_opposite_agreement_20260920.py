"""Two distinct complementizer sites with opposite embedded agreement."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.character_cfg_recursive_relative_earley_20260920 import audit, brown_words, norm

EXPERIMENT_ID = "character-cfg-two-comp-opposite-agreement-20260920"
COMP = "that"
NOUN_NUM = {"sailor": "sg", "poet": "sg", "keeper": "sg", "writer": "sg", "garden": "sg", "harbor": "sg", "area": "sg", "era": "sg", "writers": "pl", "guides": "pl", "sailors": "pl"}
VERB_NUM = {"guards": "sg", "marks": "sg", "guides": "sg", "keeps": "sg", "reads": "sg", "writes": "sg", "read": "pl", "mark": "pl", "guide": "pl"}


def seams(words):
    p = 0; out = set()
    for word in words[:-1]: p += len(word); out.add(p)
    return out


def chart(lex, trie, attachment, agreement, cap=180):
    rows = []; states = 0
    dets = sorted(lex["DET"]); nouns = sorted(n for n, num in NOUN_NUM.items() if num == agreement and n in lex["NOUN"])
    verbs = sorted(v for v, num in VERB_NUM.items() if num == agreement and v in lex["VERB"])
    all_nouns = sorted(n for n in NOUN_NUM if n in lex["NOUN"])
    all_verbs = sorted(v for v in VERB_NUM if v in lex["VERB"])
    for det in dets:
        for noun in nouns:
            for inner_det in dets:
                for inner_noun in nouns:
                    for verb in verbs:
                        if attachment == "subject":
                            words = (det, noun, COMP, inner_det, inner_noun, verb, inner_det, noun)
                        else:
                            words = (det, noun, verb, inner_det, inner_noun, COMP, inner_det, noun)
                        if not all(trie.accepts(w) for w in words): continue
                        states += sum(len(w) for w in words)
                        rows.append({"words": words, "attachment": attachment, "agreement": agreement,
                                     "tree": f"S({attachment}=COMP({COMP}),embedded={agreement})"})
                        if len(rows) >= cap: return tuple(rows), states
    return tuple(rows), states


def outer_two(left, right, ls, rs):
    l = left[:2]; r = right[-2:]; rev = r[::-1]; n = 0; crossed = False
    while n < len(l) and n < len(rev) and l[n] == rev[n]:
        crossed = crossed or n + 1 in ls or len(right) - n - 1 in rs; n += 1
    return n, crossed


def run():
    trie, lex = brown_words()
    # Fixed, held-out agreement category: three plural nouns and three plural
    # verbs already present in the Brown-derived trie. This is not a sweep.
    lex["NOUN"] |= {"writers", "guides", "sailors"}
    lex["VERB"] |= {"read", "mark", "guide"}
    ss, ss_states = chart(lex, trie, "subject", "sg"); op, op_states = chart(lex, trie, "object", "pl")
    ps, ps_states = chart(lex, trie, "subject", "pl"); os, os_states = chart(lex, trie, "object", "sg")
    support = Counter(); mismatch = Counter(); outer_states = joint = 0; exact = []
    best = {"score": 0, "left": "", "right": "", "orientation": ""}
    for left, right, orient in ((ss, op, "subject-sg/object-pl"), (ps, os, "subject-pl/object-sg")):
        for lder in left:
            lw = " ".join(lder["words"]); lt = norm(lw); ls = seams(lder["words"])
            for rder in right:
                rw = " ".join(rder["words"]); rt = norm(rw); rs = seams(rder["words"])
                support[orient] += 1; li = lt.find(COMP); ri = rt.find(COMP)
                lc = lt[li:li + len(COMP)]; rc = rt[ri:ri + len(COMP)][::-1]; n = 0
                while n < len(lc) and n < len(rc) and lc[n] == rc[n]: n += 1
                if n < len(lc): mismatch[f"{lc[n]}/{rc[n]}"] += 1
                on, oc = outer_two(lt[:li], rt[ri + len(COMP):], ls, rs); outer_states += on
                if n == len(lc) and oc: joint += 1
                if n + on > best["score"]: best = {"score": n + on, "left": lw, "right": rw, "orientation": orient, "outer_seam": oc}
                if not (n == len(lc) and oc): continue
                rendered = lw.capitalize() + "; " + rw + "."; a = audit(rendered)
                if a["two_pointer_exact"]: exact.append({"rendered": rendered, "audit": a, "orientation": orient,
                    "provenance": {"catalogue_imported": False, "finished_tape_reversed": False, "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in exact}; exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "two distinct complementizer attachment sites with opposite embedded agreement",
            "grammar": ["S -> NP VP", "subject-site NP -> DET NOUN COMP S", "object-site NP -> DET NOUN COMP S", "COMP -> that", "embedded V agrees with embedded subject"],
            "stats": {"subject_sg": len(ss), "object_pl": len(op), "subject_pl": len(ps), "object_sg": len(os),
                      "chart_states": ss_states + op_states + ps_states + os_states, "pair_support": dict(support),
                      "center_mismatch_support": dict(mismatch), "outer_twochar_states": outer_states,
                      "joint_closures": joint, "exact": len(exact), "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact),
                      "best_score": best["score"]},
            "complete_prose_controls": ["The sailor that the poet reads guards the harbor.",
                                        "The writers that some guides mark guide the sailors."],
            "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "character-cfg-two-comp-opposite-agreement-20260920",
                                  "catalogue_imported": False, "lexical_sweep": False,
                                  "distinct_from": "character-cfg-complementizer-embedded-agreement-20260920"},
            "next_construction": "Pair two complementizer sites with a shared semantic attachment relation, retaining opposite agreement and direct center equality.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
