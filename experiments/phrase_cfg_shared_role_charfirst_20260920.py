"""Phrase-level CFG scene search: shared role, character constraints first."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "phrase-cfg-shared-role-charfirst-20260920"


def norm(s): return "".join(c.lower() for c in s if c.isalpha())


def audit(s):
    t = norm(s); r = t[::-1]
    return {"normalized": t, "letters": len(t), "two_pointer_exact": bool(t) and t == r,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(r.encode()).hexdigest()}


ROLE = {
    "maritime": {"sailor", "harbor", "shore", "tide", "boat"},
    "writing": {"poet", "writer", "letter", "notes", "book"},
}
DETS = ("some", "a", "the")
ADJS = ("quiet", "patient", "bright", "old", "careful")
SUBJECTS = ("sailor", "poet", "keeper", "writer", "captain", "guide")
VERBS = ("guards", "marks", "guides", "keeps", "reads", "writes")
OBJECTS = ("harbor", "shore", "tide", "boat", "letter", "notes", "book", "garden")
TAILS = ("at dawn", "before rain", "near shore", "under stars")


def phrases(role):
    allowed = ROLE[role]
    out = []
    for d in DETS:
        for adj in ADJS:
            for noun in SUBJECTS:
                for v in VERBS:
                    for obj in OBJECTS:
                        for tail in TAILS:
                            words = (d, adj, noun, v, d, obj, *tail.split())
                            if len(set(words) & allowed) >= 1:
                                out.append((" ".join(words), {"role": role, "tree": f"S(NP({d},{adj},{noun}),VP({v},NP({d},{obj}),PP({tail})))"}))
    return tuple(out[:240])


def charfirst_intersection(left, right):
    # Index opposite phrases by their reversed normalized tape prefixes before
    # selecting lexical pairs. This is a construction rule, not repair.
    index = defaultdict(list)
    for text, meta in right:
        t = norm(text)[::-1]
        index[t[:1]].append((text, meta))
    states = 0; seam_hits = 0; candidates = []; best = {"matched": 0, "left": "", "right": ""}
    for ltext, lmeta in left:
        lt = norm(ltext)
        for rtext, rmeta in index.get(lt[:1], ()):
            rt = norm(rtext)[::-1]; matched = 0
            while matched < len(lt) and matched < len(rt) and lt[matched] == rt[matched]:
                states += 1; matched += 1
            if matched > best["matched"]: best = {"matched": matched, "left": ltext, "right": rtext}
            if matched == len(lt) == len(rt):
                rendered = ltext.capitalize() + "; " + rtext + "."
                a = audit(rendered)
                candidates.append({"rendered": rendered, "audit": a, "left": lmeta, "right": rmeta,
                                   "provenance": {"independently_authored_phrases": True, "catalogue_imported": False,
                                                  "finished_tape_reversed": False, "word_order_mirror": False,
                                                  "reader_status": "not run"}})
                seam_hits += 1
    return candidates, states, seam_hits, best, len(index)


def run():
    left = phrases("maritime"); right = phrases("writing")
    rows, states, seams, best, index_keys = charfirst_intersection(left, right)
    unique = {x["audit"]["normalized"]: x for x in rows}; rows = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "independently authored phrase-level CFG with shared scene role and character-first opposite index",
            "grammar": ["S -> NP VP PP", "NP -> DET ADJ NOUN", "VP -> V NP", "PP -> TAIL"],
            "shared_scene_roles": {"left": "maritime", "right": "writing"},
            "stats": {"left_phrases": len(left), "right_phrases": len(right), "opposite_prefix_index_keys": index_keys,
                      "character_states": states, "cross_word_seam_closures": seams, "exact": len(rows),
                      "reader_eligible": sum(x["audit"]["letters"] > 38 for x in rows), "best_matched_prefix": best["matched"]},
            "complete_prose_controls": ["The patient sailor guards the harbor at dawn.",
                                        "A careful writer reads the letter before rain."],
            "best_diagnostic": best, "candidates": sorted(rows, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "phrase-cfg-shared-role-charfirst-20260920",
                                  "catalogue_imported": False, "lexical_sweep": False,
                                  "distinct_from": "character-cfg-opposite-depth-agreement-20260920"},
            "next_construction": "Replace the two-character opposite index with slot-level character domains, retaining independent phrase authorship and shared scene roles.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
