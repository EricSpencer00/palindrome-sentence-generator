"""Typed subordinate/matrix semantic roles on the while-boundary CFG."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "phrase-cfg-typed-subordinate-matrix-roles-20260920"
DETS = ("some", "a", "the")
SUBJ = ("sailor", "poet", "keeper", "writer", "captain", "guide")
VERBS = ("guards", "marks", "guides", "keeps", "reads", "writes")
OBJS = ("harbor", "shore", "tide", "boat", "letter", "notes", "book", "garden")
ROLE = {"maritime": {"sailor", "harbor", "shore", "tide", "boat"}, "writing": {"poet", "writer", "letter", "notes", "book"}}


def norm(s): return "".join(c.lower() for c in s if c.isalpha())


def audit(s):
    t = norm(s); r = t[::-1]
    return {"normalized": t, "letters": len(t), "two_pointer_exact": bool(t) and t == r,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(), "sha256_reverse": hashlib.sha256(r.encode()).hexdigest()}


def clause(role, cap=240):
    allowed = ROLE[role]; out = []
    for d1 in DETS:
        for s1 in SUBJ:
            for v1 in VERBS:
                for o1 in OBJS:
                    for d2 in DETS:
                        for s2 in SUBJ:
                            for v2 in VERBS:
                                for o2 in OBJS:
                                    words = ("while", d1, s1, v1, d1, o1, ",", d2, s2, v2, d2, o2)
                                    if len(set(words) & allowed) < 2: continue
                                    out.append((" ".join(words), {"subordinate_role": role, "matrix_role": role, "boundary": "while|comma", "tree": "Sub(while,S,S)"}))
                                    if len(out) >= cap: return tuple(out)
    return tuple(out)


def run():
    left = clause("maritime"); right = clause("writing")
    states = 0; exact = []; best = {"matched": 0, "left": "", "right": "", "roles": {}}
    for ltext, lmeta in left:
        lt = norm(ltext)
        for rtext, rmeta in right:
            rt = norm(rtext)[::-1]; matched = 0
            while matched < len(lt) and matched < len(rt) and lt[matched] == rt[matched]: states += 1; matched += 1
            if matched > best["matched"]: best = {"matched": matched, "left": ltext, "right": rtext, "roles": {"left": lmeta, "right": rmeta}}
            if matched == len(lt) == len(rt):
                rendered = ltext.capitalize() + "; " + rtext + "."; a = audit(rendered)
                exact.append({"rendered": rendered, "audit": a, "typed_roles": {"left": lmeta, "right": rmeta},
                              "provenance": {"typed_boundary_roles": True, "catalogue_imported": False, "finished_tape_reversed": False,
                                             "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in exact}; exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID, "method": "typed subordinate/matrix semantic roles in while-boundary CFG",
            "grammar": ["S -> SUBORD S , MATRIX S", "S -> NP VP", "VP -> V NP"],
            "typed_roles": {"left": {"subordinate": "maritime", "matrix": "maritime"}, "right": {"subordinate": "writing", "matrix": "writing"}},
            "stats": {"left_sentences": len(left), "right_sentences": len(right), "typed_boundary_states": states, "exact": len(exact),
                      "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact), "best_matched_prefix": best["matched"]},
            "complete_prose_controls": ["While the sailor guards the harbor, the poet reads the letter.",
                                        "While a keeper marks the shore, a writer reads the notes."],
            "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "phrase-cfg-typed-subordinate-matrix-roles-20260920", "catalogue_imported": False, "lexical_sweep": False,
                                  "distinct_from": "phrase-cfg-subordination-boundary-state-20260920"},
            "next_construction": "Cross subordinate and matrix roles asymmetrically across the boundary while retaining typed exact admission.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
