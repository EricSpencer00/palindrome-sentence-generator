"""Shared causal boundary connective over crossed subordinate/matrix roles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "phrase-cfg-shared-because-connective-20260920"
DETS = ("some", "a", "the")
SUBJ = ("sailor", "poet", "keeper", "writer", "captain", "guide")
VERBS = ("guards", "marks", "guides", "keeps", "reads", "writes")
OBJS = ("harbor", "shore", "tide", "boat", "letter", "notes", "book", "garden")
ROLE = {"maritime": {"sailor", "harbor", "shore", "tide", "boat"},
        "writing": {"poet", "writer", "letter", "notes", "book"}}
CONNECTIVE = {"category": "CAUSAL_BECAUSE", "surface": "because", "role": "causal_explanation"}


def norm(s):
    return "".join(c.lower() for c in s if c.isalpha())


def audit(s):
    t = norm(s); r = t[::-1]
    return {"normalized": t, "letters": len(t), "two_pointer_exact": bool(t) and t == r,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(r.encode()).hexdigest()}


def sentences(subrole, matrixrole):
    out = []; sa = ROLE[subrole]; ma = ROLE[matrixrole]
    for d1 in DETS:
        for s1 in SUBJ:
            for v1 in VERBS:
                for o1 in OBJS:
                    if not ({s1, o1} & sa): continue
                    for d2 in DETS:
                        for s2 in SUBJ:
                            for v2 in VERBS:
                                for o2 in OBJS:
                                    if not ({s2, o2} & ma): continue
                                    words = ("because", d1, s1, v1, d1, o1, ",", d2, s2, v2, d2, o2)
                                    out.append((" ".join(words), {"subordinate_role": subrole,
                                        "matrix_role": matrixrole, "boundary": "because|comma",
                                        "tree": "Sub(because,S,S)"}))
                                    if len(out) >= 240: return tuple(out)
    return tuple(out)


def run():
    left = sentences("maritime", "writing"); right = sentences("writing", "maritime")
    states = support = 0; exact = []
    best = {"matched": 0, "left": "", "right": "", "connective": CONNECTIVE}
    for ltext, lmeta in left:
        lt = norm(ltext)
        for rtext, rmeta in right:
            if not (lmeta["boundary"] == rmeta["boundary"] == "because|comma"): continue
            support += 1; rt = norm(rtext)[::-1]; matched = 0
            while matched < len(lt) and matched < len(rt) and lt[matched] == rt[matched]:
                states += 1; matched += 1
            if matched > best["matched"]:
                best = {"matched": matched, "left": ltext, "right": rtext, "connective": CONNECTIVE}
            if matched == len(lt) == len(rt):
                rendered = ltext.capitalize() + "; " + rtext + "."; a = audit(rendered)
                exact.append({"rendered": rendered, "audit": a, "connective": CONNECTIVE,
                    "provenance": {"shared_boundary_connective": True, "crossed_roles": True,
                        "catalogue_imported": False, "finished_tape_reversed": False,
                        "word_order_mirror": False, "reader_status": "not run"}})
    exact = list({x["audit"]["normalized"]: x for x in exact}.values())
    return {"experiment_id": EXPERIMENT_ID,
        "method": "shared causal because connective with crossed subordinate/matrix roles",
        "grammar": ["S -> BECAUSE S , S", "S -> NP VP", "VP -> V NP"], "connective": CONNECTIVE,
        "stats": {"left_sentences": len(left), "right_sentences": len(right),
            "connective_support": support, "connective_character_states": states,
            "exact": len(exact), "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact),
            "best_matched_prefix": best["matched"]},
        "complete_prose_controls": ["Because the sailor guards the letter, the poet reads the harbor.",
            "Because a writer marks the shore, a captain guides the notes."],
        "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
        "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
        "novelty_preflight": {"status": "passed", "signature": EXPERIMENT_ID,
            "catalogue_imported": False, "lexical_sweep": False,
            "distinct_from": "phrase-cfg-shared-boundary-connective-20260920"},
        "next_construction": "Cross a concessive connective category with the same roles, retaining hard exact admission.",
        "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
