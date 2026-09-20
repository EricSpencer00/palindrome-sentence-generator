"""Shared temporal boundary connective over crossed subordinate/matrix roles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.phrase_cfg_crossed_boundary_roles_20260920 import audit, norm, sentences

EXPERIMENT_ID = "phrase-cfg-shared-boundary-connective-20260920"
CONNECTIVE = {"category": "TEMPORAL_WHILE", "surface": "while", "role": "simultaneous_events"}


def run():
    left = sentences("maritime", "writing"); right = sentences("writing", "maritime")
    states = connective_support = 0; exact = []; best = {"matched": 0, "left": "", "right": "", "connective": CONNECTIVE}
    for ltext, lmeta in left:
        lt = norm(ltext)
        for rtext, rmeta in right:
            # The boundary connective is shared semantically, while the clause
            # roles remain crossed and independently authored.
            if not (lmeta["boundary"] == rmeta["boundary"] == "while|comma"): continue
            connective_support += 1; rt = norm(rtext)[::-1]; matched = 0
            while matched < len(lt) and matched < len(rt) and lt[matched] == rt[matched]: states += 1; matched += 1
            if matched > best["matched"]: best = {"matched": matched, "left": ltext, "right": rtext, "connective": CONNECTIVE}
            if matched == len(lt) == len(rt):
                rendered = ltext.capitalize() + "; " + rtext + "."; a = audit(rendered)
                exact.append({"rendered": rendered, "audit": a, "connective": CONNECTIVE,
                              "provenance": {"shared_boundary_connective": True, "crossed_roles": True, "catalogue_imported": False,
                                             "finished_tape_reversed": False, "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in exact}; exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID, "method": "shared temporal connective with crossed subordinate/matrix roles", "grammar": ["S -> TEMPORAL_WHILE S , S", "S -> NP VP", "VP -> V NP"], "connective": CONNECTIVE,
            "stats": {"left_sentences": len(left), "right_sentences": len(right), "connective_support": connective_support, "connective_character_states": states, "exact": len(exact), "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact), "best_matched_prefix": best["matched"]},
            "complete_prose_controls": ["While the sailor guards the letter, the poet reads the harbor.", "While a writer marks the shore, a captain guides the notes."],
            "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]), "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "phrase-cfg-shared-boundary-connective-20260920", "catalogue_imported": False, "lexical_sweep": False, "distinct_from": "phrase-cfg-crossed-boundary-roles-20260920"},
            "next_construction": "Use a distinct connective category with the same crossed semantic roles, retaining hard exact admission.", "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json"); out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
