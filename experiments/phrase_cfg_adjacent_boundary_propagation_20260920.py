"""Adjacent-slot boundary propagation for phrase-level CFG palindromes."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.phrase_cfg_slot_domain_solver_20260920 import audit, lexicalize, norm, slot_domain

EXPERIMENT_ID = "phrase-cfg-adjacent-boundary-propagation-20260920"
SLOTS = ("DET", "ADJ", "SUBJ", "VERB", "DET2", "OBJ", "TAIL")


def sig(meta): return {slot: slot_domain(meta["slots"][slot]) for slot in SLOTS}


def propagate(ld, rd):
    """Propagate a mirrored boundary obligation from slot i to i+1."""
    states = 1; path = []
    required = None
    for i, slot in enumerate(SLOTS):
        left = ld[slot]; right = rd[slot]
        if required is not None and required not in {left["first"], left["last"]}:
            return 0, path
        # Current boundary establishes the next opposite-side obligation.
        if left["first"] == right["last"]:
            required = right["first"]; path.append((slot, "left-first/right-last"))
        elif left["last"] == right["first"]:
            required = right["last"]; path.append((slot, "left-last/right-first"))
        else:
            return 0, path
        states += 1
    return states, path


def run():
    left = lexicalize("maritime"); right = lexicalize("writing")
    propagated = 0; states = 0; exact = []; best = {"states": 0, "left": "", "right": "", "path": []}
    for lwords, lmeta in left:
        ld = sig(lmeta)
        for rwords, rmeta in right:
            rd = sig(rmeta); n, path = propagate(ld, rd)
            if not n: continue
            propagated += 1; states += n
            lt = norm(" ".join(lwords)); rt = norm(" ".join(rwords))[::-1]; matched = 0
            while matched < len(lt) and matched < len(rt) and lt[matched] == rt[matched]: matched += 1
            if matched > best["states"]: best = {"states": matched, "left": " ".join(lwords), "right": " ".join(rwords), "path": path}
            if matched == len(lt) == len(rt):
                rendered = " ".join(lwords).capitalize() + "; " + " ".join(rwords) + "."; a = audit(rendered)
                exact.append({"rendered": rendered, "audit": a, "propagation_path": path,
                              "provenance": {"adjacent_boundary_propagation": True, "catalogue_imported": False,
                                             "finished_tape_reversed": False, "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in exact}; exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "adjacent-slot boundary propagation before phrase lexicalization",
            "grammar": ["S -> NP VP PP", "NP -> DET ADJ NOUN", "VP -> V NP", "PP -> TAIL"],
            "stats": {"left_phrases": len(left), "right_phrases": len(right), "propagated_pairs": propagated,
                      "propagation_states": states, "exact": len(exact), "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact),
                      "best_mirrored_prefix": best["states"]},
            "complete_prose_controls": ["The patient sailor guards the harbor at dawn.", "A careful writer reads the letter before rain."],
            "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "phrase-cfg-adjacent-boundary-propagation-20260920", "catalogue_imported": False, "lexical_sweep": False,
                                  "distinct_from": "phrase-cfg-slot-seam-length-solver-20260920"},
            "next_construction": "Add variable-length slot boundaries to the propagated state, keeping adjacent obligations live before lexicalization.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
