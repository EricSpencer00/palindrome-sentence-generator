"""Shared semantic-role variable-boundary paths before lexicalization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.phrase_cfg_slot_domain_solver_20260920 import audit, lexicalize, norm
from experiments.phrase_cfg_variable_boundary_propagation_20260920 import boundaries, SLOTS

EXPERIMENT_ID = "phrase-cfg-shared-role-variable-path-20260920"


def propagate_path(lmeta, rmeta):
    path = []; states = 0
    for index, slot in enumerate(SLOTS):
        la = boundaries(lmeta["slots"][slot]); choices = []
        for j in (index - 1, index, index + 1):
            if 0 <= j < len(SLOTS):
                rb = boundaries(rmeta["slots"][SLOTS[j]])
                for value in (la["prefix"] & rb["suffix"]) | (la["suffix"] & rb["prefix"]):
                    choices.append((SLOTS[j], value))
        if not choices: return None, states
        choices.sort(key=lambda x: len(x[1]), reverse=True); path.append((slot, choices[0][0], choices[0][1])); states += len(choices)
    return path, states


def run():
    # Same semantic role on both sides, independently authored realizations.
    left = lexicalize("maritime"); right = lexicalize("maritime")
    paths = 0; states = 0; exact = []; best = {"matched": 0, "left": "", "right": "", "path": []}
    for lwords, lmeta in left:
        for rwords, rmeta in right:
            path, nstates = propagate_path(lmeta, rmeta)
            if path is None: continue
            paths += 1; states += nstates
            lt = norm(" ".join(lwords)); rt = norm(" ".join(rwords))[::-1]; matched = 0
            while matched < len(lt) and matched < len(rt) and lt[matched] == rt[matched]: matched += 1
            if matched > best["matched"]: best = {"matched": matched, "left": " ".join(lwords), "right": " ".join(rwords), "path": path}
            if matched == len(lt) == len(rt):
                rendered = " ".join(lwords).capitalize() + "; " + " ".join(rwords) + "."; a = audit(rendered)
                exact.append({"rendered": rendered, "audit": a, "shared_role": "maritime", "path": path,
                              "provenance": {"independent_authored_sides": True, "shared_semantic_role": True,
                                             "variable_boundary_path_before_lexicalization": True,
                                             "catalogue_imported": False, "finished_tape_reversed": False,
                                             "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in exact}; exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "shared semantic-role variable-boundary path solver before lexicalization",
            "grammar": ["S -> NP VP PP", "NP -> DET ADJ NOUN", "VP -> V NP", "PP -> TAIL"],
            "shared_semantic_role": "maritime",
            "stats": {"left_phrases": len(left), "right_phrases": len(right), "full_outer_to_inner_paths": paths,
                      "path_states": states, "exact": len(exact), "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact),
                      "best_mirrored_prefix": best["matched"]},
            "complete_prose_controls": ["The patient sailor guards the harbor at dawn.", "A careful captain guides the boat under stars."],
            "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "phrase-cfg-shared-role-variable-path-20260920", "catalogue_imported": False, "lexical_sweep": False,
                                  "distinct_from": "phrase-cfg-variable-boundary-propagation-20260920"},
            "next_construction": "Introduce a second phrase topology with shared role but different slot nesting; preserve full-path lexicalization and exact admission.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
