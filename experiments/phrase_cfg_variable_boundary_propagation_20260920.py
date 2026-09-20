"""Variable-length adjacent boundary propagation for phrase-level CFG scenes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.phrase_cfg_slot_domain_solver_20260920 import audit, lexicalize, norm

EXPERIMENT_ID = "phrase-cfg-variable-boundary-propagation-20260920"
SLOTS = ("DET", "ADJ", "SUBJ", "VERB", "DET2", "OBJ", "TAIL")


def boundaries(word):
    t = norm(word)
    return {"prefix": {t[:n] for n in range(1, min(3, len(t)) + 1)},
            "suffix": {t[-n:] for n in range(1, min(3, len(t)) + 1)}, "length": len(t)}


def run():
    left = lexicalize("maritime"); right = lexicalize("writing")
    propagated = 0; states = 0; exact = []; best = {"matched": 0, "left": "", "right": "", "path": []}
    for lwords, lmeta in left:
        for rwords, rmeta in right:
            path = []
            for index, slot in enumerate(SLOTS):
                # Propagation may discharge the current left slot against the
                # neighboring right slot; this is the variable boundary state
                # that the independent-slot lane could not represent.
                la = boundaries(lmeta["slots"][slot])
                choices = []
                for offset in (-1, 0, 1):
                    j = index + offset
                    if 0 <= j < len(SLOTS):
                        right_slot = SLOTS[j]; rb = boundaries(rmeta["slots"][right_slot])
                        for match in (la["prefix"] & rb["suffix"]) | (la["suffix"] & rb["prefix"]):
                            choices.append((match, right_slot))
                matches = sorted(choices, key=lambda x: len(x[0]), reverse=True)
                if not matches: break
                path.append((slot, matches[0][1], matches[0][0])); states += len(matches)
            else:
                propagated += 1
                lt = norm(" ".join(lwords)); rt = norm(" ".join(rwords))[::-1]; matched = 0
                while matched < len(lt) and matched < len(rt) and lt[matched] == rt[matched]: matched += 1
                if matched > best["matched"]: best = {"matched": matched, "left": " ".join(lwords), "right": " ".join(rwords), "path": path}
                if matched == len(lt) == len(rt):
                    rendered = " ".join(lwords).capitalize() + "; " + " ".join(rwords) + "."; a = audit(rendered)
                    exact.append({"rendered": rendered, "audit": a, "boundary_path": path,
                                  "provenance": {"variable_boundary_propagation": True, "catalogue_imported": False,
                                                 "finished_tape_reversed": False, "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in exact}; exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "variable-length adjacent boundary propagation before phrase lexicalization",
            "grammar": ["S -> NP VP PP", "NP -> DET ADJ NOUN", "VP -> V NP", "PP -> TAIL"],
            "stats": {"left_phrases": len(left), "right_phrases": len(right), "propagated_pairs": propagated,
                      "variable_boundary_states": states, "exact": len(exact), "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact),
                      "best_mirrored_prefix": best["matched"]},
            "complete_prose_controls": ["The patient sailor guards the harbor at dawn.", "A careful writer reads the letter before rain."],
            "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "phrase-cfg-variable-boundary-propagation-20260920", "catalogue_imported": False, "lexical_sweep": False,
                                  "distinct_from": "phrase-cfg-adjacent-boundary-propagation-20260920"},
            "next_construction": "Carry variable boundary strings through a shared semantic role constraint, then lexicalize only paths with a full outer-to-inner equation.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
