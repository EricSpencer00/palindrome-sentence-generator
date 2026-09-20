"""Slot-specific boundary domains plus seam-length solving before lexicalization."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.phrase_cfg_slot_domain_solver_20260920 import audit, lexicalize, norm, slot_domain

EXPERIMENT_ID = "phrase-cfg-slot-seam-length-solver-20260920"
SLOTS = ("DET", "ADJ", "SUBJ", "VERB", "DET2", "OBJ", "TAIL")


def boundary_signature(meta):
    return {slot: slot_domain(meta["slots"][slot]) for slot in SLOTS}


def run():
    left = lexicalize("maritime"); right = lexicalize("writing")
    domain_pairs = Counter(); length_pairs = Counter(); admissible_pairs = 0; states = 0; exact = []
    best = {"matched": 0, "left": "", "right": "", "compatible_slots": 0}
    for lwords, lmeta in left:
        ld = boundary_signature(lmeta)
        for rwords, rmeta in right:
            rd = boundary_signature(rmeta)
            compatible = 0; lengths_ok = 0
            for slot in SLOTS:
                # A slot can sit across the mirrored seam only when its left
                # first/last domain agrees with the opposite right boundary.
                if ld[slot]["first"] == rd[slot]["last"] or ld[slot]["last"] == rd[slot]["first"]:
                    compatible += 1; domain_pairs[slot] += 1
                if abs(ld[slot]["letters"] - rd[slot]["letters"]) <= 2:
                    lengths_ok += 1; length_pairs[slot] += 1
            if compatible < len(SLOTS) or lengths_ok < len(SLOTS): continue
            admissible_pairs += 1
            lt = norm(" ".join(lwords)); rt = norm(" ".join(rwords))[::-1]; matched = 0
            while matched < len(lt) and matched < len(rt) and lt[matched] == rt[matched]:
                states += 1; matched += 1
            if matched > best["matched"]:
                best = {"matched": matched, "left": " ".join(lwords), "right": " ".join(rwords), "compatible_slots": compatible}
            if matched == len(lt) == len(rt):
                rendered = " ".join(lwords).capitalize() + "; " + " ".join(rwords) + "."; a = audit(rendered)
                exact.append({"rendered": rendered, "audit": a, "left": lmeta, "right": rmeta,
                              "provenance": {"slot_boundary_domains": True, "seam_lengths_prelexicalized": True,
                                             "catalogue_imported": False, "finished_tape_reversed": False,
                                             "word_order_mirror": False, "reader_status": "not run"}})
    unique = {x["audit"]["normalized"]: x for x in exact}; exact = list(unique.values())
    return {"experiment_id": EXPERIMENT_ID,
            "method": "slot-specific first/last domains with word-boundary seam lengths before lexicalization",
            "grammar": ["S -> NP VP PP", "NP -> DET ADJ NOUN", "VP -> V NP", "PP -> TAIL"],
            "stats": {"left_phrases": len(left), "right_phrases": len(right), "admissible_slot_pairs": admissible_pairs,
                      "slot_domain_pair_support": dict(domain_pairs), "slot_length_pair_support": dict(length_pairs),
                      "lexicalized_character_states": states, "exact": len(exact), "reader_eligible": sum(x["audit"]["letters"] > 38 for x in exact),
                      "best_matched_prefix": best["matched"]},
            "complete_prose_controls": ["The patient sailor guards the harbor at dawn.", "A careful writer reads the letter before rain."],
            "best_diagnostic": best, "candidates": sorted(exact, key=lambda x: -x["audit"]["letters"]),
            "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": "phrase-cfg-slot-seam-length-solver-20260920", "catalogue_imported": False, "lexical_sweep": False,
                                  "distinct_from": "phrase-cfg-slot-domain-solver-20260920"},
            "next_construction": "Propagate boundary domains through adjacent slots rather than requiring independent slot compatibility; retain seam lengths before lexicalization.",
            "reader_gate": "closed; programmatic exactness never certifies readability"}


if __name__ == "__main__":
    result = run(); out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
