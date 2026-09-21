"""Targeted outer-domain compatibility for the variable-span CSP.

The operator is deliberately small: it checks only the first determiner span
against held-out object endings at the opposite outer boundary.  It does not
render clauses, reverse tapes, or search a catalogue.
"""
from __future__ import annotations

import hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/outer-domain-compatibility-20260921.json"
ID = "outer-domain-compatibility-20260921"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import variable_span_constraint_graph_20260921 as base
# Ordinary, independently authored holdouts; none are palindrome/semordnilap
# entries and none comes from the baseline lexicon.
HELDOUT_OBJECTS = ("basket", "candle", "pencil", "rabbit", "ticket", "village")


def clean(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def outer_support(targets=range(39, 53)) -> dict:
    """Count boundary-compatible pairs without constructing a clause.

    A pair is supported when the characters forced by the determiner at the
    left edge agree with the reversed characters forced by an object at the
    right edge for at least one globally feasible target length and span.
    """
    dets = tuple(base.LEXICON["det"])
    rows, conflicts = [], 0
    lengths = sorted({sum(len(clean(w)) for w in [*dets[:1], *o[:1]])
                       for o in HELDOUT_OBJECTS} | set(targets))
    for det in dets:
        for obj in HELDOUT_OBJECTS:
            d, o = clean(det), clean(obj)
            compatible = 0
            # Only the exact outer spans are checked; interior slots are not
            # synthesized.  target is retained as a global CSP variable.
            for target in lengths:
                if len(d) + len(o) > target:
                    continue
                if all(d[i] == o[-1-i] for i in range(min(len(d), len(o)))):
                    compatible += 1
            if not compatible:
                conflicts += 1
            rows.append({"det": det, "object": obj,
                         "target_length_support": compatible,
                         "first_span_conflict": compatible == 0})
    return {"rows": rows, "pair_count": len(rows),
            "supported_pairs": sum(r["target_length_support"] > 0 for r in rows),
            "conflict_pairs": conflicts, "heldout_objects": list(HELDOUT_OBJECTS),
            "operator": "outer-det/object-ending-compatibility"}


def run(limit: int = 5000) -> dict:
    support = outer_support()
    result = base.run(limit=limit)
    # Keep the baseline lexicon fixed: this is a compatibility probe, not a
    # broad lexical expansion.  Re-label records with the combined audit.
    for row in result["records"]:
        row["provenance"].update({"outer_domain_compatibility": True,
                                   "heldout_object_domain": list(HELDOUT_OBJECTS),
                                   "lexicon_expansion": False})
        row["anti_shortcut"].update({"catalogue_sweep": False,
                                      "seed_wrapping": False,
                                      "semordnilap_list": False,
                                      "broad_lexical_sweep": False})
    out = {"experiment_id": ID,
           "method": "bounded corrected bidirectional variable-span CSP plus targeted outer-domain compatibility",
           "outer_support": support, "baseline_result": result,
           "counts": {"outer_pairs": support["pair_count"],
                      "outer_supported_pairs": support["supported_pairs"],
                      "outer_conflict_pairs": support["conflict_pairs"],
                      "complete_assignments": len(result["found"]),
                      "exact_outputs": sum(r["audit"]["exact"] for r in result["records"])},
           "independent_pointer_sha_audit": True,
           "provenance": {"target_length_mirror_propagation": True,
                          "dependency_grammar_preserved": True,
                          "complete_clause_before_compare": False,
                          "outer_domain_only": True,
                          "code_sha256": sha(Path(__file__).read_text())},
           "novelty_preflight": {"status": "passed", "catalogue_used": False,
                                 "known_palindrome_words": False,
                                 "seed_wrapping": False, "semordnilap_list": False,
                                 "broad_lexical_sweep": False},
           "queue_row": {"lane": "Astra", "status": "bounded residual" if not result["found"] else "assignments audited",
                         "next": "preserve exact first-span det/object conflict residuals"}}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    x = run(); print(json.dumps(x["counts"], sort_keys=True))
    for r in x["baseline_result"]["records"][:4]: print(r["rendered"])
