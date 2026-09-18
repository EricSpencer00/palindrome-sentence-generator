"""Finite feasibility screen for moving an owner out of the pre-art gap.

The source grammar composes: actor SORT [owner's damaged artwork], provenance
relative, AND REPAIR tears IN [THIS high-gloss red art].  The validator in a
different module must independently resolve THIS to the owned first object.

Six next-pair-compatible surface alternatives are not six full palindrome
completions.  Both counts are reported.  Exhausting this bounded grammar with
zero complete admissible strings disables transport even when its eight-pair
probe has many exact one-pair continuations.  There is no model client here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.ownership_coordination_validator_20260913 import parse_sentence, render_sentence
from experiments.residual_constrained_attachment_infill_20260913 import finite_preflight, fringe_trace
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


def source_paths():
    """Independently declared finite source productions, not reflected tapes."""
    origins = ((), ("local",), ("foreign",))
    qualifications = ((), ("young",), ("retired",), ("skilled",))
    owners = ("artists", "painters", "collectors", "curators", "dealers", "patrons")
    collections = ("paintings", "prints", "posters")
    scales = ("small", "large")
    for origin, qualification, owner, collection, scale in product(origins, qualifications, owners, collections, scales):
        possessor = origin + qualification + (owner,)
        words = (("traders", "sort") + possessor + ("damaged", collection)
                 + ("that", "arrived", "from", "regional", "museums", "and", "repair", scale,
                    "tears", "in", "this", "high", "gloss", "red", "art"))
        yield {"words": words, "source_analysis": {
            "construction": "owner_in_first_conjunct_object",
            "possessor_span": [2, 2 + len(possessor)],
            "possessed_patient": "collection", "collection_head": collection,
            "repair_defect_bearer": "collection", "owner_outside_pre_art_gap": True,
            "reference_rule": "unique artwork-compatible antecedent for mass demonstrative"}}


def review(path):
    words = tuple(path["words"])
    rendered = render_sentence(words)
    tape = normalize_letters(rendered)
    central = mechanical_admission_checks(rendered, min_letters=100, max_letters=240)
    parses = parse_sentence(words)
    semantics = any(p["complete"] and p["semantic_relation_valid"] for p in parses)
    trace = fringe_trace(words)
    failures = [k for k, v in central.items() if not v]
    if not semantics:
        failures.append("independent_whole_sentence_semantics")
    # An exact string pruned online still retains its full rendered review.
    return {"words": words, "source_analysis": path["source_analysis"],
            "rendered_diagnostic": rendered, "normalized": tape, "letters": len(tape),
            "central_admission": central, "independent_parses": parses,
            "fringe_trace": trace, "failures": failures,
            "eligible_for_external_provenance": semantics and all(central.values()) and trace["termination"] == "closure",
            "external_provenance": "not_checked", "originality_claim": False, "promoted": False}


def screen(paths=None, *, probe_pairs=8, minimum_alternatives=6):
    if probe_pairs < 8 or minimum_alternatives < 6:
        raise ValueError("the eight-pair/six-alternative probe must not be relaxed")
    rows = [review(p) for p in (source_paths() if paths is None else paths)]
    by_channel = defaultdict(list)
    for row in rows:
        tape = row["normalized"]
        semantics = any(p["semantic_relation_valid"] for p in row["independent_parses"])
        # All nonexact admission checks still apply to a partial continuation.
        nonexact = all(v for k, v in row["central_admission"].items() if k != "exact_letter_palindrome")
        if (semantics and nonexact and row["fringe_trace"]["actual_pairs"] >= probe_pairs + 1
                and len(tape) > 2 * (probe_pairs + 1)):
            by_channel[tape[:probe_pairs]].append(row)
    channels = []
    for tape, matching in sorted(by_channel.items()):
        surfaces = {tuple(r["words"]) for r in matching}
        next_letters = sorted({r["normalized"][probe_pairs] for r in matching})
        channels.append({"outer_tape": tape, "actual_probe_pairs": probe_pairs,
                         "continued_actual_pairs": probe_pairs + 1,
                         "next_pair_compatible_source_derivations": len(matching),
                         "distinct_next_pair_compatible_surfaces": len(surfaces),
                         "distinct_next_pair_letters": next_letters,
                         "six_surface_alternative_probe_passed": len(surfaces) >= minimum_alternatives,
                         "six_distinct_next_letters_passed": len(next_letters) >= minimum_alternatives,
                         "witnesses": [{"rendered_diagnostic": r["rendered_diagnostic"],
                                        "actual_pairs": r["fringe_trace"]["actual_pairs"],
                                        "next_left": r["normalized"][probe_pairs],
                                        "next_right": r["normalized"][-probe_pairs - 1],
                                        "owner_span": r["independent_parses"][0]["ownership"]["surface_span"],
                                        "reference": r["independent_parses"][0]["reference"]}
                                       for r in matching]})
    mismatches = Counter((r["fringe_trace"].get("mismatch_pair"), r["fringe_trace"].get("left"),
                          r["fringe_trace"].get("right")) for r in rows)
    closures = [r for r in rows if r["central_admission"]["exact_letter_palindrome"]]
    survivors = [r for r in closures if r["eligible_for_external_provenance"]]
    return {"source_derivations": len(rows), "distinct_rendered_surfaces": len({tuple(r["words"]) for r in rows}),
            "states_exhausted": True, "probe_pairs": probe_pairs, "minimum_surface_alternatives": minimum_alternatives,
            "actual_pair_depth_distribution": dict(sorted(Counter(r["fringe_trace"]["actual_pairs"] for r in rows).items())),
            "channels": channels,
            "qualified_eight_pair_six_surface_probe_channels": sum(c["six_surface_alternative_probe_passed"] for c in channels),
            "qualified_eight_pair_six_letter_probe_channels": sum(c["six_distinct_next_letters_passed"] for c in channels),
            "full_exact_completion_surfaces": len({r["rendered_diagnostic"] for r in closures}),
            "central_and_semantic_survivors": len(survivors),
            "outer_mismatches": [{"pair": p, "left": a, "right": b, "source_derivations": n}
                                 for (p, a, b), n in sorted(mismatches.items(), key=lambda item: str(item[0]))],
            "failure_counts": dict(sorted(Counter(f for r in rows for f in r["failures"]).items())),
            "reviews": rows, "closure_reviews": closures,
            "local_infill_transport_eligible": bool(survivors),
            "transport_disabled_reason": None if survivors else "exhaustive_finite_grammar_has_no_complete_admissible_continuation"}


def run():
    result = screen()
    control = finite_preflight()
    files = (Path(__file__), ROOT / "experiments/ownership_coordination_validator_20260913.py",
             ROOT / "llm_palindrome/admission.py", ROOT / "data/known_palindromes.json")
    return {"experiment": "ownership-argument-order-residual-feasibility-v1",
            "status": "nine_pair_partial_channel_but_no_complete_continuation",
            "argument_order_screen": result,
            "fixed_owner_control": {"source_derivations": control["source_attachment_derivations"],
                                    "distinct_rendered_surfaces": control["distinct_rendered_surfaces"],
                                    "actual_pair_depth_distribution": control["actual_pair_depth_distribution"],
                                    "exact_closures": control["exact_closures"],
                                    "first_mismatch_pairs": sorted({r["fringe_trace"]["mismatch_pair"] for r in control["reviews"]}),
                                    "evidence": "runs/residual-constrained-attachment-infill-20260913.json"},
            "provenance": {"files": {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in files},
                           "external_searches": 0, "external_check_required_before_any_promotion": True,
                           "originality_claim": False},
            "model_transport_implemented": False, "model_queries_executed": 0,
            "next_repair": "The fixed sort/high-gloss channel fails at pair 10 (r/l). A next repair must jointly change the repair-event clause onset and the final material-property realization, rather than enumerate more owners inside this already exhausted coordination.",
            "promoted_candidates": []}


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    result = run()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = {k: v for k, v in result["argument_order_screen"].items()
               if k not in {"reviews", "channels", "closure_reviews"}}
    print(json.dumps({"status": result["status"], "screen": summary,
                      "control": result["fixed_owner_control"], "model_queries_executed": 0}, indent=2))
