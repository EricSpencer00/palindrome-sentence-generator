"""Finite measurement-cleft feasibility, scanning every frontier depth >=11.

Two whole semantic analyses are generated without reversing a source tape.
Central admission and independent scalar/identity parsing precede all channel
qualification.  The search reports actual depth and live cuts, never raw
boundary geometry.  No full palindrome product or model interface is built.
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
from experiments.measurement_cleft_source_parser_20260913 import parse_source
from experiments.measurement_cleft_final_parser_20260913 import parse_final, render_final
from experiments.joint_onset_material_feasibility_20260913 import boundary_witness
from experiments.residual_constrained_attachment_infill_20260913 import fringe_trace
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


def productions():
    materials = ((("red", "metalwork"), ("red", "metal", "work")),
                 (("old", "woodwork"), ("old", "wood", "work")),
                 (("clear", "glasswork"), ("clear", "glass", "work")))
    roles = (
        ("height", "lower", ("sections", "sectors"), ("ridges",), ("ridge", "height"),
         ("coarse", "sandpaper"), ("coarse", "sand", "paper"), ("reduces", "cuts"), "reduces"),
        ("roughness", "lower", ("surfaces", "patches"), ("areas",), ("surface", "roughness"),
         ("fine", "sandpaper"), ("fine", "sand", "paper"), ("smooths",), "reduces"),
        ("firmness", "higher", ("joints", "seams"), ("connections",), ("joint", "firmness"),
         ("fresh", "adhesive"), ("fresh", "adhesive"), ("secures",), "increases"),
        ("permeability", "lower", ("slabs", "panels"), ("plates",), ("surface", "permeability"),
         ("fresh", "adhesive"), ("fresh", "adhesive"), ("seals",), "reduces"),
    )
    discourse = (((), ()), (("nonetheless",), ("none", "the", "less")))
    for material, role, concession in product(materials, roles, discourse):
        metric, comparative, heads, feature, measure, source_cause, target_cause, headed_verbs, scalar_verb = role
        variants = [("scalar_what_cleft", ("what", "is", comparative), "is", measure, scalar_verb)]
        variants += [("headed_relative_equative", (head, "whose", metric, "is", comparative), "are", feature, verb)
                     for head, verb in product(heads, headed_verbs)]
        for mode, beginning, copula, identity, action in variants:
            prefix = beginning + ("than", "before", "the", "recent", "repair", "on", "the")
            tail = (copula, "the") + identity + ("that",)
            yield {"source_tokens": concession[0] + prefix + material[0] + tail + source_cause + (action,),
                   "target_tokens": concession[1] + prefix + material[1] + tail + target_cause + (action,),
                   "source_relation": {"mode": mode, "property": metric, "finite_repair_predicate": action,
                                       "equative_patient_binding": "one_feature_on_one_material"}}


def review(path):
    source, target = tuple(path["source_tokens"]), tuple(path["target_tokens"])
    text, source_text = render_final(target), " ".join(source).capitalize() + "."
    checks = mechanical_admission_checks(text, min_letters=100, max_letters=240)
    source_checks = mechanical_admission_checks(source_text, min_letters=100, max_letters=240)
    a, b = parse_source(source), parse_final(target)
    semantics = all(any(p["semantic_relation_valid"] for p in parses) for parses in (a, b))
    trace = fringe_trace(target)
    boundary = boundary_witness(source, target, trace["actual_pairs"])
    failures = [k for k, v in checks.items() if not v]
    if not semantics:
        failures.append("independent_scalar_patient_repair_binding")
    if not boundary["same_letter_tape"]:
        failures.append("same_source_target_tape")
    return {**path, "rendered_diagnostic": text, "source_rendered_diagnostic": source_text,
            "normalized": normalize_letters(text), "letters": len(normalize_letters(text)),
            "central_admission": checks, "source_central_admission": source_checks,
            "source_parse": a, "independent_final_parse": b, "fringe_trace": trace,
            "live_boundary_witness": boundary, "failures": failures,
            "eligible_for_external_provenance": semantics and boundary["same_letter_tape"]
                and all(checks.values()) and all(source_checks.values()) and trace["termination"] == "closure",
            "external_provenance": "not_checked", "originality_claim": False, "promoted": False}


def frontier_witnesses(row, minimum_pairs=11):
    """Examine later reached frontiers too; a cut at12 needs depth>=13."""
    if minimum_pairs < 11:
        raise ValueError("eleven actual pairs are mandatory")
    text = row["normalized"]
    actual = row["fringe_trace"]["actual_pairs"]
    safe = all(v for field in ("central_admission", "source_central_admission")
               for k, v in row[field].items() if k != "exact_letter_palindrome")
    semantics = all(any(p["semantic_relation_valid"] for p in row[k]) for k in ("source_parse", "independent_final_parse"))
    result = []
    for depth in range(minimum_pairs, actual + 1):
        # A lone odd-center character is not an additional outer PAIR.
        if len(text) <= 2 * depth + 1:
            continue
        boundary = boundary_witness(row["source_tokens"], row["target_tokens"], depth)
        live = bool(boundary["same_letter_tape"] and boundary["crossed_left_disagreements"] and boundary["crossed_right_disagreements"]
                    and safe and semantics)
        result.append({"depth": depth, "prefix": text[:depth], "rendered_diagnostic": row["rendered_diagnostic"],
                       "boundary_witness": boundary, "two_live_boundary_sides": live,
                       "next_left": text[depth], "next_right": text[-depth - 1],
                       "paired_next_letter": text[depth] if live and text[depth] == text[-depth - 1] else None})
    return result


def discover(paths=None, *, minimum_pairs=11, minimum_letters=6):
    if minimum_pairs < 11 or minimum_letters < 6:
        raise ValueError("eleven actual pairs and six distinct paired next letters are mandatory")
    rows = [review(path) for path in (productions() if paths is None else paths)]
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        for witness in frontier_witnesses(row, minimum_pairs):
            grouped[(witness["depth"], witness["prefix"])].append({"derivation_index": index, **witness})
    channels, live_derivations = [], set()
    for (depth, prefix), witnesses in sorted(grouped.items()):
        paired = sorted({w["paired_next_letter"] for w in witnesses if w["paired_next_letter"]})
        live_derivations.update(w["derivation_index"] for w in witnesses if w["two_live_boundary_sides"])
        indices = {w["derivation_index"] for w in witnesses}
        channels.append({"depth": depth, "prefix": prefix, "derivation_pairs": len(indices),
                         "distinct_target_surfaces": len({tuple(rows[i]["target_tokens"]) for i in indices}),
                         "paired_next_letters": paired, "qualified": len(paired) >= minimum_letters, "witnesses": witnesses})
    mismatches = Counter((r["fringe_trace"].get("mismatch_pair"), r["fringe_trace"].get("left"), r["fringe_trace"].get("right")) for r in rows)
    closures = [r for r in rows if r["central_admission"]["exact_letter_palindrome"]]
    return {"source_target_derivation_pairs": len(rows), "distinct_target_surfaces": len({tuple(r["target_tokens"]) for r in rows}),
            "distinct_source_analyses": len({tuple(r["source_tokens"]) for r in rows}), "finite_exhausted": True,
            "actual_pair_depth_distribution": dict(sorted(Counter(r["fringe_trace"]["actual_pairs"] for r in rows).items())),
            "reached_minimum_pair_derivations": sum(r["fringe_trace"]["actual_pairs"] >= minimum_pairs for r in rows),
            "live_two_sided_boundary_derivations_at_or_beyond_minimum": len(live_derivations),
            "qualified_channels": sum(c["qualified"] for c in channels), "channels": channels,
            "outer_mismatches": [{"pair": p, "left": a, "right": b, "derivation_pairs": n}
                for (p, a, b), n in sorted(mismatches.items(), key=lambda item: (item[0][0] is None, item[0][0] or 0, str(item[0][1:])))],
            "failure_counts": dict(sorted(Counter(f for row in rows for f in row["failures"]).items())),
            "exact_closure_derivations": len(closures), "distinct_exact_closure_surfaces": len({r["rendered_diagnostic"] for r in closures}),
            "reviews": rows, "closure_reviews": closures, "full_palindrome_product_compiled": False,
            "model_interface_implemented": False, "model_queries_executed": 0, "promoted_candidates": []}


def run():
    result = discover()
    files = (Path(__file__), ROOT / "experiments/measurement_cleft_source_parser_20260913.py",
             ROOT / "experiments/measurement_cleft_final_parser_20260913.py", ROOT / "llm_palindrome/admission.py",
             ROOT / "data/known_palindromes.json")
    return {"experiment": "measurement-cleft-finite-repair-predicate-residual-v1", "status": "finite_screen_unqualified",
            "screen": result, "provenance": {"file_hashes": {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in files},
                "external_candidate_provenance_searches": 0, "external_review_required_before_promotion": True,
                "lexical_reference_reused": "https://dictionary.cambridge.org/us/dictionary/english-german/nonetheless",
                "originality_claim": False},
            "next_construction": "Use a non-cleft temporal comparison: a repair-relative patient subject followed by a nominalized scalar change and a reference to its earlier recorded state. This removes the current equative/free-relative opener and final transitive-repair-verb restriction while preserving explicit same-patient before/after binding."}


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    result = run()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], **{k: v for k, v in result["screen"].items()
                     if k not in {"reviews", "channels", "closure_reviews"}}}, indent=2))
