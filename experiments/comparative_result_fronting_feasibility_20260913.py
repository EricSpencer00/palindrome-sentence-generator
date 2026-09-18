"""Comparative result-fronting with a postponed physical material subject.

Finite endpoint feasibility only.  Both grammars separately enforce the same
patient and ordered scalar repair effect.  Dictionary-supported concessive
spellings and material compounds supply alternative token analyses, not a
palindrome seed.  They do not count without an actual eleven-pair traversal.
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
from experiments.comparative_fronting_source_parser_20260913 import parse_source
from experiments.comparative_fronting_final_parser_20260913 import parse_final, render_final
from experiments.joint_onset_material_feasibility_20260913 import boundary_witness
from experiments.residual_constrained_attachment_infill_20260913 import fringe_trace
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


def productions():
    materials = ((("metalwork",), ("metal", "work"), "metal", False),
                 (("woodwork",), ("wood", "work"), "wood", False),
                 (("glasswork",), ("glass", "work"), "glass", False),
                 (("oak", "panels"), ("oak", "panels"), "wood", True),
                 (("wooden", "vessel"), ("wooden", "vessel"), "wood", False))
    repairs = (("polishing", ("fine", "abrasives"), ("rough",), False, (("smoother",),)),
               ("sealing", ("fresh", "adhesive"), ("leaky",), False, (("less", "permeable"),)),
               ("tightening", ("steel", "screws"), ("loose", "joints", "in"), True, (("firmer",), ("more", "secure"))),
               ("sanding", ("coarse", "abrasives"), ("reworked", "ridges", "on"), True, (("lower",),)))
    discourse = (((), ()), (("nonetheless",), ("none", "the", "less")))
    for material, repair, discourse_pair in product(materials, repairs, discourse):
        source_material, target_material, kind, plural_material = material
        operation, means, initial, plural_feature, measurements = repair
        if operation == "tightening" and kind == "glass":
            continue
        number = "are" if plural_material or plural_feature else "is"
        for comparison in measurements:
            middle = (comparison + ("than", "before", "after", operation, "with") + means
                      + ("under", "steady", "pressure", "for", "several", "hours", number, "the") + initial)
            yield {"source_tokens": discourse_pair[0] + middle + source_material,
                   "target_tokens": discourse_pair[1] + middle + target_material,
                   "source_relation": {"comparison": comparison, "repair": operation, "material": kind,
                                       "construction": "fronted_result_postposed_material", "concessive": bool(discourse_pair[0])}}


def review(path):
    a, b = tuple(path["source_tokens"]), tuple(path["target_tokens"])
    source_text, text = " ".join(a).capitalize() + ".", render_final(b)
    source_checks = mechanical_admission_checks(source_text, min_letters=100, max_letters=240)
    checks = mechanical_admission_checks(text, min_letters=100, max_letters=240)
    source_parse, final_parse = parse_source(a), parse_final(b)
    semantics = all(any(p["semantic_relation_valid"] for p in parses) for parses in (source_parse, final_parse))
    trace = fringe_trace(b)
    boundary = boundary_witness(a, b, trace["actual_pairs"])
    failures = [k for k, v in checks.items() if not v]
    if not semantics:
        failures.append("independent_ordered_scalar_semantics")
    if not boundary["same_letter_tape"]:
        failures.append("same_source_target_tape")
    return {**path, "rendered_diagnostic": text, "source_rendered_diagnostic": source_text,
            "normalized": normalize_letters(text), "letters": len(normalize_letters(text)),
            "source_central_admission": source_checks, "central_admission": checks,
            "source_parse": source_parse, "independent_final_parse": final_parse,
            "fringe_trace": trace, "live_boundary_witness": boundary, "failures": failures,
            "eligible_for_external_provenance": semantics and boundary["same_letter_tape"]
                and all(checks.values()) and all(source_checks.values()) and trace["termination"] == "closure",
            "external_provenance": "not_checked", "originality_claim": False, "promoted": False}


def discover(paths=None, *, minimum_pairs=11, minimum_letters=6):
    if minimum_pairs < 11 or minimum_letters < 6:
        raise ValueError("eleven actual pairs and six distinct paired next letters are mandatory")
    rows = [review(path) for path in (productions() if paths is None else paths)]
    grouped = defaultdict(list)
    for row in rows:
        if row["fringe_trace"]["actual_pairs"] >= minimum_pairs:
            grouped[row["normalized"][:minimum_pairs]].append(row)
    channels = []
    for prefix, group in sorted(grouped.items()):
        witnesses = []
        for row in group:
            tape = row["normalized"]
            boundary = boundary_witness(row["source_tokens"], row["target_tokens"], minimum_pairs)
            safe = all(v for field in ("source_central_admission", "central_admission")
                       for k, v in row[field].items() if k != "exact_letter_palindrome")
            semantics = all(any(p["semantic_relation_valid"] for p in row[k]) for k in ("source_parse", "independent_final_parse"))
            live = bool(boundary["same_letter_tape"] and boundary["crossed_left_disagreements"] and boundary["crossed_right_disagreements"]
                        and safe and semantics and len(tape) > 2 * minimum_pairs)
            next_letter = tape[minimum_pairs] if live and tape[minimum_pairs] == tape[-minimum_pairs - 1] else None
            witnesses.append({"rendered_diagnostic": row["rendered_diagnostic"], "actual_pairs": row["fringe_trace"]["actual_pairs"],
                              "boundary_witness": boundary, "two_live_boundary_sides": live, "paired_next_letter": next_letter})
        paired = sorted({w["paired_next_letter"] for w in witnesses if w["paired_next_letter"]})
        channels.append({"prefix": prefix, "derivation_pairs": len(group),
                         "distinct_target_surfaces": len({tuple(r["target_tokens"]) for r in group}),
                         "paired_next_letters": paired, "qualified": len(paired) >= minimum_letters, "witnesses": witnesses})
    mismatch = Counter((r["fringe_trace"].get("mismatch_pair"), r["fringe_trace"].get("left"), r["fringe_trace"].get("right")) for r in rows)
    closures = [r for r in rows if r["central_admission"]["exact_letter_palindrome"]]
    return {"source_target_derivation_pairs": len(rows), "distinct_target_surfaces": len({tuple(r["target_tokens"]) for r in rows}),
            "distinct_source_analyses": len({tuple(r["source_tokens"]) for r in rows}), "finite_exhausted": True,
            "actual_pair_depth_distribution": dict(sorted(Counter(r["fringe_trace"]["actual_pairs"] for r in rows).items())),
            "reached_eleven_pair_derivations": sum(r["fringe_trace"]["actual_pairs"] >= minimum_pairs for r in rows),
            "live_two_sided_boundary_derivations_at_eleven_pairs": sum(w["two_live_boundary_sides"] for c in channels for w in c["witnesses"]),
            "qualified_channels": sum(c["qualified"] for c in channels), "channels": channels,
            "outer_mismatches": [{"pair": p, "left": a, "right": b, "derivation_pairs": n}
                for (p, a, b), n in sorted(mismatch.items(), key=lambda item: (item[0][0] is None, item[0][0] or 0, str(item[0][1:])))],
            "failure_counts": dict(sorted(Counter(f for row in rows for f in row["failures"]).items())),
            "exact_closure_derivations": len(closures), "distinct_exact_closure_surfaces": len({r["rendered_diagnostic"] for r in closures}),
            "reviews": rows, "closure_reviews": closures, "full_palindrome_product_compiled": False,
            "transport_implemented": False, "model_queries_executed": 0, "promoted_candidates": []}


def run():
    result = discover()
    files = (Path(__file__), ROOT / "experiments/comparative_fronting_source_parser_20260913.py",
             ROOT / "experiments/comparative_fronting_final_parser_20260913.py", ROOT / "llm_palindrome/admission.py",
             ROOT / "data/known_palindromes.json")
    return {"experiment": "comparative-result-fronting-postposed-material-residual-v1", "status": "finite_screen_unqualified",
            "screen": result, "provenance": {"file_hashes": {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in files},
                "external_candidate_provenance_searches": 0, "external_review_required_before_promotion": True,
                "lexical_validation_requests": 1,
                "lexical_validation": {"term": "nonetheless / none the less", "purpose": "spelling alternative only, not a palindrome source",
                    "source": "https://dictionary.cambridge.org/us/dictionary/english-german/nonetheless", "accessed": "2026-09-13"},
                "originality_claim": False},
            "next_construction": "Use an equative or comparative measurement cleft with the scalar property introduced inside a relative clause and a final finite repair predicate. This relocates both the initial comparative-adjective and final material-noun bottlenecks instead of adding synonyms to the exhausted inversion."}


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    result = run()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], **{k: v for k, v in result["screen"].items()
                     if k not in {"reviews", "channels", "closure_reviews"}}}, indent=2))
