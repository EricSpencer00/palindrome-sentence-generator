"""Finite scalar result-state repair screen, with independent causal reparse.

Source/final lexical analyses are composed normally and compared, never
reflected.  No full palindrome product or authoring/model interface is built.
Eleven real pairs, two live boundary sides and six distinct paired next
letters are required before a channel may qualify.
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
from experiments.result_state_source_parser_20260913 import parse_source
from experiments.result_state_final_parser_20260913 import parse_final, render_final
from experiments.joint_onset_material_feasibility_20260913 import boundary_witness
from experiments.residual_constrained_attachment_infill_20260913 import fringe_trace
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


def productions():
    substrates = ((("metalwork",), ("metal", "work"), "metal", False),
                  (("woodwork",), ("wood", "work"), "wood", False),
                  (("glasswork",), ("glass", "work"), "glass", False),
                  (("oak", "panels"), ("oak", "panels"), "wood", True))
    relations = (
        (("rough",), False, "polished", ("fine", "abrasives"),
         ((("smoother",), ("smoother",)), (("smoother", "than", "before"), ("smoother", "than", "before")))),
        (("leaky",), False, "sealed", ("fresh", "adhesive"),
         ((("airtight",), ("air", "tight")), (("less", "permeable"), ("less", "permeable")))),
        (("loose", "joints", "in"), True, "tightened", ("steel", "screws"),
         ((("firmer",), ("firmer",)), (("more", "secure"), ("more", "secure")))),
        (("reworked", "ridges", "on"), True, "sanded", ("coarse", "abrasives"),
         ((("lower",), ("lower",)), (("lower", "than", "before"), ("lower", "than", "before")))),
    )
    for substrate, relation in product(substrates, relations):
        a, b, material, plural_material = substrate
        initial, plural_feature, repair, means, results = relation
        if repair == "tightened" and material == "glass":
            continue
        copula = "are" if plural_feature or plural_material else "is"
        middle = (("that", "arrived", "from", "regional", "museums", repair, "with")
                  + means + ("under", "steady", "pressure", copula, "now"))
        for source_end, target_end in results:
            yield {"source_tokens": initial + a + middle + source_end,
                   "target_tokens": initial + b + middle + target_end,
                   "source_relation": {"initial_property_tokens": initial, "repair": repair,
                                       "material": material, "same_controlled_patient": True}}


def review(path):
    source, target = tuple(path["source_tokens"]), tuple(path["target_tokens"])
    rendered = render_final(target)
    source_rendered = " ".join(source).capitalize() + "."
    source_checks = mechanical_admission_checks(source_rendered, min_letters=100, max_letters=240)
    checks = mechanical_admission_checks(rendered, min_letters=100, max_letters=240)
    a, b = parse_source(source), parse_final(target)
    semantics = all(any(p["semantic_relation_valid"] for p in parses) for parses in (a, b))
    trace = fringe_trace(target)
    boundary = boundary_witness(source, target, trace["actual_pairs"])
    failures = [k for k, v in checks.items() if not v]
    if not semantics:
        failures.append("independent_causal_semantics")
    if not boundary["same_letter_tape"]:
        failures.append("same_source_target_tape")
    return {**path, "rendered_diagnostic": rendered, "source_rendered_diagnostic": source_rendered,
            "normalized": normalize_letters(rendered), "letters": len(normalize_letters(rendered)),
            "source_central_admission": source_checks, "central_admission": checks,
            "source_parse": a, "independent_final_parse": b, "fringe_trace": trace,
            "live_boundary_witness": boundary, "failures": failures,
            "eligible_for_external_provenance": semantics and boundary["same_letter_tape"]
                and all(checks.values()) and all(source_checks.values()) and trace["termination"] == "closure",
            "external_provenance": "not_checked", "originality_claim": False, "promoted": False}


def discover(paths=None, *, minimum_pairs=11, minimum_letters=6):
    if minimum_pairs < 11 or minimum_letters < 6:
        raise ValueError("eleven real pairs and six distinct paired next letters may not be relaxed")
    rows = [review(p) for p in (productions() if paths is None else paths)]
    grouped = defaultdict(list)
    for row in rows:
        if row["fringe_trace"]["actual_pairs"] >= minimum_pairs:
            grouped[row["normalized"][:minimum_pairs]].append(row)
    channels = []
    for tape, group in sorted(grouped.items()):
        witnesses = []
        for row in group:
            text = row["normalized"]
            b = boundary_witness(row["source_tokens"], row["target_tokens"], minimum_pairs)
            central_safe = all(v for checkset in ("source_central_admission", "central_admission")
                               for k, v in row[checkset].items() if k != "exact_letter_palindrome")
            semantic_safe = all(any(p["semantic_relation_valid"] for p in row[k]) for k in ("source_parse", "independent_final_parse"))
            live = bool(b["same_letter_tape"] and b["crossed_left_disagreements"] and b["crossed_right_disagreements"]
                        and central_safe and semantic_safe and len(text) > 2 * minimum_pairs)
            next_letter = text[minimum_pairs] if live and text[minimum_pairs] == text[-minimum_pairs - 1] else None
            witnesses.append({"rendered_diagnostic": row["rendered_diagnostic"], "actual_pairs": row["fringe_trace"]["actual_pairs"],
                              "boundary_witness": b, "two_live_boundary_sides": live, "paired_next_letter": next_letter})
        letters = sorted({w["paired_next_letter"] for w in witnesses if w["paired_next_letter"]})
        channels.append({"prefix": tape, "source_target_derivation_pairs": len(group),
                         "distinct_target_surfaces": len({tuple(r["target_tokens"]) for r in group}),
                         "paired_next_letters": letters, "qualified": len(letters) >= minimum_letters, "witnesses": witnesses})
    counts = Counter((r["fringe_trace"].get("mismatch_pair"), r["fringe_trace"].get("left"), r["fringe_trace"].get("right")) for r in rows)
    closures = [r for r in rows if r["central_admission"]["exact_letter_palindrome"]]
    return {"source_target_derivation_pairs": len(rows), "distinct_target_surfaces": len({tuple(r["target_tokens"]) for r in rows}),
            "distinct_source_analyses": len({tuple(r["source_tokens"]) for r in rows}), "finite_exhausted": True,
            "actual_pair_depth_distribution": dict(sorted(Counter(r["fringe_trace"]["actual_pairs"] for r in rows).items())),
            "reached_eleven_pair_derivations": sum(r["fringe_trace"]["actual_pairs"] >= minimum_pairs for r in rows),
            "live_two_sided_boundary_derivations_at_eleven_pairs": sum(w["two_live_boundary_sides"] for c in channels for w in c["witnesses"]),
            "qualified_channels": sum(c["qualified"] for c in channels), "channels": channels,
            "outer_mismatches": [{"pair": p, "left": a, "right": b, "derivation_pairs": n}
                for (p, a, b), n in sorted(counts.items(), key=lambda item: (item[0][0] is None, item[0][0] or 0, str(item[0][1:])))],
            "failure_counts": dict(sorted(Counter(f for r in rows for f in r["failures"]).items())),
            "exact_closure_derivations": len(closures), "distinct_exact_closure_surfaces": len({r["rendered_diagnostic"] for r in closures}),
            "reviews": rows, "closure_reviews": closures, "full_palindrome_product_compiled": False,
            "authoring_interface_implemented": False, "model_queries_executed": 0, "promoted_candidates": []}


def run():
    result = discover()
    files = (Path(__file__), ROOT / "experiments/result_state_source_parser_20260913.py",
             ROOT / "experiments/result_state_final_parser_20260913.py", ROOT / "llm_palindrome/admission.py",
             ROOT / "data/known_palindromes.json")
    return {"experiment": "agentless-material-result-state-residual-v1", "status": "finite_screen_unqualified",
            "screen": result, "provenance": {"file_hashes": {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in files},
                "external_searches_executed": 0, "external_review_required_before_promotion": True, "originality_claim": False},
            "next_construction": "Use comparative result-fronting with a postponed material subject, so the initial scalar predicate and final material noun phrase are jointly generated. This changes which constituents occupy the endpoints instead of adding more repair participles or comparative synonyms to this exhausted subject-first result grammar."}


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    result = run()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], **{k: v for k, v in result["screen"].items()
                     if k not in {"reviews", "channels", "closure_reviews"}}}, indent=2))
