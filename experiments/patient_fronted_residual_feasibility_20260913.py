"""Joint patient/locative-first repair and final applicator feasibility screen.

No reflected tape, fixed center, full palindrome product, model client, or
authoring interface.  Qualification requires eleven matched outer pairs,
two actually crossed boundary differences (one at each side), and six
distinct next characters that match on both fronts in the same channel.
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
from experiments.patient_fronted_source_parser_20260913 import parse_source
from experiments.patient_fronted_final_parser_20260913 import parse_final, render_final
from experiments.joint_onset_material_feasibility_20260913 import boundary_witness
from experiments.residual_constrained_attachment_infill_20260913 import fringe_trace
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


def productions():
    # All bodies are selected by patient type before their full endpoints are
    # screened.  No opening/tool phrase pair was selected by reversing letters.
    patients = (
        (("red", "metalwork"), ("red", "metal", "work"), False, "cracks", ("epoxy",)),
        (("salvaged", "woodwork"), ("salvaged", "wood", "work"), False, "cracks", ("glue",)),
        (("red", "acrylic", "panels"), ("red", "acrylic", "panels"), True, "cracks", ("cement",)),
        (("stone", "sculptures"), ("stone", "sculptures"), True, "cracks", ("epoxy",)),
        (("torn", "canvas", "paintings"), ("torn", "canvas", "paintings"), True, "tears", ("starch", "paste")),
        (("damaged", "paper", "prints"), ("damaged", "paper", "prints"), True, "tears", ("starch", "paste")),
        (("old", "glasswork"), ("old", "glass", "work"), False, "cracks", ("epoxy",)),
    )
    tools = ((("paintbrushes",), ("paint", "brushes")), (("a", "roller"), ("a", "roller")),
             (("a", "spatula"), ("a", "spatula")), (("a", "spreader"), ("a", "spreader")),
             (("a", "syringe"), ("a", "syringe")))
    for patient, tool, agent, order in product(patients, tools, ("artists", "conservators", "technicians"), ("passive", "locative")):
        source_np, target_np, plural, defect, medium = patient
        pronoun, possessive, copula = ("them", "their", "are") if plural else ("it", "its", "is")
        active, passive = ("mend", "mended") if defect == "tears" else ("repair", "repaired")
        if order == "passive":
            common = (("that", "arrived", "from", "regional", "museums", copula, passive, "along", possessive, defect,
                       "with") + medium + ("applied", "to", pronoun, "by", "local", agent, "using"))
            source, target = source_np + common + tool[0], target_np + common + tool[1]
        else:
            prefix = ("along", "the", defect, "in")
            common = (("from", "regional", "museums", "local", agent, active, pronoun, "with") + medium + ("using",))
            source, target = prefix + source_np + common + tool[0], prefix + target_np + common + tool[1]
        yield {"source_tokens": source, "target_tokens": target,
               "source_construction": order, "joint_semantics": {"defect": defect, "medium": medium,
                    "tool_role": "adhesive_applicator", "patient_number": "plural" if plural else "singular"}}


def review(path):
    source, target = tuple(path["source_tokens"]), tuple(path["target_tokens"])
    rendered = render_final(target)
    source_rendered = " ".join(source).capitalize() + "."
    a = mechanical_admission_checks(source_rendered, min_letters=100, max_letters=240)
    b = mechanical_admission_checks(rendered, min_letters=100, max_letters=240)
    source_parses, target_parses = parse_source(source), parse_final(target)
    semantics = all(any(p["semantic_relation_valid"] for p in analyses) for analyses in (source_parses, target_parses))
    trace = fringe_trace(target)
    changes = boundary_witness(source, target, trace["actual_pairs"])
    failures = [k for k, v in b.items() if not v]
    if not semantics:
        failures.append("independent_source_and_final_semantics")
    if not changes["same_letter_tape"]:
        failures.append("same_source_target_tape")
    return {**path, "rendered_diagnostic": rendered, "source_rendered_diagnostic": source_rendered,
            "normalized": normalize_letters(rendered), "letters": len(normalize_letters(rendered)),
            "source_central_admission": a, "central_admission": b,
            "source_parse": source_parses, "independent_final_parse": target_parses,
            "fringe_trace": trace, "live_boundary_witness": changes, "failures": failures,
            "eligible_for_external_provenance": semantics and changes["same_letter_tape"]
                and all(a.values()) and all(b.values()) and trace["termination"] == "closure",
            "external_provenance": "not_checked", "originality_claim": False, "promoted": False}


def discover(paths=None, *, minimum_pairs=11, minimum_letters=6):
    if minimum_pairs < 11 or minimum_letters < 6:
        raise ValueError("eleven actual pairs and six distinct paired next letters are mandatory")
    rows = [review(path) for path in (productions() if paths is None else paths)]
    groups = defaultdict(list)
    for row in rows:
        if row["fringe_trace"]["actual_pairs"] >= minimum_pairs:
            groups[row["normalized"][:minimum_pairs]].append(row)
    channels = []
    for prefix, members in sorted(groups.items()):
        witnesses = []
        for row in members:
            tape = row["normalized"]
            b = boundary_witness(row["source_tokens"], row["target_tokens"], minimum_pairs)
            central_safe = all(v for name in ("central_admission", "source_central_admission")
                               for k, v in row[name].items() if k != "exact_letter_palindrome")
            semantic_safe = all(any(p["semantic_relation_valid"] for p in row[k]) for k in ("source_parse", "independent_final_parse"))
            live = (b["same_letter_tape"] and b["crossed_left_disagreements"] and b["crossed_right_disagreements"]
                    and central_safe and semantic_safe and len(tape) > 2 * minimum_pairs)
            paired = tape[minimum_pairs] if live and tape[minimum_pairs] == tape[-minimum_pairs - 1] else None
            witnesses.append({"rendered_diagnostic": row["rendered_diagnostic"], "actual_pairs": row["fringe_trace"]["actual_pairs"],
                              "boundary_witness": b, "two_live_boundary_sides": bool(live), "paired_next_letter": paired})
        next_letters = sorted({w["paired_next_letter"] for w in witnesses if w["paired_next_letter"]})
        channels.append({"prefix": prefix, "derivation_pairs": len(members),
                         "distinct_target_surfaces": len({tuple(r["target_tokens"]) for r in members}),
                         "paired_next_letters": next_letters, "qualified": len(next_letters) >= minimum_letters,
                         "witnesses": witnesses})
    mismatches = Counter((r["fringe_trace"].get("mismatch_pair"), r["fringe_trace"].get("left"),
                          r["fringe_trace"].get("right")) for r in rows)
    closures = [r for r in rows if r["central_admission"]["exact_letter_palindrome"]]
    return {"source_target_derivation_pairs": len(rows),
            "distinct_target_surfaces": len({tuple(r["target_tokens"]) for r in rows}),
            "distinct_source_analyses": len({tuple(r["source_tokens"]) for r in rows}),
            "finite_exhausted": True,
            "actual_pair_depth_distribution": dict(sorted(Counter(r["fringe_trace"]["actual_pairs"] for r in rows).items())),
            "reached_eleven_pair_derivations": sum(r["fringe_trace"]["actual_pairs"] >= minimum_pairs for r in rows),
            "live_two_sided_boundary_derivations_at_eleven_pairs": sum(w["two_live_boundary_sides"] for c in channels for w in c["witnesses"]),
            "qualified_channels": sum(c["qualified"] for c in channels), "channels": channels,
            "outer_mismatches": [{"pair": p, "left": a, "right": b, "derivation_pairs": n}
                for (p, a, b), n in sorted(mismatches.items(), key=lambda item: (item[0][0] is None, item[0][0] or 0, str(item[0][1:])) )],
            "failure_counts": dict(sorted(Counter(f for row in rows for f in row["failures"]).items())),
            "exact_closure_derivations": len(closures),
            "distinct_exact_closure_surfaces": len({r["rendered_diagnostic"] for r in closures}),
            "closure_reviews": closures, "reviews": rows, "full_palindrome_product_compiled": False,
            "authoring_interface_implemented": False, "model_queries_executed": 0, "promoted_candidates": []}


def run():
    result = discover()
    files = (Path(__file__), ROOT / "experiments/patient_fronted_source_parser_20260913.py",
             ROOT / "experiments/patient_fronted_final_parser_20260913.py", ROOT / "llm_palindrome/admission.py",
             ROOT / "data/known_palindromes.json")
    return {"experiment": "patient-locative-fronted-repair-residual-v1", "status": "finite_screen_unqualified",
            "screen": result, "provenance": {"files": {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in files},
                "external_searches_executed": 0, "external_review_required_before_promotion": True, "originality_claim": False},
            "next_construction": "Use an independently typed agentless result-state/participial construction that places the repaired material property at the beginning and its measured state at the end. This removes the applicator-noun suffix bottleneck; another inventory of tools inside the current passive/locative frames cannot supply eleven live pairs."}


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    result = run()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], **{k: v for k, v in result["screen"].items()
                     if k not in {"reviews", "channels", "closure_reviews"}}}, indent=2))
