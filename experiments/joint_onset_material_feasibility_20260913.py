"""Joint semantic onset/material-choice finite residual screen, no full product.

Each source analysis and final analysis is lexically composed from typed role
choices, never a reflected source tape.  Compounds permit real token-cut
differences; orthographic changes never count unless matching outer traversal
actually crosses them.  Qualification requires BOTH boundary sides crossed,
ten real pairs and six DISTINCT compatible next letters in one channel.
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
from experiments.joint_material_repair_validator_20260913 import parse_sentence, render_sentence
from experiments.residual_constrained_attachment_infill_20260913 import fringe_trace
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


def productions():
    actors = ((("traders",), ("traders",)), (("conservators",), ("conservators",)),
              (("metalworkers",), ("metal", "workers")), (("woodworkers",), ("wood", "workers")))
    openings = (("sort", ("sort",), {"metal", "wood", "fabric"}),
                ("solder", ("solder", "seams", "in"), {"metal"}),
                ("seal", ("seal", "pores", "in"), {"wood", "fabric"}),
                ("brush", ("brush", "dust", "from"), {"metal", "wood", "fabric"}))
    materials = (("metal", ("metal", "panels")), ("wood", ("oak", "panels")),
                 ("fabric", ("canvas", "paintings")))
    finishes = (("high_gloss", ("high", "gloss", "red"), ("high", "gloss", "red"), {"metal", "wood", "fabric"}),
                ("matte", ("matte", "red"), ("matte", "red"), {"metal", "wood", "fabric"}),
                ("polished_hardwood", ("polished", "red", "hardwood"), ("polished", "red", "hard", "wood"), {"wood"}))
    owners = ("artists", "collectors", "patrons")
    for actor, opening, material, finish, owner in product(actors, openings, materials, finishes, owners):
        kind, patient = material
        if kind not in opening[2] or kind not in finish[3]:
            continue
        repair = ("mend", "tears") if kind == "fabric" else ("repair", "cracks")
        middle = (("local", owner, "damaged") + patient
                  + ("that", "arrived", "from", "regional", "museums", "and") + repair + ("in", "this"))
        a = actor[0] + opening[1] + middle + finish[1] + ("art",)
        b = actor[1] + opening[1] + middle + finish[2] + ("art",)
        yield {"source_tokens": a, "target_tokens": b,
               "source_semantics": {"first_action": opening[0], "material": kind, "finish": finish[0],
                                    "owner": owner, "both_events_and_finish_bearer": "owned_artwork"}}


def cuts(words):
    result, depth = set(), 0
    for word in words[:-1]:
        depth += len(word)
        result.add(depth)
    return result


def boundary_witness(source, target, actual_pairs):
    a, b = "".join(source), "".join(target)
    differences = cuts(source) ^ cuts(target)
    return {"same_letter_tape": a == b, "source_word_cuts": sorted(cuts(source)),
            "target_word_cuts": sorted(cuts(target)),
            "all_boundary_disagreements": sorted(differences),
            "crossed_left_disagreements": sorted(d for d in differences if 0 < d < actual_pairs),
            "crossed_right_disagreements": sorted(len(b) - d for d in differences if 0 < len(b) - d < actual_pairs)}


def review(production):
    source, target = tuple(production["source_tokens"]), tuple(production["target_tokens"])
    rendered = render_sentence(target)
    source_rendered = render_sentence(source)
    central = mechanical_admission_checks(rendered, min_letters=100, max_letters=240)
    source_central = mechanical_admission_checks(source_rendered, min_letters=100, max_letters=240)
    source_parses, target_parses = parse_sentence(source), parse_sentence(target)
    semantics = all(any(p["semantic_relation_valid"] for p in parses) for parses in (source_parses, target_parses))
    trace = fringe_trace(target)
    witness = boundary_witness(source, target, trace["actual_pairs"])
    failures = [name for name, value in central.items() if not value]
    if not semantics:
        failures.append("independent_source_and_target_semantics")
    if not witness["same_letter_tape"]:
        failures.append("source_target_same_tape")
    return {**production, "rendered_diagnostic": rendered, "source_rendered_diagnostic": source_rendered,
            "normalized": normalize_letters(rendered), "letters": len(normalize_letters(rendered)),
            "central_admission": central, "source_central_admission": source_central,
            "source_independent_parses": source_parses, "target_independent_parses": target_parses,
            "fringe_trace": trace, "boundary_witness": witness, "failures": failures,
            "eligible_for_external_provenance": semantics and witness["same_letter_tape"]
                and all(central.values()) and all(source_central.values()) and trace["termination"] == "closure",
            "external_provenance": "not_checked", "originality_claim": False, "promoted": False}


def discover(paths=None, *, pairs=10, letters=6):
    if pairs < 10 or letters < 6:
        raise ValueError("ten real pairs and six distinct paired next letters are mandatory")
    rows = [review(p) for p in (productions() if paths is None else paths)]
    groups = defaultdict(list)
    for row in rows:
        if row["fringe_trace"]["actual_pairs"] >= pairs:
            groups[row["normalized"][:pairs]].append(row)
    channels = []
    for tape, members in sorted(groups.items()):
        witnesses = []
        for row in members:
            text = row["normalized"]
            geometric = boundary_witness(row["source_tokens"], row["target_tokens"], pairs)
            semantic = all(any(p["semantic_relation_valid"] for p in row[k])
                           for k in ("source_independent_parses", "target_independent_parses"))
            # Neither catalogue material nor an unlexical source analysis can
            # count as a production crossing witness.
            safe = all(v for k, v in row["central_admission"].items() if k != "exact_letter_palindrome")
            source_safe = all(v for k, v in row["source_central_admission"].items() if k != "exact_letter_palindrome")
            live = len(text) > 2 * pairs
            next_left = text[pairs] if live else None
            next_right = text[-pairs - 1] if live else None
            two_sided = bool(geometric["crossed_left_disagreements"] and geometric["crossed_right_disagreements"])
            witnesses.append({"rendered_diagnostic": row["rendered_diagnostic"], "actual_pairs": row["fringe_trace"]["actual_pairs"],
                              "boundary_analysis": geometric, "two_sided_crossings_at_probe": two_sided,
                              "safe_independent_semantic_analysis": semantic and safe and source_safe and geometric["same_letter_tape"],
                              "next_left": next_left, "next_right": next_right,
                              "paired_next": next_left if next_left == next_right and live else None})
        valid = [w for w in witnesses if w["two_sided_crossings_at_probe"] and w["safe_independent_semantic_analysis"]]
        paired = sorted({w["paired_next"] for w in valid if w["paired_next"]})
        channels.append({"outer_tape": tape, "actual_pairs_required": pairs,
                         "source_target_derivation_pairs": len(members),
                         "distinct_rendered_target_surfaces": len({tuple(r["target_tokens"]) for r in members}),
                         "two_sided_boundary_eligible_derivations": len(valid),
                         "paired_next_letters": paired,
                         "unfiltered_paired_next_letters": sorted({w["paired_next"] for w in witnesses if w["paired_next"]}),
                         "qualified": len(paired) >= letters, "witnesses": witnesses})
    mismatches = Counter((r["fringe_trace"].get("mismatch_pair"), r["fringe_trace"].get("left"),
                          r["fringe_trace"].get("right")) for r in rows)
    closure_rows = [r for r in rows if r["central_admission"]["exact_letter_palindrome"]]
    return {"source_target_derivation_pairs": len(rows),
            "distinct_rendered_target_surfaces": len({tuple(r["target_tokens"]) for r in rows}),
            "distinct_source_analyses": len({tuple(r["source_tokens"]) for r in rows}),
            "finite_exhausted": True,
            "actual_pair_depth_distribution": dict(sorted(Counter(r["fringe_trace"]["actual_pairs"] for r in rows).items())),
            "raw_two_sided_boundary_geometry_pairs": sum(bool(r["boundary_witness"]["all_boundary_disagreements"])
                and bool([d for d in r["boundary_witness"]["all_boundary_disagreements"] if d < len(r["normalized"]) // 2])
                and bool([d for d in r["boundary_witness"]["all_boundary_disagreements"] if d > len(r["normalized"]) // 2]) for r in rows),
            "production_two_sided_crossings_after_ten_pairs": sum(c["two_sided_boundary_eligible_derivations"] for c in channels),
            "qualified_channels": sum(c["qualified"] for c in channels), "channels": channels,
            "outer_mismatches": [{"pair": p, "left": a, "right": b, "derivation_pairs": n}
                                 for (p, a, b), n in sorted(mismatches.items(), key=lambda item: str(item[0]))],
            "failure_counts": dict(sorted(Counter(f for r in rows for f in r["failures"]).items())),
            "exact_closure_derivations": len(closure_rows),
            "distinct_exact_closure_surfaces": len({r["rendered_diagnostic"] for r in closure_rows}),
            "closure_reviews": closure_rows, "reviews": rows,
            "full_palindrome_product_compiled": False, "model_transport_implemented": False,
            "model_queries_executed": 0, "promoted_candidates": []}


def run():
    result = discover()
    files = (Path(__file__), ROOT / "experiments/joint_material_repair_validator_20260913.py",
             ROOT / "llm_palindrome/admission.py", ROOT / "data/known_palindromes.json")
    return {"experiment": "joint-clause-onset-final-material-property-residual-v1",
            "status": "finite_screen_unqualified", "screen": result,
            "provenance": {"local_hashes": {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in files},
                           "external_searches_executed": 0, "external_review_required_before_promotion": True,
                           "known_catalogue_used_only_as_negative_filter": True, "originality_claim": False},
            "next_construction": "Use a typed locative/patient-fronted repair clause, jointly generating its initial material noun phrase and final agent/tool attachment. The current actor-first/glossy-art channel exhausts at pair11 (d/g), and its compound boundary alternatives never reach that channel. Merely adding ownership or coating synonyms cannot establish the missing two-sided ten-pair continuation."}


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    result = run()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], **{k: v for k, v in result["screen"].items()
                     if k not in {"reviews", "channels", "closure_reviews"}}}, indent=2))
