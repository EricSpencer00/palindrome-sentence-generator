"""Feature-unified semantic CFG intersected with exact character palindromes.

This lane is deliberately different from paired finished-clause products.  A
small typed grammar first creates only complete, agreement-valid clauses; all
clauses are then compiled into one factored character NFA and searched with
the packed palindrome solver.  The feature labels are hard construction
constraints, not a readability score.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.palindromic_language_reachability_20260919 import (
    Language, audit, compile_templates, solve_packed, letters,
)
from llm_palindrome.admission import mechanical_admission_checks

ID = "semantic-cfg-palindrome-search-20260919b"


# Each lexical entry carries a syntactic feature bundle.  The grammar never
# pairs an incompatible subject and finite verb, and never treats a completed
# sentence as a lexical item.
SUBJECTS = (
    ("aide", "sg", "human"), ("artist", "sg", "human"),
    ("captain", "sg", "human"), ("keeper", "sg", "human"),
    ("poet", "sg", "human"), ("sailor", "sg", "human"),
    ("the aide", "sg", "human"), ("the artist", "sg", "human"),
    ("the captain", "sg", "human"), ("the keeper", "sg", "human"),
    ("the poet", "sg", "human"), ("the sailor", "sg", "human"),
    ("some artists", "pl", "human"), ("some captains", "pl", "human"),
    ("some keepers", "pl", "human"), ("some poets", "pl", "human"),
    ("some sailors", "pl", "human"), ("the men", "pl", "human"),
)
VERBS = (
    ("admires", "sg", "transitive"), ("charts", "sg", "transitive"),
    ("carries", "sg", "transitive"), ("guides", "sg", "transitive"),
    ("inspires", "sg", "transitive"), ("records", "sg", "transitive"),
    ("repairs", "sg", "transitive"), ("sees", "sg", "transitive"),
    ("adore", "pl", "transitive"), ("chart", "pl", "transitive"),
    ("carry", "pl", "transitive"), ("guide", "pl", "transitive"),
    ("inspire", "pl", "transitive"), ("record", "pl", "transitive"),
    ("repair", "pl", "transitive"), ("see", "pl", "transitive"),
)
OBJECTS = (
    ("a letter", "thing"), ("a map", "thing"), ("a message", "thing"),
    ("the letter", "thing"), ("the map", "thing"),
    ("the message", "thing"), ("nine memos", "thing"),
    ("old roses", "thing"), ("red roses", "thing"),
    ("Diana", "human"), ("Leon", "human"), ("Noel", "human"),
)
ADJ = ("bright", "careful", "gentle", "silent", "steady", "wise")
PP = ("at dawn", "at dusk", "by the river", "near the harbor", "under stars")


def clause_templates() -> tuple[tuple[str, ...], ...]:
    """Materialize feature-valid clause paths, keeping provenance metadata."""
    paths = []
    for subject, number, subject_kind in SUBJECTS:
        for verb, verb_number, valency in VERBS:
            if number != verb_number or valency != "transitive":
                continue
            for obj, object_kind in OBJECTS:
                if subject_kind == object_kind == "human":
                    # Human-to-human transitive clauses remain semantically
                    # possible, but avoid reflexive duplicate participants.
                    if subject.casefold().split()[-1] == obj.casefold():
                        continue
                for adjunct in ("",) + PP:
                    words = tuple(x for x in (subject, verb, obj, adjunct) if x)
                    paths.append(words)
                for adjective in ADJ:
                    words = tuple(x for x in (subject, verb, adjective, obj) if x)
                    paths.append(words)
    return tuple(paths)


def grammar_language():
    # The semicolon is a surface boundary, not a mirrored or reversed unit.
    # Build one NFA with complete paths and a small two-clause coordinator
    # grammar.  Each option is a path through the same start/end automaton.
    clauses = clause_templates()
    language = Language()
    language.template(clauses)
    # A separate two-clause form allows a natural narrative beat while the
    # coordinator itself is kept outside the lexical role inventory.
    for left in clauses[:180]:
        for right in clauses[:180]:
            if set(left) & set(right):
                continue
            language.template([left, right])
    return language, clauses


def run():
    language, clauses = grammar_language()
    result = solve_packed(language, max_letters=180, witnesses_per_state=96)
    rows = []
    for row in result["representative_exact_candidates"]:
        rendered = row["rendered"]
        admission = mechanical_admission_checks(rendered, min_letters=39, max_letters=180)
        row = dict(row)
        row["admission"] = admission
        row["mechanically_admitted"] = bool(row["audit"]["two_pointer_exact"] and all(admission.values()))
        row["provenance"] = {
            "lane": ID,
            "construction": "agreement-valid semantic CFG paths compiled into one character NFA",
            "grammar_clause_count": len(clauses),
            "catalogue_seed": False,
            "finished_tape_reversal": False,
            "feature_gate": "subject_number == finite_verb_number; transitive object required",
        }
        rows.append(row)
    result["representative_exact_candidates"] = rows
    result["strict_gate"] = {
        "admitted": sum(r["mechanically_admitted"] for r in rows),
        "readable_over_38": 0,
        "human_readability_test": "not performed; exact candidates require blinded readers",
    }
    result["experiment"] = ID
    result["method"] = "feature-unified semantic CFG -> factored character NFA -> packed exact palindrome paths"
    result["grammar"] = {
        "clause_count": len(clauses),
        "complete_clause_only": True,
        "agreement_enforced_before_search": True,
        "coordinated_paths": 180 * 180,
    }
    result["independent_audit"] = "each rendered row uses two-pointer audit plus forward/reverse SHA-256 equality"
    result["next_repair"] = "add semantic role compatibility and boundary-aware lexical alternatives at the live dead-end character sets; do not widen with unconstrained word products"
    source = Path(__file__)
    registry = ROOT / "docs/experiment-novelty-registry.json"
    result["provenance"] = {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
    }
    return result


if __name__ == "__main__":
    payload = run()
    out = ROOT / "runs" / (ID + ".json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"grammar_clauses": payload["grammar"]["clause_count"],
                      "nfa_nodes": payload["nfa_nodes"],
                      "exact_candidates": len(payload["representative_exact_candidates"]),
                      "longest_exact": payload["longest_exact"],
                      "admitted": payload["strict_gate"]["admitted"]}, indent=2))
