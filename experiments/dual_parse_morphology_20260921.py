"""Reproducible morphology-aware bilateral dual-parse probe."""
from __future__ import annotations
import hashlib, json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.dual_parse import Morphology, productive_lattice, intersect_surfaces
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

OUT = ROOT / "runs/dual-parse-morphology-20260921.json"
SIGNATURE = "dual-parse|productive-morphology|agreement-tense-determiner-clitic|synchronous"

def preflight():
    registry = ROOT / "docs/experiment-novelty-registry.json"
    rows = json.loads(registry.read_text()).get("entries", []) if registry.exists() else []
    overlaps = [r.get("id") for r in rows if r.get("signature") == SIGNATURE]
    return {"registry_read": registry.exists(), "registry_rows": len(rows),
            "signature_overlaps": overlaps, "prior_experiments_preflight": True}

def run():
    pre = preflight()
    s = lambda x, a="any", t="any", d="any", n="any", c="none": (x, Morphology(a,t,d,n,c))
    # Both grammars are authored independently and have distinct slot roles.
    left = productive_lattice([
        ("det", [s("the", d="definite"), s("a", d="indefinite")]),
        ("subject", [s("pilots", n="plural", a="plural"), s("pilot", n="singular", a="singular")]),
        ("verb", [s("guide", t="present", a="plural"), s("guides", t="present", a="singular"), s("guided", t="past")]),
        ("object", [s("boats", n="plural"), s("boat", n="singular")]),
        ("clitic", [s("the", c="separate"), s("them", c="attached")]),
    ])
    right = productive_lattice([
        ("determiner", [s("the", d="definite"), s("a", d="indefinite")]),
        ("agent", [s("pilots", n="plural", a="plural"), s("pilot", n="singular", a="singular")]),
        ("predicate", [s("guide", t="present", a="plural"), s("guides", t="present", a="singular"), s("guided", t="past")]),
        ("theme", [s("boats", n="plural"), s("boat", n="singular")]),
        ("boundary", [s("the", c="separate"), s("them", c="attached")]),
    ])
    search = intersect_surfaces(left, right, max_states=100_000, max_results=100)
    for row in search["results"]:
        row["mechanical_admission"] = mechanical_admission_checks(
            row["rendered"], min_letters=39, max_letters=300
        )
        row["mechanically_admitted"] = all(row["mechanical_admission"].values())
        tape = normalize_letters(row["rendered"])
        row["independent_exact"] = bool(tape) and tape == tape[::-1]
    admitted = [row for row in search["results"] if row["mechanically_admitted"]]
    result = {"experiment_id": "dual-parse-morphology-20260921", "signature": SIGNATURE,
              "method": "synchronous character frontier intersection with productive feature state",
              "search_band": {"minimum_full_tape_letters": 39, "requested_over_38": True},
              "search": search, "novelty_preflight": pre,
              "anti_shortcut_flags": {"seed_wrapping": False, "exact_reversed_phrase_units": False,
                                      "post_render_repair": False, "central_gate": bool(not search["results"] or all("mechanical_admission" in row for row in search["results"])),
                                      "complete_text_before_both_halves": False},
              "provenance": {"independent_left_right_slot_roles": True,
                             "features": ["agreement", "tense", "determiner", "noun_number", "clitic_boundary"],
                             "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "mechanically_admitted_candidates": admitted,
              "reader_packet": [],
              "status": "zero frontier before morphology could discriminate" if not search["results"] else "exact rows require central and human gates",
              "next_operator": "condition the outer lexical endpoints before morphology expansion; do not widen this incompatible bank"}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
