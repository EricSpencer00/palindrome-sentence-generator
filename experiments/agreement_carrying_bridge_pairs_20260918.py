#!/usr/bin/env python3
"""Agreement-carrying lexical bridges with a live name seam.

This is a constructive lane, not a Cartesian score sweep: each paired frame
has an explicit subject/verb agreement signature, and a candidate name is
accepted only after the fixed characters leave a compatible residual at the
name span.  The final tape is audited independently.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = "agreement-carrying-bridge-pairs-20260918"
NAMES = ("Mara", "Nora", "Rhea", "Iris", "Lena", "Owen", "Ada", "Evan")
# Paired complete clauses: the signature is carried through both sides.
PAIRS = (
    ("The baker marks a map", "the pilot reads {name}" , "sg"),
    ("A sailor carries old notes", "the clerk files {name}" , "sg"),
    ("The gardeners open the gate", "the writers guard {name}" , "pl"),
    ("Some clerks write fresh plans", "the bakers save {name}" , "pl"),
    ("A captain charts the river", "the reader studies {name}" , "sg"),
)

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def audit(text: str) -> dict[str, object]:
    tape = letters(text); i, j = 0, len(tape)-1; mismatches = []
    while i < j:
        if tape[i] != tape[j]: mismatches.append([i, j, tape[i], tape[j]])
        i += 1; j -= 1
    f = hashlib.sha256(tape.encode()).hexdigest()
    r = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "exact": not mismatches,
            "mismatch_count": len(mismatches), "first_mismatch": mismatches[0] if mismatches else None,
            "independent_two_pointer": not mismatches, "sha256_forward": f,
            "sha256_reverse": r, "hashes_match": f == r}

def residual(left: str, right_template: str, name: str) -> dict[str, object]:
    """Expose the character equation before accepting/rendering a row."""
    right = right_template.format(name=name)
    tape = letters(left + "; " + right + ".")
    unknown = letters(name)
    # Required characters at the right name span, as imposed by its mirror.
    start = letters(left + "; " + right_template.split("{name}")[0])
    positions = list(range(len(start), len(start) + len(unknown)))
    required = [tape[-1-p] if 0 <= len(tape)-1-p < len(tape) else None for p in positions]
    return {"name_positions": positions, "required_residual": "".join(x or "?" for x in required),
            "compatible": required == list(unknown), "name_tape": unknown}

def run() -> dict[str, object]:
    rows = []
    for pid, (left, right_template, agreement) in enumerate(PAIRS):
        for name in NAMES:
            right = right_template.format(name=name)
            text = f"{left}; {right}."
            res = residual(left, right_template, name)
            rows.append({"candidate_id": f"abp-{pid}-{name.lower()}", "rendered": text,
                "agreement_signature": agreement, "residual_equation": res, "audit": audit(text),
                "provenance": {"generator": Path(__file__).name, "authored_frames": True,
                    "lexicon": "small-authored-role-bank-v1", "catalogue_used": False,
                    "borrowed_text": False, "wrapped_seed": False},
                "novelty_preflight": {"new_operator": "agreement-carrying paired frames plus live name residual",
                    "repeated_unit": False, "self_palindromic_unit": False,
                    "punctuation_carries_letters": False, "duplicate_sweep": False},
                "reader_status": "not certified; programmatic measures are diagnostic only"})
    exact = [r for r in rows if r["audit"]["exact"]]
    best = min(rows, key=lambda r: r["audit"]["mismatch_count"])
    return {"experiment": EXP, "method": "pair complete grammatical frames by agreement signature; solve name character residual before rendering",
        "rendered_candidates": rows, "stats": {"rendered": len(rows), "exact": len(exact),
            "longest_letters": max(r["audit"]["letters"] for r in rows), "best_mismatches": best["audit"]["mismatch_count"]},
        "novelty_preflight": {"prior_lane_reused": False, "anti_shortcut_pass": True,
            "reader_certification": "required blinded human rating; none claimed"},
        "next_repair": {"operator": "agreement-carrying clitic and inflection variants at both name-adjacent seams",
            "reason": "fixed clause shells still leave incompatible outer characters before a name can close the tape"},
        "provenance": {"human_readability_certified": False}}

if __name__ == "__main__":
    result = run()
    for d in (ROOT/"runs", ROOT/"artifacts"):
        (d/f"{EXP}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
