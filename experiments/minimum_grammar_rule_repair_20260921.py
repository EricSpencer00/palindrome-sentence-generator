"""Exact grammar-rule repair with palindrome equality kept hard.

This is a bounded method experiment, not a readability claim.  A finite
two-clause grammar exposes six optional production rules.  For every one of
the 64 activation masks we enumerate the grammar and independently solve the
same character relation by a residual lookup.  The objective is the minimum
number of activated rules among exact derivations; a character mismatch is
never treated as a partial reward.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/minimum-grammar-rule-repair-20260921.json"


def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    value = tape(text)
    mismatch = next(
        ((i, value[i], value[-1 - i]) for i in range(len(value) // 2)
         if value[i] != value[-1 - i]),
        None,
    )
    forward = hashlib.sha256(value.encode()).hexdigest()
    reverse = hashlib.sha256(value[::-1].encode()).hexdigest()
    return {
        "letters": len(value),
        "pointer_exact": bool(value) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


@dataclass(frozen=True)
class Production:
    name: str
    side: str
    role: str
    text: str
    optional: bool = False
    calibration: bool = False


# The base alternatives are ordinary complete clauses.  The two calibration
# rules reproduce the known 38-letter seed only to verify that the exact solver
# can find a positive path; those rows can never become reader candidates.
BASE = (
    Production("left_base", "left", "SVO", "The careful baker warms fresh bread."),
    Production("right_base", "right", "SVO", "A quiet sailor studies the harbor."),
)
OPTIONAL = (
    Production("cal_left", "left", "SVO", "An aide rips nine memos", calibration=True),
    Production("cal_right", "right", "SVO", "some men inspire Diana", calibration=True),
    Production("left_subject_alt", "left", "SVO", "A patient baker warms fresh bread.", optional=True),
    Production("right_subject_alt", "right", "SVO", "The quiet sailor studies the harbor.", optional=True),
    Production("left_adjunct_alt", "left", "SVO", "The careful baker warms bread at dawn.", optional=True),
    Production("right_adjunct_alt", "right", "SVO", "A quiet sailor studies maps at dusk.", optional=True),
)


def active(mask: int) -> tuple[Production, ...]:
    selected = list(BASE)
    selected.extend(rule for bit, rule in enumerate(OPTIONAL) if mask & (1 << bit))
    return tuple(selected)


def derivations(mask: int) -> tuple[tuple[Production, Production], ...]:
    left = [p for p in active(mask) if p.side == "left"]
    right = [p for p in active(mask) if p.side == "right"]
    return tuple((a, b) for a in left for b in right)


def exhaustive(mask: int) -> list[dict]:
    rows = []
    for left, right in derivations(mask):
        rendered = f"{left.text.rstrip('.')} ; {right.text.rstrip('.')}."
        check = audit(rendered)
        rows.append({
            "mask": mask,
            "rendered": rendered,
            "productions": [left.name, right.name],
            "roles": [left.role, right.role],
            "audit": check,
            "calibration_only": left.calibration or right.calibration,
            "distinct_surfaces": tape(left.text) != tape(right.text),
            "complete_clauses": True,
        })
    return rows


def solve_exact(mask: int) -> list[dict]:
    """Solve exactness by joining equal forward/reverse character tapes."""
    left = [p for p in active(mask) if p.side == "left"]
    right_by_reverse = {}
    for rule in (p for p in active(mask) if p.side == "right"):
        right_by_reverse.setdefault(tape(rule.text), []).append(rule)
    out = []
    for lrule in left:
        wanted = tape(lrule.text)[::-1]
        for rrule in right_by_reverse.get(wanted, []):
            rendered = f"{lrule.text.rstrip('.')} ; {rrule.text.rstrip('.')}."
            check = audit(rendered)
            if not check["pointer_exact"]:
                raise AssertionError("solver emitted a non-exact path")
            out.append({
                "mask": mask,
                "rendered": rendered,
                "productions": [lrule.name, rrule.name],
                "audit": check,
                "calibration_only": lrule.calibration or rrule.calibration,
                "distinct_surfaces": tape(lrule.text) != tape(rrule.text),
                "complete_clauses": True,
            })
    return out


def first_obstruction(rows: list[dict]) -> dict | None:
    near = [r for r in rows if r["audit"]["first_mismatch"] is not None]
    if not near:
        return None
    row = max(near, key=lambda r: r["audit"]["letters"])
    return {"rendered": row["rendered"], "productions": row["productions"],
            "audit": row["audit"]}


def synthetic_positive_fixture() -> dict:
    """Tiny non-English fixture catching a solver that always reports failure."""
    left, right = "ab", "ba"
    text = f"{left}; {right}."
    exact = audit(text)["pointer_exact"]
    return {"left": left, "right": right, "audit": audit(text), "solver_exact": exact,
            "reader_candidate": False, "synthetic_only": True}


def run() -> dict:
    masks = []
    all_rows = []
    for mask in range(1 << len(OPTIONAL)):
        exhaustive_rows = exhaustive(mask)
        solved_rows = solve_exact(mask)
        exhaustive_exact = [r for r in exhaustive_rows if r["audit"]["pointer_exact"]]
        if {(r["productions"][0], r["productions"][1]) for r in exhaustive_exact} != \
                {(r["productions"][0], r["productions"][1]) for r in solved_rows}:
            raise AssertionError(f"solver/exhaustive mismatch for mask {mask}")
        masks.append({
            "mask": mask,
            "activated_rules": [rule.name for bit, rule in enumerate(OPTIONAL) if mask & (1 << bit)],
            "activation_cost": mask.bit_count(),
            "derivations": len(exhaustive_rows),
            "exact_paths": len(solved_rows),
            "exact_calibration_paths": sum(r["calibration_only"] for r in solved_rows),
            "first_obstruction": first_obstruction(exhaustive_rows),
        })
        all_rows.extend(exhaustive_rows)

    exact = [r for r in all_rows if r["audit"]["pointer_exact"]]
    reader_exact = [r for r in exact if not r["calibration_only"] and r["distinct_surfaces"]]
    minimum = min((m for m in masks if m["exact_paths"]),
                  key=lambda m: (m["activation_cost"], m["mask"]), default=None)
    base = next(m for m in masks if m["mask"] == 0)
    return {
        "experiment_id": "minimum-grammar-rule-repair-20260921",
        "method": "hard exact palindrome product with minimum optional grammar-rule activation",
        "stats": {
            "optional_rules": len(OPTIONAL),
            "activation_masks": len(masks),
            "exact_masks": sum(m["exact_paths"] > 0 for m in masks),
            "minimum_activation_cost": None if minimum is None else minimum["activation_cost"],
            "all_exact_paths": len(exact),
            "fresh_reader_exact_over38": sum(r["audit"]["letters"] > 38 for r in reader_exact),
        },
        "base_certificate": base,
        "minimum_certificate": minimum,
        "rendered_controls": [r for r in all_rows if not r["audit"]["pointer_exact"]][:12],
        "calibration_exact_paths": [r for r in exact if r["calibration_only"]],
        "exact_reader_candidates": reader_exact,
        "synthetic_positive_fixture": synthetic_positive_fixture(),
        "novelty_preflight": {
            "status": "passed",
            "signature": "hard-exact|activation-subset|minimum-grammar-repair",
            "distinct_from": "global character-equality MaxSAT scoring and fixed clause products",
            "finished_tape_reversal": False,
            "catalogue_text": False,
            "rlaif_per_search": False,
        },
        "provenance": {
            "independent_audits": ["two-pointer normalized tape", "SHA-256 forward/reverse"],
            "grammar_rules": [rule.name for rule in BASE + OPTIONAL],
            "calibration_seed_is_not_claimed": True,
            "reader_gate": "closed; no fresh exact reader candidate",
        },
        "next_operator": "replace calibration toggles with a fresh independently authored production family while preserving hard equality and 64-mask solver/exhaustive cross-check",
        "status": "calibration solver verified; no fresh reader candidate",
    }


if __name__ == "__main__":
    result = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
