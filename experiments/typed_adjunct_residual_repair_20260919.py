"""Second repair for the typed seam lane: agreement-safe adjunct growth.

The prior lane found the 38-letter anchor but no longer closure.  This repair
does not widen the same clause product blindly.  It appends one independently
authored adjunct to a complete clause, carries the resulting residual through
the boundary index, and retains only pairs in which at least one side uses the
new adjunct edge.  Exact rows are still audited independently and reader
eligibility remains closed until a blinded study.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.typed_constituent_seam_search_20260919 import (
    Constituent,
    _content,
    _hidden_proper_span,
    _render,
    _zipper,
    audit,
    clauses,
)
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT_ID = "typed-adjunct-residual-repair-20260919"
ADJUNCTS = ("at dawn", "near the river", "under the moon", "after rain",
            "before dawn", "with care", "by the shore", "for Diana")


def augmented_clauses() -> tuple[Constituent, ...]:
    result = list(clauses())
    for item in clauses():
        if item.kind == "anchor":
            continue
        for adjunct in ADJUNCTS:
            text = f"{item.text} {adjunct}"
            result.append(Constituent(text, "complete_clause_with_adjunct", item.number,
                                      item.valency, _content(text)))
    return tuple(result)


def run() -> dict[str, object]:
    items = augmented_clauses()
    by_tape: dict[str, list[Constituent]] = defaultdict(list)
    for item in items:
        tape = normalize_letters(item.text)
        by_tape[tape].append(item)

    rows: list[dict[str, object]] = []
    for left in items:
        for right in by_tape.get(normalize_letters(left.text)[::-1], ()):
            if left.content_words & right.content_words:
                continue
            if left.kind == "anchor" and right.kind == "anchor":
                continue
            text = _render(left, right)
            checks = mechanical_admission_checks(text, min_letters=30, max_letters=2000)
            adjunct_edge = (
                left.kind == "complete_clause_with_adjunct"
                or right.kind == "complete_clause_with_adjunct"
            )
            row = {
                "rendered": text,
                "left_kind": left.kind,
                "right_kind": right.kind,
                "adjunct_edge": adjunct_edge,
                "zipper": _zipper(left.text, right.text),
                "audit": audit(text),
                "mechanical_checks": checks,
                "hidden_proper_span": _hidden_proper_span(text),
                "reader_status": "unreviewed; programmatic checks never certify readability",
            }
            row["mechanically_admitted"] = (
                row["audit"]["two_pointer_exact"]
                and row["adjunct_edge"]
                and not row["hidden_proper_span"]
                and all(checks.values())
            )
            rows.append(row)

    # Preserve intact prose controls from the new grammar even when no exact
    # adjunct closure exists.  They make the failed lane inspectable and give
    # the next repair a concrete residual rather than a bare zero count.
    controls = sorted(
        (item for item in items if item.kind == "complete_clause_with_adjunct"),
        key=lambda item: (len(normalize_letters(item.text)), item.text), reverse=True,
    )[:120]
    control_pairs = []
    for left in controls:
        for right in controls:
            if left.text != right.text and left.content_words.isdisjoint(right.content_words):
                control_pairs.append((left, right))
                break
        if len(control_pairs) >= 3:
            break
    for left, right in control_pairs:
        text = _render(left, right)
        if any(row["rendered"] == text for row in rows):
            continue
        rows.append({
            "rendered": text,
            "left_kind": left.kind,
            "right_kind": right.kind,
            "adjunct_edge": True,
            "control": True,
            "zipper": _zipper(left.text, right.text),
            "audit": audit(text),
            "mechanical_checks": mechanical_admission_checks(text, min_letters=30, max_letters=2000),
            "hidden_proper_span": _hidden_proper_span(text),
            "mechanically_admitted": False,
            "reader_status": "intact prose control; not an exact candidate",
        })

    unique = {}
    for row in rows:
        unique.setdefault(row["rendered"], row)
    rows = list(unique.values())
    exact = [row for row in rows if row["audit"]["two_pointer_exact"]]
    admitted = [row for row in exact if row["mechanically_admitted"]]
    longest = max(rows, key=lambda row: row["audit"]["letters"], default=None)
    return {
        "experiment_id": EXPERIMENT_ID,
        "method": "agreement-safe adjunct edge expansion with exact residual tape index",
        "status": "completed_exact" if exact else "completed_no_exact_closure",
        "adjuncts": list(ADJUNCTS),
        "clauses": len(items),
        "rows": rows,
        "exact_candidates": exact,
        "stats": {"rows": len(rows), "exact": len(exact),
                  "mechanically_admitted": len(admitted),
                  "longest_letters": longest["audit"]["letters"] if longest else 0,
                  "longest_exact_letters": max((r["audit"]["letters"] for r in exact), default=0)},
        "provenance": {"lexical_source": "authored complete clauses plus new adjunct bank",
                       "catalogue_imported": False, "finished_tape_reversed": False,
                       "independent_audits": ["outside-in two-pointer", "forward/reverse SHA-256"],
                       "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "next_repair": {
            "action": "replace the first failing adjunct boundary with a valency-compatible adjunct whose opening character matches the live residual, then rerun only that typed edge",
            "reason": "adjunct growth preserved complete prose but did not produce a new exact closure",
            "reader_test": "none until a new exact row clears the shared mechanical gate",
        },
    }


if __name__ == "__main__":
    result = run()
    (ROOT / "runs" / f"{EXPERIMENT_ID}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
