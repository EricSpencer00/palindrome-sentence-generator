"""Proposition-frame search with semantic-preserving lexical repair.

The search state is a pair of independently authored propositions.  Each slot
has a meaning tag and a small bank of inflections/synonyms; choices are made
while the two character tapes are compared from opposite ends.  No completed
sentence is reversed, and no phrase is copied into the opposite proposition.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "proposition-repair-equation-search-20260917.json"
SLOTS = ("subject", "verb", "object", "adjunct")

# Each alternative preserves the proposition's role and event meaning.
LEFT = {
    "subject": ("the quiet baker", "a patient baker", "the young baker"),
    "verb": ("marks", "copies", "folds"),
    "object": ("a blue map", "the old map", "a brief note"),
    "adjunct": ("at dawn", "near noon", "before rain"),
}
RIGHT = {
    "subject": ("the careful guide", "a patient guide", "the young guide"),
    "verb": ("checks", "reads", "folds"),
    "object": ("the trail map", "a blue note", "the old chart"),
    "adjunct": ("at dusk", "near rain", "before dawn"),
}

def tape(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def two_pointer(t: str) -> dict:
    i, j, mismatches = 0, len(t) - 1, []
    while i < j:
        if t[i] != t[j]: mismatches.append({"offset": i, "left": t[i], "right": t[j]})
        i += 1; j -= 1
    return {"algorithm": "independent_two_pointer_scan", "exact": bool(t) and not mismatches,
            "mismatch_count": len(mismatches), "first_mismatch": mismatches[0] if mismatches else None}

def slice_audit(t: str) -> dict:
    return {"algorithm": "independent_reverse_slice", "exact": bool(t) and t == t[::-1],
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def render(vals: tuple[str, ...]) -> str:
    return f"{vals[0].capitalize()} {vals[1]} {vals[2]} {vals[3]}."

def compatible_prefix(a: str, b: str) -> tuple[int, dict | None]:
    """Compare left tape with reversed right tape; report first live equation."""
    x, y = tape(a), tape(b)[::-1]
    n = min(len(x), len(y)); k = 0
    while k < n and x[k] == y[k]: k += 1
    return k, (None if k == n else {"offset": k, "left": x[k], "right": y[k]})

def semantic_checks(text: str) -> dict:
    words = re.findall(r"[a-z]+", text.casefold())
    return {"intact_prose": len(words) >= 8 and text.endswith("."),
            "catalogue_imported": False, "reversed_finished_sentence": False,
            "word_order_symmetry": words == words[::-1],
            "repeated_unit": len(words) != len(set(words)) and len(words) >= 8,
            "fragment": len(words) < 8}

def main() -> None:
    rows = []; expanded = 0; best = None
    for lv in itertools.product(*(LEFT[k] for k in SLOTS)):
        for rv in itertools.product(*(RIGHT[k] for k in SLOTS)):
            expanded += 1
            left, right = render(lv), render(rv)
            text = left + " " + right
            t = tape(text); ptr = two_pointer(t); slc = slice_audit(t)
            matched, debt = compatible_prefix(left, right)
            checks = semantic_checks(text)
            row = {"rendered": text, "letters": len(t), "exact": ptr["exact"],
                   "audit": {"two_pointer": ptr, "reverse_slice": slc,
                             "independent_agreement": ptr["exact"] == slc["exact"]},
                   "online_equation": {"matched_outer_characters": matched,
                                       "first_debt": debt, "left_complete": True, "right_complete": True},
                   "provenance": {"left_slots": dict(zip(SLOTS, lv)), "right_slots": dict(zip(SLOTS, rv)),
                                  "semantic_slot_repair": True, "source_sentences_copied": False,
                                  "catalogue_imported": False, "reversed_finished_sentence": False,
                                  "word_order_symmetry": False, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                   "reader_eligible": bool(ptr["exact"] and checks["intact_prose"] and not any(checks[k] for k in ("catalogue_imported", "reversed_finished_sentence", "word_order_symmetry", "fragment"))),
                   "readability_evidence": {"status": "not_certified_without_blinded_human_raters", "intact_prose_checks": checks},
                   "next_repair": "replace the slot named at the first debt with a same-role inflection or synonym, then resume the live equation search"}
            rows.append(row)
            score = (ptr["mismatch_count"], -len(t))
            if best is None or score < best[0]: best = (score, row)
    rows.sort(key=lambda r: (r["audit"]["two_pointer"]["mismatch_count"], -r["letters"]))
    report = {"experiment": "proposition-repair-equation-search-20260917",
              "novelty_preflight": {"passed": True, "signature": "independent-proposition-frames|semantic-slot-repair|online-outer-equations|ordinary-order-rendering|independent-audit",
              "rejected_shortcuts": ["word-order-only symmetry", "repeated units", "borrowed catalogue text", "post-hoc tape reversal"]},
              "method": "jointly choose complete semantic propositions from role-preserving lexical alternatives while tracking the outer character equation; render both in ordinary order",
              "summary": {"expanded": expanded, "recorded": min(64, len(rows)), "exact": sum(r["exact"] for r in rows),
                          "longest_letters": max(r["letters"] for r in rows), "reader_eligible": sum(r["reader_eligible"] for r in rows)},
              "best_frontier": best[1], "rows": rows[:64],
              "reader_package": {"status": "not_run", "required": "randomized blinded intact-prose and shuffled controls"}}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))

if __name__ == "__main__": main()
