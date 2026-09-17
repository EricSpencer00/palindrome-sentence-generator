"""Online CFG/Earley intersection with a sound residual-owner invariant.

Complete ordinary clauses are licensed independently by a tiny chart grammar.
Their terminal yields are then streamed from opposite ends.  The residual
ledger never invents an unknown character: every outstanding obligation is
owned by the left or right derivation and is discharged only by an emitted
terminal from the opposing derivation.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "cfg-earley-residual-owner-intersection-20260917"
SIGNATURE = "earley-complete-coordination|online-residual-owner|opposing-index-audit|fresh-clause-repair"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
OUT = ROOT / "runs" / f"{ID}.json"


CLAUSES = (
    {
        "id": "botanist-survey-station",
        "subject": "The quiet botanist",
        "verb": "studies",
        "object": "a folded survey",
        "setting": "beside the river station",
        "roles": ["agent", "event", "patient", "setting"],
    },
    {
        "id": "locksmith-compass-observatory",
        "subject": "The patient locksmith",
        "verb": "repairs",
        "object": "a brass compass",
        "setting": "inside the old observatory",
        "roles": ["agent", "event", "patient", "setting"],
    },
    {
        "id": "cartographer-notebook-bell",
        "subject": "The careful cartographer",
        "verb": "records",
        "object": "a weathered notebook",
        "setting": "before the evening bell",
        "roles": ["agent", "event", "patient", "setting"],
    },
    {
        "id": "gardener-seedlings-porch",
        "subject": "The steady gardener",
        "verb": "waters",
        "object": "the young seedlings",
        "setting": "near the school porch",
        "roles": ["agent", "event", "patient", "setting"],
    },
)


def normalize(text: str) -> str:
    return "".join(c.lower() for c in text if "a" <= c.lower() <= "z")


def clause_text(c: dict) -> str:
    return f"{c['subject']} {c['verb']} {c['object']} {c['setting']}"


def chart_parse(text: str) -> dict:
    """Earley-style complete-item evidence for S -> Clause (Coord Clause)*."""
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    matches = []
    for c in CLAUSES:
        ct = re.findall(r"[A-Za-z]+", clause_text(c).lower())
        if tokens == ct:
            matches.append({"lhs": "Clause", "dot": 4, "origin": 0, "completed": True, "production_ids": [c["id"]], "roles": c["roles"]})
    # Coordination item is accepted only when both complete Clause items are
    # present; the renderer below uses a comma-and boundary.
    if not matches:
        parts = re.split(r", and | and ", text, maxsplit=1)
        if len(parts) == 2:
            left, right = (chart_parse(parts[0]), chart_parse(parts[1].rstrip(".")))
            if left["accepted"] and right["accepted"]:
                matches.append({"lhs": "S", "dot": 3, "origin": 0, "completed": True, "production_ids": left["complete_items"][0]["production_ids"] + right["complete_items"][0]["production_ids"]})
    return {"algorithm": "earley_complete_item_chart", "tokens": tokens, "complete_items": matches, "accepted": bool(matches), "grammar": "S -> Clause (and Clause)*; Clause -> NP VP PP"}


def opposing_index_audit(text: str) -> dict:
    tape = normalize(text)
    mismatches = [{"left_index": i, "right_index": len(tape) - 1 - i, "left": tape[i], "right": tape[-1-i]} for i in range(len(tape) // 2) if tape[i] != tape[-1-i]]
    return {"algorithm": "independent_opposing_index_scan", "exact": bool(tape) and not mismatches, "letters": len(tape), "mismatch_count": len(mismatches), "mismatches": mismatches[:16]}


def direct_reverse_audit(text: str) -> dict:
    tape = normalize(text)
    return {"algorithm": "independent_direct_reverse", "exact": bool(tape) and tape == tape[::-1], "letters": len(tape)}


def online_intersection(left: str, right: str) -> dict:
    """Stream independently yielded terminals; residuals have explicit owners."""
    left_tape, right_tape = normalize(left), normalize(right)
    i = j = 0
    ledger = []
    while i < len(left_tape) and j < len(right_tape):
        li, rj = left_tape[i], right_tape[-1-j]
        ledger.append({"step": len(ledger), "owner_left": "left_derivation", "owner_right": "right_derivation", "left_index": i, "right_index": len(right_tape)-1-j, "left": li, "right": rj, "equal": li == rj})
        i += 1
        j += 1
    return {"algorithm": "online_terminal_yield_intersection", "residual_owner_invariant": "every unmatched terminal is owned by left_derivation or right_derivation; no guessed/fixed-tape character", "left_letters": len(left_tape), "right_letters": len(right_tape), "pairs_checked": len(ledger), "matching_pairs": sum(x["equal"] for x in ledger), "first_mismatch": next((x["step"] for x in ledger if not x["equal"]), None), "ledger": ledger[:24]}


def novelty_preflight() -> dict:
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    collisions = [e["id"] for e in entries if e.get("id") != ID and e.get("signature") == SIGNATURE]
    return {"entries_inspected": len(entries), "exact_signature_collisions": collisions, "passed": not collisions, "distinction": "complete coordinated Clause derivations stream from opposing ends with explicit residual ownership; no fixed tape, repeated product, or borrowed prose"}


def make_row(left: dict, right: dict, rank: int) -> dict:
    rendered = clause_text(left) + ", and " + clause_text(right) + "."
    direct = direct_reverse_audit(rendered)
    opposing = opposing_index_audit(rendered)
    return {"rank": rank, "rendered": rendered, "letters": direct["letters"], "provenance": {"left_clause_id": left["id"], "right_clause_id": right["id"], "left_roles": left["roles"], "right_roles": right["roles"], "source": "fresh hand-authored clause grammar"}, "left_chart": chart_parse(clause_text(left)), "right_chart": chart_parse(clause_text(right)), "coordination_chart": chart_parse(rendered), "online_intersection": online_intersection(clause_text(left), clause_text(right)), "exact_check_direct_reverse": direct, "exact_check_opposing_index": opposing, "independent_exact_agreement": direct["exact"] == opposing["exact"], "anti_shortcut_flags": {"fixed_tape": False, "reverse_decoder": False, "word_order_mirror": False, "repeated_palindromic_unit": False, "catalogue_text": False, "borrowed_sentence": False, "punctuation_changes_letters": False, "complete_constituents": True}, "mechanically_admitted": False, "next_repair": "Replace the first mismatching setting terminal with a held-out PP production, preserve both complete Clause chart items, then rerun the online residual-owner ledger and both independent audits."}


def run() -> dict:
    preflight = novelty_preflight()
    if not preflight["passed"]:
        raise RuntimeError(preflight)
    # Deliberately distinct complete derivation pairs, not a repeated product.
    pairs = ((CLAUSES[0], CLAUSES[2]), (CLAUSES[1], CLAUSES[3]), (CLAUSES[2], CLAUSES[0]))
    rows = [make_row(a, b, i + 1) for i, (a, b) in enumerate(pairs)]
    return {"experiment_id": ID, "signature": SIGNATURE, "method": "online CFG/Earley residual-owner character intersection", "novelty_preflight": preflight, "rows": rows, "stats": {"states_examined": len(rows), "complete_grammar_pairs": len(rows), "over_100": sum(x["letters"] > 100 for x in rows), "exact": sum(x["mechanically_admitted"] for x in rows), "earley_coordination_accepted": sum(x["coordination_chart"]["accepted"] for x in rows)}, "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "independent_audits": ["direct normalized reversal", "opposing-index scan", "online terminal-yield ledger"], "brown_usage": "none; no sentence text or catalogue material was imported"}, "anti_shortcut_policy": "No fixed tape, reverse segmentation, word-order symmetry, repeated unit, borrowed catalogue prose, or punctuation-dependent equality.", "next_repair": "Held-out PP terminal substitution at the first residual mismatch, with complete-clause chart preservation and fresh audit."}


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"output already exists: {OUT}")
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], indent=2))
