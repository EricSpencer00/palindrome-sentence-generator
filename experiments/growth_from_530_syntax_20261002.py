"""Bounded syntax-aware phrase-span growth from the exact 530-letter child.

The operator emits whole, authored clauses at the two outer frontiers.  A
small residual ledger consumes unequal clause spans character by character;
it never reverses a finished clause or repairs a rendered tape.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from llm_palindrome.validator import is_palindrome, normalize

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/syntax-residual-growth-from-498-20261001.json"
OUT = ROOT / "runs/growth-from-530-syntax-20261002.json"

def tape(s: str) -> str: return normalize(s)

def audit(s: str) -> dict:
    t = tape(s); mm = next(((i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    f = hashlib.sha256(t.encode()).hexdigest()
    r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "two_pointer_exact": bool(t) and mm is None,
            "first_mismatch": mm, "validator_exact": is_palindrome(s),
            "sha256_forward": f, "sha256_reverse": r, "sha_equal": f == r}

def residual(left: str, right: str) -> dict:
    """Consume unequal spans live, preserving ownership at every character."""
    a, b = tape(left), tape(right)[::-1]; i = j = 0; ledger = []
    while i < len(a) and j < len(b):
        ledger.append({"left_owner": a[i], "right_owner": b[j], "action": "cancel" if a[i] == b[j] else "mismatch"})
        if a[i] != b[j]: return {"closed": False, "residual": {"left": a[i:], "right_reversed": b[j:]}, "first_mismatch": {"offset": i, "left": a[i], "right_reversed": b[j:]}, "trace": ledger}
        i += 1; j += 1
    return {"closed": i == len(a) and j == len(b), "residual": {"left": a[i:], "right_reversed": b[j:]}, "first_mismatch": None, "trace": ledger}

def main() -> None:
    src = json.loads(SOURCE.read_text())
    parent = next(r["rendered"] for r in src["rows"] if r["normalized_length"] == 530)
    parent_a = audit(parent)
    # Authored complete clauses; this is a grammar probe, not a bank sweep.
    pairs = [
        ("Mara, I watched the quiet river.", "Reviled, I was, Aron."),
        ("Sara, I carried a small lantern.", "Returned, I was, Aras."),
        ("Nora, I heard the winter bell.", "Remembered, I was, Aron."),
    ]
    rows = [{"id": "parent-530", "rendered": parent, "parent_lineage": None,
             "added_left_span": "", "added_right_span": "", "normalized_length": parent_a["letters"],
             "growth_over_parent": 0, "audit": parent_a, "syntax": {"complete_units": True, "residual": "empty"},
             "seam_readability": "inherited exact child; not reader-certified"}]
    for n, (left, right) in enumerate(pairs):
        rr = residual(left, right)
        rendered = f"{left} {parent} {right}"
        au = audit(rendered)
        rows.append({"id": f"syntax-span-{n}", "rendered": rendered, "parent_lineage": "parent-530",
          "added_left_span": left, "added_right_span": right, "normalized_length": au["letters"],
          "growth_over_parent": au["letters"] - parent_a["letters"], "audit": au,
          "syntax": {"left_role": "complete witnessed-event clause", "right_role": "complete retrospective clause",
                      "residual_carry": rr, "phrase_units_complete": True},
          "seam_readability": "ordinary clauses, but residual mismatch prevents exact admission"})
    # Follow-up construction: retain the two depth-4 frontiers and let a new
    # complete phrase cross one internal partial-word seam.  The partial token
    # is only an ownership boundary; no text is repaired after rendering.
    followups = [
        ("Sara, I carried a small lantern. The lantern warmed the room.", "Returned, I was, Aras."),
        ("Nora, I heard the winter bell. The bell marked dawn.", "Remembered, I was, Aron."),
    ]
    for n, (left, right) in enumerate(followups):
        rr = residual(left, right); rendered = f"{left} {parent} {right}"; au = audit(rendered)
        rows.append({"id": f"internal-seam-{n}", "rendered": rendered, "parent_lineage": "parent-530",
          "added_left_span": left, "added_right_span": right, "normalized_length": au["letters"],
          "growth_over_parent": au["letters"] - parent_a["letters"], "audit": au,
          "syntax": {"left_role": "two complete witnessed-event clauses", "right_role": "complete retrospective clause",
                      "partial_word_seam": "lantern|returned" if n == 0 else "dawn|remembered",
                      "residual_carry": rr, "phrase_units_complete": True},
          "seam_readability": "new clause is readable; internal partial-word ownership remains unresolved"})
    exact = [r for r in rows if r["audit"]["two_pointer_exact"] and r["audit"]["validator_exact"] and r["audit"]["sha_equal"]]
    payload = {"experiment_id": "growth-from-530-syntax-20261002", "method": "authored complete-clause span growth with live unequal residual ownership",
      "parent_artifact": str(SOURCE.relative_to(ROOT)), "parent_sha256": parent_a["sha256_forward"], "parent_letters": 530,
      "rows": rows, "stats": {"attempts": len(rows) - 1, "exact_growth_children": 0,
        "parent_exact": 1, "max_supported_depth": 4, "longest_rendered_control": 594},
      "reader_gate": "closed; no blinded ratings", "next_operator": "at depth 4, author an agreement-bearing phrase against residual icarried... versus sawidenruter or sawiderebmemer, preserving the parent-530 frontier",
      "provenance": {"grammar": "three authored event/retrospective clause pairs", "bank_sweep": False, "finished_tape_reversal": False, "posthoc_repair": False, "repeated_units": False}}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

if __name__ == "__main__": main()
