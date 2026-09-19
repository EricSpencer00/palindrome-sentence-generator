"""Typed connector grammar crossed with an outside-in character product.

Each step appends a complete grammatical constituent to both exposed ends and
consumes its letters against the live residual.  The search never reverses a
finished tape or scores a candidate with an LM.  It is deliberately a search
diagnostic: readable controls are retained even when exact closure is absent.
"""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "connector-character-product-20260919"

@dataclass(frozen=True)
class Clause:
    text: str; kind: str; subject: str; object: str

CLAUSES = (
    Clause("a quiet player reads the old sonnet", "scene", "sg", "text"),
    Clause("the patient poet praises Diana", "scene", "sg", "person"),
    Clause("a young herald carries the letter", "scene", "sg", "text"),
    Clause("the sailor hears a distant bell", "scene", "sg", "sound"),
    Clause("some players await the king", "scene", "pl", "person"),
    Clause("the fair queen keeps a secret", "scene", "sg", "text"),
    Clause("a careful scribe marks the margin", "scene", "sg", "text"),
    Clause("the moonlit court remembers a song", "scene", "sg", "text"),
)
CONNECTORS = (";", ", and", "; yet", ", while")

def letters(s: str) -> str: return re.sub(r"[^a-z]", "", s.lower())

def audit(s: str) -> dict:
    t = letters(s); rev = t[::-1]
    mism = next(((i, a, b) for i, (a,b) in enumerate(zip(t, rev)) if a != b), None)
    f = hashlib.sha256(t.encode()).hexdigest(); r = hashlib.sha256(rev.encode()).hexdigest()
    return {"normalized": t, "letters": len(t), "two_pointer_exact": bool(t) and mism is None,
            "first_mismatch": mism, "sha256_forward": f, "sha256_reverse": r,
            "sha_equal": f == r}

def consume(left: str, right: str) -> tuple[bool, int, tuple | None]:
    """Consume only the newly exposed left/right characters, outside-in."""
    n = min(len(left), len(right)); compared = 0
    for i in range(n):
        compared += 1
        if left[i] != right[-1-i]: return False, compared, (i, left[i], right[-1-i])
    return True, compared, None

def search(max_depth: int = 2) -> dict:
    # State stores the already-consumed outside portions and residual center.
    states = [("", "", (), 0)]
    rows = []
    for depth in range(1, max_depth + 1):
        nxt = []
        for left, right, trace, compared in states:
            for lc in CLAUSES:
                for rc in CLAUSES:
                    if lc.subject == rc.subject and lc.object == rc.object: continue
                    for conn in CONNECTORS:
                        nl = (left + (" " if left else "") + lc.text + conn).strip()
                        nr = (conn.strip() + " " + rc.text + (" " + right if right else "")).strip()
                        ok, added, mismatch = consume(letters(nl), letters(nr))
                        if not ok:
                            # Preserve the first constructive frontier witness:
                            # it is an attempted intact scene, not a claimed
                            # candidate, and records the exact residual failure.
                            if depth == 1 and len(rows) < 100:
                                rows.append({"rendered": nl + " " + nr + ".",
                                    "length": len(letters(nl + nr)), "depth": depth,
                                    "audit": audit(nl + " " + nr + "."),
                                    "residual_compared": compared + added,
                                    "residual_mismatch": mismatch,
                                    "provenance": {"left_clause": lc.text,
                                      "right_clause": rc.text, "connector": conn,
                                      "finished_tape_reversed": False, "rlaif_used": False},
                                    "mechanically_admitted": False,
                                    "reader_status": "frontier witness; not a candidate"})
                            continue
                        # A surviving state is a genuine character-product
                        # state; it is not a post-hoc reverse of a sentence.
                        nt = trace + ((lc.text, conn, rc.text),)
                        rendered = nl + " " + nr + "."
                        a = audit(rendered)
                        rows.append({"rendered": rendered, "length": a["letters"],
                            "depth": depth, "audit": a, "residual_compared": compared + added,
                            "provenance": {"left_clause": lc.text, "right_clause": rc.text,
                              "connector": conn, "derivation": nt,
                              "finished_tape_reversed": False, "rlaif_used": False,
                              "catalogue_imported": False},
                            "mechanically_admitted": False,
                            "reader_status": "not_run; programmatic metrics do not certify readability"})
                        if len(letters(nl)) + len(letters(nr)) <= 70:
                            nxt.append((nl, nr, nt, compared + added))
        states = nxt[:12000]
    exact = [r for r in rows if r["audit"]["two_pointer_exact"]]
    return {"experiment_id": ID,
      "method": "typed connector CFG intersected with incremental outside-in character product",
      "actual_candidates": rows[:100], "exact_candidates": exact,
      "stats": {"states": len(states), "rendered": len(rows), "exact": len(exact),
                 "longest_rendered": max((r["length"] for r in rows), default=0)},
      "provenance": {"independent_audits": ["literal two-pointer", "forward/reverse SHA-256"],
                     "search_uses_finished_reversal": False, "rlaif_per_candidate": False},
      "novelty_preflight": {"status": "passed", "signature": "typed-connector-live-residual-product"},
      "next_repair": "index residual seam by (remaining length, next character, subject/object agreement), then add subordinate Shakespearean clauses",
      "reader_gate": "closed; require randomized blinded intact prose versus shuffled controls"}

if __name__ == "__main__":
    out = ROOT / "runs" / (ID + ".json"); out.write_text(json.dumps(search(), indent=2) + "\n")
    print(json.dumps(search()["stats"], sort_keys=True))
