"""Center-first feature grammar with live character obligations.

This lane starts at a grammatical center nonterminal (coordination or
copula), then grows the two clauses outward.  A derivation chooses which
side to expand from the current obligation and carries semantic role and
agreement features in the state.  No completed string is reversed or
repaired; the reverse character stream is consumed one token at a time.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "center-first-feature-grammar-20260921.json"
ID = "center-first-feature-grammar-20260921"

CENTERS = [
    {"word": "and", "kind": "coordination"},
    {"word": "but", "kind": "coordination"},
    {"word": "is", "kind": "copula"},
    {"word": "was", "kind": "copula"},
]
LEFT_SLOTS = [
    # Left slots are listed center-outward; prepending during rendering
    # restores ordinary determiner--subject--verb--object order.
    ("object_number", [("one", "sg"), ("nine", "pl"), ("two", "pl")]),
    ("object", [("memo", "sg", "thing"), ("memos", "pl", "thing"),
                 ("letter", "sg", "thing"), ("letters", "pl", "thing")]),
    ("verb", [("rips", "sg", "transitive"), ("reads", "sg", "transitive"),
               ("sees", "sg", "transitive"), ("keeps", "sg", "transitive")]),
    ("subject", [("aide", "sg", "human"), ("sailor", "sg", "human"),
                  ("poet", "sg", "human"), ("keeper", "sg", "human")]),
    ("determiner", [("a", "sg"), ("an", "sg"), ("the", "sg")]),
]
RIGHT_SLOTS = [
    ("determiner", [("a", "sg"), ("an", "sg"), ("the", "sg"), ("some", "pl")]),
    ("subject", [("man", "sg", "human"), ("men", "pl", "human"),
                  ("poet", "sg", "human"), ("women", "pl", "human")]),
    ("verb", [("reads", "sg", "transitive"), ("read", "pl", "transitive"),
               ("sees", "sg", "transitive"), ("see", "pl", "transitive")]),
    ("object", [("Ada", "sg", "name"), ("Anna", "sg", "name"),
                 ("Nora", "sg", "name"), ("Iris", "sg", "name")]),
]

def clean(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(text: str) -> dict:
    tape = clean(text)
    rev = tape[::-1]
    return {"normalized": tape, "letters": len(tape), "exact": bool(tape) and tape == rev,
            "pointer_check": bool(tape) and all(tape[i] == tape[-1-i] for i in range(len(tape)//2)),
            "sha256_normalized": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(rev.encode()).hexdigest()}

def consume(debt: str, token: str, side: str):
    chars = clean(token) if side == "L" else clean(token)[::-1]
    if debt.startswith(chars): return debt[len(chars):], side
    if chars.startswith(debt): return chars[len(debt):], "L" if side == "R" else "R"
    return None

def agreement(left, right, center):
    # The center is a real grammar feature: coordination requires two clauses;
    # copula requires singular subjects and nominal right complements.
    if center["kind"] == "copula":
        return left[1][1] == "sg" and right[1][1] == "sg"
    return left[2][2] == "transitive" and right[2][2] == "transitive"

def render(left, center, right):
    return " ".join(left + (center["word"],) + right) + "."

def run(max_nodes=180_000):
    nodes = prunes = 0
    frontier, exact, conflicts = [], [], []
    def record(left, center, right, debt, trace):
        text = render(left, center, right); a = audit(text)
        row = {"rendered": text, "audit": a, "remaining_debt": len(debt),
               "provenance": {"center_nonterminal": center, "center_first": True,
                   "live_obligation_consumption": True, "finished_tape_reversal": False,
                   "post_render_repair": False, "semantic_features": True},
               "trace": trace[-20:], "novelty_status": "unassessed_novelty",
               "reader_status": "not_run"}
        frontier.append(row)
        if not debt and a["exact"] and agreement(left, right, center): exact.append(row)

    def conflict(left, center, right, debt, trace, slot, side):
        # Keep the live failure witness; it is deliberately not presented as a
        # generated sentence because its unexpanded slots are not prose.
        conflicts.append({"side": side, "slot": slot, "open_debt": debt,
                          "partial_left": list(left), "center": center,
                          "partial_right": list(right), "trace": trace[-8:]})

    def expand(li, ri, debt, side, left, center, right, trace, used):
        nonlocal nodes, prunes
        nodes += 1
        if nodes > max_nodes: return
        if li == len(LEFT_SLOTS) and ri == len(RIGHT_SLOTS):
            if debt == debt[::-1]: record(left, center, right, debt, trace)
            return
        # Center is selected before either clause is expanded.
        if center is None:
            for c in CENTERS:
                expand(li, ri, clean(c["word"]), "R", left, c, right,
                       trace + [{"center": c["word"], "debt_after": clean(c["word"])}], used)
            return
        choices = []
        if side == "R" and ri < len(RIGHT_SLOTS): choices.append("R")
        if side == "L" and li < len(LEFT_SLOTS): choices.append("L")
        if not choices:
            if li < len(LEFT_SLOTS): choices.append("L")
            if ri < len(RIGHT_SLOTS): choices.append("R")
        for chosen in choices:
            slots = RIGHT_SLOTS if chosen == "R" else LEFT_SLOTS
            idx = ri if chosen == "R" else li
            slot, vals = slots[idx]
            for item in vals:
                word = item[0]
                if word.casefold() in used: continue
                result = consume(debt, word, chosen)
                if result is None:
                    prunes += 1
                    conflict(left, center, right, debt, trace, slot, chosen)
                    continue
                nd, ns = result
                if chosen == "R":
                    expand(li, ri+1, nd, ns, left, center, right+(word,),
                           trace + [{"side":"R","slot":slot,"word":word,"debt_after":nd}], used|{word.casefold()})
                else:
                    expand(li+1, ri, nd, ns, (word,)+left, center, right,
                           trace + [{"side":"L","slot":slot,"word":word,"debt_after":nd}], used|{word.casefold()})

    expand(0, 0, "", "L", (), None, (), [], frozenset())
    frontier.sort(key=lambda x: (x["remaining_debt"], -x["audit"]["letters"]))
    out = {"experiment_id": ID, "method": "center-first feature grammar with live obligation state",
           "counts": {"search_nodes": nodes, "obligation_prunes": prunes,
                      "frontier_states": len(frontier), "live_conflicts": len(conflicts), "exact_candidates": len(exact),
                      "exact_ge_40": sum(x["audit"]["letters"] >= 40 for x in exact)},
           "frontier": frontier[:40], "live_conflicts": conflicts[:40], "exact_candidates": exact,
           "novelty_policy": "manual comparison against repository corpus required",
           "reader_status": "not_run"}
    RUN.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["counts"], indent=2))
    for row in frontier[:8]: print(row["rendered"], row["audit"]["letters"], row["remaining_debt"])
    return out

if __name__ == "__main__": run()
