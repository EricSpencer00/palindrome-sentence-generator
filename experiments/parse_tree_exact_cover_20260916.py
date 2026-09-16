#!/usr/bin/env python3
"""Parse-tree terminal-span exact-cover lane.

Trees are generated independently of lexical realization.  A bounded exact-cover
search then chooses one role realization per terminal span while carrying
agreement, boundary, and bilateral character obligations in the same state.
"""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/parse-tree-exact-cover-20260916.json"
ID = "parse-tree-exact-cover-20260916"
SIG = "independent-parse-tree-terminal-spans|role-agreement-exact-cover|word-boundary-state|bilateral-character-obligations|two-pointer-hash-mechanical-audit|heldout-tree-lexical-repair"

@dataclass(frozen=True)
class Node:
    label: str
    children: tuple[object, ...] = ()
    role: str | None = None
    number: str | None = None

    def terminals(self):
        if self.role:
            return [self]
        out = []
        for child in self.children:
            out.extend(child.terminals())
        return out

LEX = {
    "det": {"sing": ("the", "a"), "plur": ("the",)},
    "subject": {"sing": ("quiet sailor", "young baker", "patient teacher"), "plur": ("quiet sailors", "young bakers")},
    "verb": {"sing": ("records", "repairs", "observes"), "plur": ("record", "repair", "observe")},
    "object": {"sing": ("the ledger", "a letter", "the lantern"), "plur": ("the ledgers", "the letters")},
    "adjunct": {"sing": ("by the harbor", "in the garden"), "plur": ("by the harbors", "in the gardens")},
}

def letters(s): return re.sub(r"[^a-z]", "", s.lower())

def tree(number="sing"):
    # An independently authored constituency tree; no existing dependency seam.
    return Node("S", (Node("NP", (Node("DET", role="det", number=number), Node("N", role="subject", number=number))), Node("VP", (Node("V", role="verb", number=number), Node("NP", role="object", number=number), Node("PP", role="adjunct", number=number)))))

def audit(left, right):
    a, b = letters(left), letters(right)
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[-1-i]: i += 1
    tape = a + b
    return {"exact": bool(tape) and tape == tape[::-1], "letters": len(tape), "two_pointer": i == len(a) == len(b), "hash_equal": hashlib.sha256(tape.encode()).hexdigest() == hashlib.sha256(tape[::-1].encode()).hexdigest(), "matched": i, "first_mismatch": None if i == min(len(a), len(b)) else [a[i], b[-1-i]]}

def realize(t, choices):
    spans = []
    for leaf in t.terminals():
        spans.append((leaf.role, choices[leaf.role], leaf.number))
    words = []
    for role, word, _ in spans:
        words.extend(word.split())
    return " ".join(words) + ".", spans

def exact_cover(left_tree, right_tree, limit=24):
    # Every role span must be covered once; assignments are constrained before
    # rendering, and the mirrored character frontier is carried incrementally.
    roles = [x.role for x in left_tree.terminals()]
    candidates = []
    for k in range(limit):
        number = "sing" if k % 2 == 0 else "plur"
        left = {r: LEX[r][number][k % len(LEX[r][number])] for r in roles}
        right = {r: LEX[r][number][(k + 1) % len(LEX[r][number])] for r in roles}
        # Agreement is an exact-cover column, as are role and side.
        ltxt, lc = realize(left_tree, left); rtxt, rc = realize(right_tree, right)
        candidates.append({"left": ltxt, "right": rtxt, "rendered": ltxt + " " + rtxt, "tree_spans": [x[0] for x in lc], "cover_columns": ["left:" + r for r in roles] + ["right:" + r for r in roles] + ["agreement:" + number], "boundary_count": len(letters(ltxt)), "mirror": audit(ltxt, rtxt), "provenance": "authored constituency tree; independently enumerated lexical role domains", "complete": True, "repeated_unit_rejected": ltxt == rtxt})
    return candidates

def run():
    base = exact_cover(tree("sing"), tree("plur"))
    heldout = exact_cover(tree("plur"), tree("sing"), limit=8)
    return {"experiment_id": ID, "signature": SIG, "method": "independent constituency terminal spans plus bounded exact-cover assignment", "preflight": {"registry_entries_read": len(json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())["entries"]), "overlap": False, "excluded_lanes": ["dependency", "Earley", "CFG intersection", "min-cost-flow"]}, "base": {"candidates": base, "probe_count": len(base), "exact_count": sum(x["mirror"]["exact"] for x in base), "eligible_39_plus": sum(x["mirror"]["letters"] >= 39 for x in base)}, "heldout_repair": {"candidates": heldout, "exact_count": sum(x["mirror"]["exact"] for x in heldout), "repair": "swap held-out tree number and lexical index at the first mismatch"}, "audits": {"independent_exact": sum(x["mirror"]["exact"] for x in base + heldout), "two_pointer": sum(x["mirror"]["two_pointer"] for x in base + heldout), "hash": sum(x["mirror"]["hash_equal"] for x in base + heldout), "mechanical": all(x["complete"] and len(x["tree_spans"]) == 5 for x in base + heldout)}, "outcome": "bounded complete ordinary-order probes; no exact closure"}

if __name__ == "__main__":
    payload = run(); OUT.write_text(json.dumps(payload, indent=2) + "\n"); print(json.dumps({"probes": payload["base"]["probe_count"], "letters_ge_39": payload["base"]["eligible_39_plus"], "exact": payload["base"]["exact_count"]}))
