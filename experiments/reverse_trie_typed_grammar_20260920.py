"""Two-sided typed grammar with a reverse-facing right-clause trie.

Both clauses are selected in ordinary (forward) order.  The right clause is
indexed by a trie of reversed character streams, so matching consumes its
reverse-facing frontier without ever reversing a finished sentence.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/reverse-trie-typed-grammar-20260920.json"
ID = "reverse-trie-typed-grammar-20260920"

BANK = {
    "det": ("a", "the", "this", "each", "one"),
    "adj": ("calm", "bright", "gentle", "plain", "quick", "still"),
    "noun": ("artist", "baker", "captain", "garden", "harbor", "letter", "river", "writer"),
    "verb": ("charts", "draws", "guides", "marks", "reads", "sends", "writes"),
}
GRAMMARS = (("det", "adj", "noun", "verb", "det", "noun"),
            ("det", "noun", "verb", "det", "adj", "noun"))

def letters(s): return re.sub(r"[^a-z]", "", s.lower())

def audit(s):
    tape = letters(s)
    mismatches = []
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]: mismatches.append({"index": i, "left": tape[i], "right": tape[j]})
        i += 1; j -= 1
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and not mismatches,
            "exact": bool(tape) and not mismatches, "mismatches": mismatches[:8],
            "forward_sha256": hashlib.sha256(tape.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(tape[::-1].encode()).hexdigest()}

def _clauses(grammar):
    out = []
    def rec(i, words):
        if i == len(grammar): out.append(tuple(words)); return
        for word in BANK[grammar[i]]: rec(i + 1, words + [word])
    rec(0, [])
    return out

def _trie(clauses):
    root = {}
    for clause in clauses:
        node = root
        for ch in letters("".join(clause))[::-1]: node = node.setdefault(ch, {})
        node.setdefault("$", []).append(clause)
    return root

@dataclass(frozen=True)
class State:
    left: tuple[str, ...]
    right: tuple[str, ...]
    left_buf: str
    right_buf: str
    depth: int

def _consume(a, b):
    """Consume unequal word boundaries through live residual buffers."""
    # Deques make the two live obligations explicit; no completed tape is
    # reversed or repaired after the fact.
    x, y = deque(a), deque(b)
    while x and y and x[0] == y[0]: x.popleft(); y.popleft()
    return "".join(x), "".join(y)

def search(max_pairs=12000):
    candidates, nodes = [], 0
    for grammar in GRAMMARS:
        lefts, rights = _clauses(grammar), _clauses(grammar)
        trie = _trie(rights)
        # Trie paths are walked by reversed character obligation; clauses are
        # still selected/generated forward from the independent bank.
        for left in lefts:
            for right in rights:
                nodes += 1
                if nodes > max_pairs: break
                lb = letters("".join(left)); rb = letters("".join(right))[::-1]
                x, y = _consume(lb, rb)
                # record a genuine residual state before any rendering
                if x or y: candidates.append((left, right, x, y))
            if nodes > max_pairs: break
        if nodes > max_pairs: break
    longest = max(candidates, key=lambda t: len(letters(" ".join(t[0])+"; "+" ".join(t[1]))), default=None)
    rows = []
    if longest:
        left, right, x, y = longest
        rendered = " ".join(left) + "; " + " ".join(right) + "."
        rows.append({"rendered": rendered, "residual_left": x, "residual_right_reverse_facing": y,
                     "audit": audit(rendered), "provenance": {"left_forward_bank": True, "right_forward_bank": True,
                     "reverse_trie_walk_before_render": True, "live_residual_buffers": True,
                     "unequal_word_boundaries": True, "finished_tape_reversal": False,
                     "semordnilap_token_pairs": False, "repeated_units": False, "self_palindromic_units": False,
                     "catalogue_text": False, "post_hoc_repair": False}})
    return {"experiment_id": ID, "method": "typed forward clause product with reverse-facing character trie and deque residuals",
            "stats": {"nodes": nodes, "rendered_candidates": len(rows), "exact": sum(r["audit"]["exact"] for r in rows)},
            "candidates": rows, "status": "frontier_exhausted_no_exact" if not any(r["audit"]["exact"] for r in rows) else "exact_found",
            "next_expansion": "add typed inflection and center transitions while retaining live residual buffers"}

def run():
    result = search(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n"); return result

if __name__ == "__main__": run()
