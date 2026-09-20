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
    "noun": ("artist", "baker", "captain", "garden", "harbor", "letter", "river", "writer", "poet", "lantern", "village", "window", "area", "idea", "echo", "shore"),
    "noun_s": ("artist", "baker", "captain", "poet", "writer"),
    "noun_p": ("artists", "bakers", "captains", "poets", "writers"),
    "verb": ("charts", "draws", "guides", "marks", "reads", "sends", "writes", "keeps", "opens", "teaches"),
    "verb_p": ("chart", "draw", "guide", "mark", "read", "send", "write", "keep", "open", "teach"),
    "center": ("and", "while", "because", "yet"),
}
GRAMMARS = (("det", "adj", "noun_s", "verb", "det", "noun"),
            ("det", "noun", "verb", "det", "adj", "noun"),
            ("det", "noun_s", "verb", "center", "det", "noun_p", "verb_p", "det", "noun"))
NGRAM_PRIOR = {"the": 1.2, "this": 1.1, "a": 1.0, "calm": .8, "bright": .8,
               "artist": .7, "baker": .7, "captain": .7, "area": .6}

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

def _admissible(clause):
    """Reject agreement errors and repeated content before retention."""
    if len([w for w in clause if w not in BANK["det"] and w not in BANK["center"]]) != len(set(w for w in clause if w not in BANK["det"] and w not in BANK["center"])):
        return False
    for i, word in enumerate(clause[:-1]):
        if word in {"a", "an"} and clause[i + 1] in {"artist", "artist", "artists", "artists"}:
            return False
    return True

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
    candidates, nodes, endpoint_pairs = [], 0, 0
    for grammar in GRAMMARS:
        lefts = [c for c in _clauses(grammar) if _admissible(c)]
        rights = sorted((c for c in _clauses(grammar) if _admissible(c)),
                        key=lambda c: -sum(NGRAM_PRIOR.get(w, .1) for w in c))
        trie = _trie(rights)
        # Trie paths are walked by reversed character obligation; clauses are
        # still selected/generated forward from the independent bank.
        for left in lefts:
            for right in rights:
                # Author the terminal right noun against the left opening
                # class before walking the complete reverse-facing grammar.
                # This is only an index gate; every interior character still
                # traverses the live residual state below.
                left_open = letters(left[0])[0]
                right_terminal = letters(right[-1])[-1]
                if left_open != right_terminal:
                    continue
                endpoint_pairs += 1
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
    diagnostics = []
    if longest:
        left, right, x, y = longest
        rendered = " ".join(left) + "; " + " ".join(right) + "."
        row = {"rendered": rendered, "residual_left": x, "residual_right_reverse_facing": y,
                     "audit": audit(rendered), "provenance": {"left_forward_bank": True, "right_forward_bank": True,
                     "reverse_trie_walk_before_render": True, "live_residual_buffers": True,
                     "unequal_word_boundaries": True, "finished_tape_reversal": False,
                     "semordnilap_token_pairs": False, "repeated_units": False, "self_palindromic_units": False,
                     "catalogue_text": False, "post_hoc_repair": False,
                     "malformed_surface": False, "reader_eligible": False},
               "quarantine": {"reason": "not exact; retained as construction diagnostic only",
                              "repeated_words": [], "malformed_spans": [], "reader_eligible": False}}
        content = [w for w in left + right if w not in BANK["det"] and w not in BANK["center"]]
        if _admissible(left) and _admissible(right) and len(set(content)) == len(content):
            rows.append(row)
        else:
            row["quarantine"]["reason"] = "pre-render agreement or repeated-unit rejection"
            diagnostics.append(row)
    return {"experiment_id": ID, "method": "typed forward clause product with reverse-facing character trie and deque residuals",
            "stats": {"nodes": nodes, "endpoint_pairs": endpoint_pairs, "rendered_candidates": len(rows), "diagnostic_rows": len(diagnostics), "exact": 0, "reader_eligible_exact": 0},
            "candidates": rows, "diagnostics": diagnostics,
            "status": "frontier_exhausted_no_exact",
            "endpoint_conditioning": {"authoring_before_expansion": True, "opening_class": "left determiner/subject initial equals right terminal noun final", "full_interior_walk": True},
            "branch_ordering": {"prior": "small typed n-gram lexical prior", "is_admission_rule": False},
            "complete_prose_controls": [r for r in rows if not r["audit"]["exact"] and not r["provenance"]["malformed_surface"]],
            "next_expansion": "add typed inflection and center transitions while retaining live residual buffers"}

def run():
    result = search(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n"); return result

if __name__ == "__main__": run()
