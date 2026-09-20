"""Bilateral lexicalized grammar CSP for ordinary two-clause sentences.

The left clause is expanded in ordinary order while the right clause is
expanded from its right edge.  Each selected right word contributes its
letters in reverse order to the live character residual, so no completed tape
is built and reversed.  Both sides are grammar derivations; the final output
is only accepted after the independent pointer/SHA audit and shortcut filters.
"""
from __future__ import annotations

import json
from pathlib import Path

from forward_lexicalized_grammar_20260920 import (
    ATOMIC,
    GRAMMAR,
    Word,
    admission_ok,
    independent_audit,
    letters,
    render_path,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/bilateral-grammar-csp-20260920.json"


def _consume(left_residual: str, right_residual: str) -> tuple[str, str] | None:
    """Consume equal available prefixes of the two live residual streams."""
    k = 0
    limit = min(len(left_residual), len(right_residual))
    while k < limit and left_residual[k] == right_residual[k]:
        k += 1
    if k < limit:
        return None
    return left_residual[k:], right_residual[k:]


def palindromic_residual(left_residual: str, right_residual: str) -> bool:
    """Close complete derivations when their remaining middle is symmetric.

    The right residual is already in reverse reading order. If L = reverse(R)
    + C, then L + R is a palindrome exactly when C is a palindrome. Requiring
    C to be empty incorrectly fixes the clause seam at an even-length center.
    This test applies only after both grammar derivations are complete.
    """
    residual = _consume(left_residual, right_residual)
    if residual is None:
        return False
    center = residual[0] or residual[1]
    return all(center[i] == center[-1-i] for i in range(len(center) // 2))


def bilateral_grammar_csp(
    lexicon=ATOMIC,
    max_words: int = 10,
    max_nodes: int = 100_000,
    grammar=None,
    left_symbols=None,
    right_symbols=None,
):
    """Search two independent grammar clauses while matching characters live.

    ``left`` is the first clause in normal order.  ``right_rev`` is the second
    clause accumulated from its final word toward its first word.  Thus the
    right side's reversed character stream is available at every transition,
    while ``reversed(right_rev)`` is still an ordinary grammatical clause at
    extraction time.
    """
    grammar = grammar or GRAMMAR
    by_pos = {}
    for word in lexicon:
        if letters(word.text):
            by_pos.setdefault(word.pos, []).append(word)

    found = []
    stats = {"nodes": 0, "pruned": 0, "complete": 0,
             "center_rejections": 0, "nonempty_center_closures": 0}

    def search(lsymbols, rsymbols, left, right_rev, lres="", rres=""):
        if stats["nodes"] >= max_nodes:
            return
        stats["nodes"] += 1
        if not lsymbols and not rsymbols:
            stats["complete"] += 1
            if not left or not right_rev:
                return
            if not palindromic_residual(lres, rres):
                stats["center_rejections"] += 1
                return
            if lres or rres:
                stats["nonempty_center_closures"] += 1
            words = left + list(reversed(right_rev))
            if not admission_ok(words):
                return
            text = " ".join(words)
            audit = independent_audit(text)
            if audit["exact"]:
                found.append({
                    "length": audit["letters"],
                    "rendered": render_path(text, lexicon),
                    "audit": audit,
                    "words": words,
                    "center_residual": lres or rres,
                    "left_clause_letters": sum(len(letters(w)) for w in left),
                    "right_clause_letters": sum(len(letters(w)) for w in right_rev),
                    "provenance": {
                        "left_clause_grammar": True,
                        "right_clause_reverse_expansion": True,
                        "live_shared_character_residual": True,
                        "free_word_boundaries": True,
                        "center_inside_word_allowed": True,
                        "post_hoc_repair": False,
                        "finished_tape_reversal": False,
                        "word_order_mirroring": False,
                        "repeated_units": False,
                        "catalogue_text": False,
                    },
                })
            return
        if len(left) + len(right_rev) >= max_words:
            return

        # Expand the left grammar from its front.  Grammar expansion has no
        # character effect, so it does not perturb the residual invariant.
        if lsymbols and lsymbols[0] in grammar:
            for production in grammar[lsymbols[0]]:
                search(list(production) + list(lsymbols[1:]), rsymbols,
                       left, right_rev, lres, rres)
            return

        # Expand the right grammar from its back.  Replacing the final
        # nonterminal by its forward production makes its final terminal the
        # next symbol selected, i.e. the right edge of the clause.
        if rsymbols and rsymbols[-1] in grammar:
            for production in grammar[rsymbols[-1]]:
                search(lsymbols, list(rsymbols[:-1]) + list(production),
                       left, right_rev, lres, rres)
            return

        # Once both grammar frontiers expose terminals, choose one lexical
        # edge on each side in the same transition.  This is the key
        # bilateral operation: the character residual is tested before either
        # side can run far ahead of the other.
        if lsymbols and rsymbols and lsymbols[0] not in grammar and rsymbols[-1] not in grammar:
            for left_word in by_pos.get(lsymbols[0], ()):
                for right_word in by_pos.get(rsymbols[-1], ()):
                    residual = _consume(
                        lres + letters(left_word.text),
                        rres + letters(right_word.text)[::-1],
                    )
                    if residual is None:
                        stats["pruned"] += 1
                        continue
                    search(
                        lsymbols[1:], rsymbols[:-1],
                        left + [left_word.text],
                        right_rev + [right_word.text],
                        *residual,
                    )
            return

        if lsymbols:
            for word in by_pos.get(lsymbols[0], ()):
                residual = _consume(lres + letters(word.text), rres)
                if residual is None:
                    stats["pruned"] += 1
                    continue
                search(lsymbols[1:], rsymbols, left + [word.text], right_rev,
                       *residual)
            return

        if rsymbols:
            for word in by_pos.get(rsymbols[-1], ()):
                residual = _consume(lres, rres + letters(word.text)[::-1])
                if residual is None:
                    stats["pruned"] += 1
                    continue
                search(lsymbols, rsymbols[:-1], left, right_rev + [word.text],
                       *residual)

    search(list(left_symbols or ("CLAUSE",)), list(right_symbols or ("CLAUSE",)), [], [])
    stats["status"] = "timeout" if stats["nodes"] >= max_nodes else (
        "SAT" if found else "UNSAT"
    )
    return {
        "method": "bilateral lexicalized grammar CSP with reverse-edge expansion",
        "paths": sorted(found, key=lambda row: (row["length"], row["rendered"])),
        "stats": stats,
        "provenance": {
            "grammar": "independent CLAUSE derivation on each side; S = CLAUSE CLAUSE",
            "lexicon": "caller-supplied lexical entries with POS features",
            "character_equation": "shared prefix residuals consume x[i] = x[N-1-i] online",
            "next_construction": "add typed adjunct/relative productions only after this bilateral baseline is audited",
        },
    }


if __name__ == "__main__":
    result = bilateral_grammar_csp(
        tuple(word for word in ATOMIC if word.text in
              {"an", "aide", "rips", "nine", "memos", "some", "men", "inspire", "diana"}),
        max_nodes=100_000,
    )
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
    for path in result["paths"]:
        print(path["rendered"])
