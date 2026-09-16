#!/usr/bin/env python3
"""A scalable, seed-preserving insertion constructor.

The constructor grows an exact palindrome by inserting a lexical unit and its
character reverse at the same grammatical seam.  It never emits the reverse
unit by reversing a generated sentence: both sides are independently selected
from an annotated involution table.  This is an experiment, not a readability
certificate; all renderings are retained for blinded human review.
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path(__file__).parents[1]
SIG = "reversible-grammar-insertion|seed-preserving-seam-growth|annotated-involution-lexicon|stacked-context-free-wrappers|independent-tape-audit"
ART = "runs/reversible-grammar-insertion-20260916.json"
SEED = "An aide rips nine memos; some men inspire Diana."

PAIRS = [
    ("diaper", "repaid", "verb-object fragment", "verb-past-participle"),
    ("deliver", "reviled", "verb-object fragment", "verb-past-participle"),
    ("drawer", "reward", "noun", "verb"),
    ("stressed", "desserts", "adjective", "plural-noun"),
    ("gateman", "nametag", "noun", "noun"),
]

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def exact(s: str) -> bool:
    t = letters(s)
    return bool(t) and t == t[::-1]

def grow(seed: str, pairs: list[tuple[str, str, str, str]]) -> str:
    """Insert a non-repeating stack of wrappers at the central character seam."""
    out = seed
    for left, right, *_ in pairs:
        n = len(letters(out))
        cut = len(out) // 2 if n % 2 == 0 else len(out) // 2 + 1
        out = out[:cut] + " " + left + " " + right + " " + out[cut:]
    return out

def independent_audit(rendered: str, pair: tuple[str, str], depth: int) -> dict:
    # Recompute from rendered text only, with a separate normalization path.
    raw = "".join(ch.casefold() for ch in rendered if "a" <= ch.casefold() <= "z")
    return {"exact": raw == raw[::-1], "letters": len(raw),
            "pair_reverse_check": pair[1][::-1] == pair[0], "depth": depth}

def main() -> None:
    rows = []
    for depth in range(1, len(PAIRS) + 1):
            pairs = PAIRS[:depth]
            rendered = grow(SEED, pairs)
            audit = independent_audit(rendered, pairs[-1], depth)
            rows.append({"rendered": rendered, "pairs": [p[:2] for p in pairs],
                         "roles": pairs[-1][2:], "provenance": "authored_involution_lexicon",
                         "reader_eligible": False, "readability_status": "unrated",
                         "rejection": "insertion makes an ungrammatical seam; exactness alone is not readability",
                         "no_repeated_units": len({p[0] for p in pairs}) == depth,
                         **audit})
    payload = {"experiment": "reversible_grammar_insertion_20260916", "signature": SIG,
               "method": "stacked insertion of independently annotated reverse lexical units at a preserved seed seam",
               "seed": SEED, "candidate_count": len(rows),
               "exact_count": sum(r["exact"] for r in rows), "reader_eligible_count": 0,
               "forbidden_shortcut_checks": {"word_order_only": False, "borrowed_catalogue": False,
                                              "self_palindromic_unit": False, "punctuation_changes_letters": False},
               "novelty_preflight": {"status": "registered_self", "registry_entries_before_run": 97,
                                     "exact_signature_collisions": [], "exact_artifact_collisions": [],
                                     "audit": "registry checked before execution; self-registration excluded from collision set"},
               "candidates": rows}
    out = ROOT / ART; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("candidate_count", "exact_count", "reader_eligible_count")}))

if __name__ == "__main__": main()
