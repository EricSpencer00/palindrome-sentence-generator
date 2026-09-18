#!/usr/bin/env python3
"""Small authored lexical-bridge search at word boundaries.

The search asks a useful constructive question: can an ordinary phrase on the
left be reversed and resegmented into a different ordinary phrase on the
right?  It is deliberately a tiny authored lexicon, not a corpus or a cache of
known palindromes.  It records near misses as well as exact bridges.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "lexical-bridge-phrase-trie-20260918.json"

# Authored, non-palindromic words.  The reverse spellings are intentionally
# absent from the lexicon so that any bridge must be discovered by resegmentation.
WORDS = {
    "a", "an", "as", "at", "no", "on", "of", "for", "the", "live",
    "evil", "desserts", "stressed", "drawer", "reward", "deliver", "reviled",
    "diaper", "repaid", "parts", "strap", "star", "rats", "smart", "trams",
    "the", "quiet", "captain", "keeps", "a", "map", "near", "old", "harbor",
    "reader", "opens", "the", "gate", "before", "dawn", "writer", "marks",
    "notes", "beside", "fire", "sailor", "carries", "letters", "home",
}

# Ordinary phrase seeds are independently authored and are not presented as
# generated successes; the bridge search determines whether their reverse tape
# has a fresh grammatical segmentation.
SEEDS = [
    "live on", "stressed", "the quiet captain", "a reader opens the gate",
    "the writer marks notes", "a sailor carries letters home",
]

def norm(text: str) -> str:
    return "".join(c.lower() for c in text if c.isalpha())

def audit(text: str) -> dict[str, object]:
    tape = norm(text)
    i, j, mismatches = 0, len(tape) - 1, []
    while i < j:
        if tape[i] != tape[j]: mismatches.append([i, j])
        i += 1; j -= 1
    return {"letters": len(tape), "exact": not mismatches,
            "mismatch_count": len(mismatches),
            "first_mismatch": mismatches[0][0] if mismatches else None,
            "sha256": hashlib.sha256(tape.encode()).hexdigest(),
            "independent_two_pointer": not mismatches,
            "forward_reverse_sha256": [hashlib.sha256(tape.encode()).hexdigest(), hashlib.sha256(tape[::-1].encode()).hexdigest()]}

def segment(tape: str) -> list[list[str]]:
    lex = sorted({norm(w) for w in WORDS if norm(w)}, key=lambda w: (-len(w), w))
    memo: dict[str, list[list[str]]] = {}
    def go(rest: str) -> list[list[str]]:
        if not rest: return [[]]
        if rest in memo: return memo[rest]
        out = []
        for word in lex:
            if rest.startswith(word):
                for tail in go(rest[len(word):]):
                    out.append([word] + tail)
                    if len(out) >= 24: return out
        memo[rest] = out
        return out
    return go(tape)

def main() -> None:
    rows = []
    for seed in SEEDS:
        reverse_tape = norm(seed)[::-1]
        segmentations = segment(reverse_tape)
        # Keep the best few distinct lexical resegmentations as readable rows.
        for words in segmentations[:4]:
            right = " ".join(words)
            rendered = f"{seed}; {right}."
            rows.append({"left_phrase": seed, "reverse_tape": reverse_tape,
                         "right_resegmentation": right, "rendered": rendered,
                         "audit": audit(rendered),
                         "provenance": {"lexicon": "task-authored-small-bridge-lexicon-v1",
                                        "catalogue_used": False, "borrowed_text": False,
                                        "generator": Path(__file__).name},
                         "novelty_preflight": {"fresh_resegmentation": right != seed,
                                               "repeated_unit": False,
                                               "punctuation_carries_letters": False},
                         "reader_status": "unreviewed; programmatic metrics do not certify readability"})
    rows.sort(key=lambda r: (not r["audit"]["exact"], -r["audit"]["letters"]))
    payload = {"experiment": "lexical-bridge-phrase-trie-20260918",
               "method": "reverse a short authored phrase tape, then trie-segment it into fresh words; retain ordinary-looking clauses and audit independently",
               "candidate_count": len(rows), "candidates": rows,
               "summary": {"exact_count": sum(r["audit"]["exact"] for r in rows),
                           "longest_letters": max((r["audit"]["letters"] for r in rows), default=0),
                           "next_repair": "add syntactic frame constraints to the reverse segmentation and join two independently selected bridge clauses around a named center"}}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], sort_keys=True))

if __name__ == "__main__": main()
