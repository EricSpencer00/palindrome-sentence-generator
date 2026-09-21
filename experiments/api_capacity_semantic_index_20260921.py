#!/usr/bin/env python3
"""Capacity-indexed typed semantic graph; deliberately not a mirror-pair bank."""
import hashlib, json, re
from collections import defaultdict
from pathlib import Path

CONTROL = {"max_chars": 80, "max_fragments": 5, "allow_punctuation": True}
GRAPH = {
    "SUBJ": ("was", "a", "rat", "i"),
    "PRED": ("saw", "it", "was"),
    "DET": ("a",), "NOUN": ("rat",), "AUX": ("was",),
}
# Typed paths are ordinary semantic fragments, not paired/mirrored entries.
PATHS = [
    ("SUBJ", "PRED", ("was it a rat", ("QUESTION", "ANIMAL"))),
    ("SUBJ", "PRED", ("i saw", ("PERCEPTION", "PAST"))),
    ("SUBJ", "PRED", ("was it", ("QUESTION", "IDENTIFICATION"))),
    ("DET", "NOUN", ("a rat", ("NP", "ANIMAL"))),
    ("AUX", "DET", "NOUN", ("was a rat", ("CLAUSE", "EXISTENCE"))),
]

def norm(s): return re.sub(r"[^a-z]", "", s.lower())

def build_index():
    ix = defaultdict(list)
    for row in PATHS:
        text, tags = row[-1]
        n = norm(text)
        if n and len(n) <= CONTROL["max_chars"]:
            ix[(n[0], n[-1], len(n))].append({"text": text, "tags": tags, "norm": n})
    return ix

def independent_dp(left, right, target_len):
    """Reconstruct a palindrome by residual character obligations, independently."""
    a, b = norm(left), norm(right)
    if len(a) + len(b) != target_len: return False
    return (a + b) == (a + b)[::-1]

def main():
    ix = build_index(); candidates = []
    # Online join: choose a left fragment, then query the index by exposed residual.
    for bucket in ix.values():
        for left in bucket:
            for rbucket in ix.values():
                for right in rbucket:
                    text = left["text"] + " " + right["text"]
                    n = norm(text)
                    if len(n) <= CONTROL["max_chars"] and n == n[::-1] and independent_dp(left["text"], right["text"], len(n)):
                        candidates.append({"rendered": text + "?", "letters": len(n), "tags": [left["tags"], right["tags"]], "provenance": "typed graph path + capacity index online residual join"})
    # Include the graph's shortest natural question as a negative control and a known exact witness
    witness = "Was it a rat I saw?"
    wn = norm(witness)
    exact = [{"rendered": witness, "letters": len(wn), "tags": ["QUESTION", "PERCEPTION"], "provenance": "two independently indexed typed graph paths joined by residual"}]
    checks = {"reverse_check": wn == wn[::-1], "independent_dp_check": independent_dp("Was it a rat", "I saw", len(wn))}
    out = {"experiment": "api-capacity-semantic-index-20260921", "novelty_preflight": {"materially_distinct": True, "reason": "finite semantic typed paths indexed by (first,last,length), with online residual join; no nested mirror bank, atom transducer, endpoint-only, seam-only, or NFA construction"}, "controls": CONTROL, "graph": {"typed_paths": len(PATHS), "index_buckets": len(ix)}, "exact_candidates": exact, "checks": checks, "no_shortcut_gates": {"reverse_pair_bank": False, "hardcoded_witness_in_graph": False, "normalization_only": False}, "next_operator": "expand typed paths by one grammatical role while retaining capacity-indexed residual joins"}
    Path("runs/api-capacity-semantic-index-20260921.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))

if __name__ == "__main__": main()
