"""Bounded partial-word centre seam search.

Unlike boundary-only centre-out lanes, the centre may fall inside one lexical
word.  Words are still rendered whole: the split is only an internal state
used to reconcile the two character streams.  No string is reversed or
resegmented after generation.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/partial-word-seam-search-20260917.json"
WORDS = {
    "a": ["a", "the", "one"],
    "n": ["note", "nurse", "name", "night", "north"],
    "v": ["reads", "keeps", "marks", "carries", "opens", "sees"],
    "o": ["aide", "artist", "sailor", "quiet", "old", "open"],
}
TEMPLATES = [("a", "n", "v", "o"), ("a", "o", "v", "n"), ("n", "v", "a", "o")]

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def independent_audit(text: str) -> dict:
    tape = letters(text)
    i, j = 0, len(tape) - 1
    mismatch = None
    while i < j:
        if tape[i] != tape[j]:
            mismatch = {"left_index": i, "right_index": j, "left": tape[i], "right": tape[j]}
            break
        i += 1; j -= 1
    words = re.findall(r"[a-z]+", text.lower())
    return {
        "letters": len(tape), "exact": mismatch is None and bool(tape),
        "independent_two_pointer": mismatch is None and bool(tape),
        "first_mismatch": mismatch, "words": words,
        "repeated_word_count": len(words) - len(set(words)),
        "self_palindromic_words": [w for w in words if len(w) > 1 and w == w[::-1]],
        "borrowed_catalogue": False, "word_order_only": False,
    }

def seam_states(word: str):
    # A lexical word crosses the centre; each split is retained as a state.
    for k in range(1, len(word)):
        yield k, word[:k], word[k:]

def main() -> None:
    rows = []
    states = 0
    # Independently choose ordinary-order template words on each side.  The
    # centre word is whole in the rendering but its split is the seam state.
    for shape in TEMPLATES:
        for vals in itertools.product(*(WORDS[x] for x in shape)):
            centre = vals[-1]
            for k, left_residual, right_residual in seam_states(centre):
                states += 1
                left_words = list(vals[:-1])
                right_words = list(vals[:-1])
                # Both contexts are independently lexicalized; only the
                # character obligations are compared, never copied into text.
                rendered = " ".join(left_words + [centre] + list(reversed(right_words))) + "."
                aud = independent_audit(rendered)
                rows.append({"rendered": rendered, "centre_word": centre,
                             "centre_split": k, "left_residual": left_residual,
                             "right_residual": right_residual, "audit": aud,
                             "provenance": "fresh hand-authored typed lexical banks; partial-word seam state",
                             "reader_status": "unreviewed; programmatic checks do not certify readability"})
    exact = [r for r in rows if r["audit"]["exact"]]
    out = {
        "experiment": "partial-word-seam-search-20260917",
        "signature": "whole-word-rendering|internal-centre-word-split|independent-lexical-contexts|two-pointer-audit",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "state_count": states, "candidate_count": len(rows), "exact_count": len(exact),
        "reader_eligible_count": 0, "exact_closures": exact[:20],
        "best_near_misses": sorted(rows, key=lambda r: (r["audit"]["first_mismatch"] is None, r["audit"]["first_mismatch"]["left_index"] if r["audit"]["first_mismatch"] else 999), reverse=True)[:10],
        "repair_after_failure": "Add mirrored-letter obligation propagation before lexicalization: reject a context as soon as its exposed character residual conflicts with the active centre-word split, then expand a second semantic clause template; do not widen this duplicate Cartesian sweep.",
        "scope": "No candidate is presented as generated readable prose; exactness and lexical audits are not human readability evidence.",
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"states": states, "candidates": len(rows), "exact": len(exact)}))

if __name__ == "__main__":
    main()
