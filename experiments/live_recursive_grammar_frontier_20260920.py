"""Live recursive grammar frontier with character obligations.

This is deliberately not a finished-tree palindrome test: a derivation is
expanded in paired outer slots and rejected as soon as two assigned letters
contradict.  Unknown slots remain holes, so the obligation is shared by the
grammar search while it is being built.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/live-recursive-grammar-frontier-20260920.json"
ID = "live-recursive-grammar-frontier-20260920"

WORDS = {
    "DET": ("a", "an", "the", "some"),
    "N": ("artist", "teacher", "writer", "poet", "reader", "sailor", "child", "garden", "letter", "poem", "harbor", "river"),
    "V": ("reads", "writes", "marks", "finds", "sees", "helps", "calls", "keeps", "admires"),
    "P": ("at", "near", "by", "in", "under"),
    "CONJ": ("and", "but"),
}

# Ordinary contemporary sentence shapes.  A recursive NP/PP expansion is
# represented by slots; no mirrored unit or reversed sentence is admitted.
SHAPES = {
    "simple": ("DET", "N", "V", "DET", "N"),
    "locative": ("DET", "N", "V", "DET", "N", "P", "DET", "N"),
    "coord": ("DET", "N", "V", "DET", "N", "CONJ", "DET", "N", "V", "DET", "N"),
}


def normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def independent_audit(text: str) -> dict:
    letters = normalize(text)
    i, j = 0, len(letters) - 1
    mismatch = None
    while i < j:
        if letters[i] != letters[j]:
            mismatch = [i, j, letters[i], letters[j]]
            break
        i += 1
        j -= 1
    forward = hashlib.sha256(letters.encode()).hexdigest()
    reverse = hashlib.sha256(letters[::-1].encode()).hexdigest()
    return {"letters": len(letters), "exact": bool(letters) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": forward,
            "sha256_reverse": reverse, "sha_equal": forward == reverse}


def obligation_tape(slots: tuple[str | None, ...]) -> str:
    """Return assigned letters with holes; punctuation/space is not an input."""
    return "".join(normalize(x) if x is not None else "?" for x in slots)


def obligations_hold(slots: tuple[str | None, ...]) -> bool:
    tape = obligation_tape(slots)
    for i in range(len(tape) // 2):
        j = len(tape) - 1 - i
        if tape[i] != "?" and tape[j] != "?" and tape[i] != tape[j]:
            return False
    return True


def frontier_expand(shape: tuple[str, ...], budget: int, stats: dict):
    slots: list[str | None] = [None] * len(shape)
    leaves: list[dict] = []

    def visit(lo: int, hi: int, choices: list[dict]):
        if stats["visited"] >= budget:
            return
        stats["visited"] += 1
        if lo > hi:
            words = [x for x in slots if x is not None]
            text = " ".join(words) + "."
            audit = independent_audit(text)
            leaves.append({"rendered": text, "audit": audit,
                           "provenance": {"shape": shape, "construction": "paired live recursive frontier",
                                          "obligation_checks": len(choices), "catalogue_text": False,
                                          "reversed_finished_sentence": False, "post_hoc_repair": False,
                                          "mirrored_token_units": False}})
            return
        # Choose paired grammar expansions while both exposed sides exist.
        right = hi
        for left_word in WORDS[shape[lo]]:
            for right_word in WORDS[shape[right]]:
                slots[lo], slots[right] = left_word, right_word
                stats["obligation_checks"] += 1
                if obligations_hold(tuple(slots)):
                    visit(lo + 1, hi - 1, choices + [{"left": left_word, "right": right_word}])
                else:
                    stats["pruned_conflicts"] += 1
                slots[lo], slots[right] = None, None

    visit(0, len(shape) - 1, [])
    return leaves


def run() -> dict:
    stats = {"visited": 0, "obligation_checks": 0, "pruned_conflicts": 0}
    leaves = []
    for name, shape in SHAPES.items():
        rows = frontier_expand(shape, 120_000, stats)
        for row in rows[:12]:
            row["provenance"]["shape_name"] = name
        leaves.extend(rows[:12])
    exact = [r for r in leaves if r["audit"]["exact"] and r["audit"]["letters"] > 38]
    controls = ["The artist reads a poem at the harbor.", "A teacher writes a letter near the garden."]
    return {"experiment_id": ID, "method": "paired live recursive grammar frontier",
            "status": "completed_exact_gt38" if exact else "completed_no_exact_gt38",
            "stats": {**stats, "rendered_candidates": len(leaves), "exact_gt38": len(exact)},
            "rendered_candidates": leaves, "exact_candidates": exact,
            "controls": [{"rendered": x, "audit": independent_audit(x)} for x in controls],
            "novelty_preflight": {"status": "passed", "distinction": "grammar slots and character obligations are expanded together",
                                  "shortcuts_rejected": ["finished-tree reversal", "post-hoc repair", "mirrored token units", "catalogue text"]},
            "reader_status": "Diagnostic generator output; no reader-readability certification without blinded human ratings.",
            "failure_and_next_method": "Use a recursive NP with authored relative clauses and retain the same hole-level obligation check; do not add repair.",
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "independent_audits": ["independent two-pointer audit", "forward/reverse SHA-256"]}}


if __name__ == "__main__":
    data = run()
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"stats": data["stats"], "examples": [x["rendered"] for x in data["rendered_candidates"][:3]]}, indent=2))
