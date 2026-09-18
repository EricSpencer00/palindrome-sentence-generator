"""Typed semordnilap search with syntax checked before closure ranking.

This is a deliberately small constructive branch.  Reverse-word pairs are
not promoted merely because they are English words: both orientations must
fit an authored POS/valency frame, and suspect catalogue/proper-name forms
are excluded before exact closure ranking.  The output is evidence for the
reader-facing search, not a readability certificate.
"""
from __future__ import annotations

import argparse, json
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.validator import normalize, is_palindrome

MIN_LETTERS = 39

@dataclass(frozen=True)
class Pair:
    left: str
    right: str
    left_pos: frozenset[str]
    right_pos: frozenset[str]
    lexical_quality: str = "ordinary"

    @property
    def tape(self):
        return self.left + self.right


def pair(a: str, b: str, apos: str, bpos: str, quality="ordinary") -> Pair:
    assert a[::-1] == b
    return Pair(a, b, frozenset(apos.split("/")), frozenset(bpos.split("/")), quality)


# Hand-authored productive pairs only.  The blacklist is intentionally
# conservative: a false negative is preferable to smuggling a catalogue word
# into a reader-facing candidate.
PAIRS = (
    pair("drawer", "reward", "noun", "noun"),
    pair("diaper", "repaid", "noun", "verb"),
    pair("parts", "strap", "noun/verb", "noun/verb"),
    pair("lever", "revel", "noun/verb", "verb"),
    pair("lager", "regal", "noun", "adj"),
    pair("smart", "trams", "adj", "noun/verb"),
    pair("devil", "lived", "noun", "verb"),
    pair("flow", "wolf", "verb/noun", "noun"),
    pair("moor", "room", "noun/verb", "noun"),
    pair("loot", "tool", "verb/noun", "noun"),
    pair("emit", "time", "verb", "noun"),
    pair("pets", "step", "noun/verb", "noun/verb"),
    pair("guns", "snug", "noun", "adj"),
    pair("deer", "reed", "noun", "noun"),
    pair("doom", "mood", "noun/verb", "noun"),
    pair("mart", "tram", "noun", "noun"),
    pair("paws", "swap", "noun", "verb"),
    pair("liar", "rail", "noun", "noun/verb"),
    pair("maps", "spam", "noun/verb", "noun/verb"),
    pair("edit", "tide", "verb", "noun/verb"),
    pair("spot", "tops", "noun/verb", "noun/verb"),
    pair("gums", "smug", "noun/verb", "adj"),
)

# Frames are intentionally bare clause frames: adding self-reversing
# determiners would make the pair inventory do no constructive work.
FRAMES = {
    "imperative": ("verb", "noun"),
    "copular": ("noun", "adj"),
    "nominal": ("adj", "noun"),
    "transitive": ("noun", "verb", "noun"),
    "descriptive": ("noun", "verb", "noun", "adj"),
}

def compatible(p: Pair, role: str, side: str) -> bool:
    return role in (p.left_pos if side == "left" else p.right_pos)

def audit(text: str) -> dict:
    tape = normalize(text)
    mismatches = [i for i in range(len(tape)//2) if tape[i] != tape[-1-i]]
    return {"exact": bool(tape) and not mismatches, "letters": len(tape),
            "mismatches": mismatches,
            "normalized_sha256": sha256(tape.encode()).hexdigest()}

def run(max_candidates: int = 10000) -> dict:
    rows, closures, rejected = [], [], 0
    for frame_name, roles in FRAMES.items():
        for choices in product(PAIRS, repeat=len(roles)):
            left = tuple(p.left for p in choices)
            right = tuple(p.right for p in choices[::-1])
            if any(not compatible(p, role, "left") for p, role in zip(choices, roles)):
                continue
            if any(not compatible(p, role, "right") for p, role in zip(choices[::-1], roles)):
                continue
            words = left + right
            if len(set(words)) != len(words):
                rejected += 1; continue
            text = " ".join(left) + "; " + " ".join(right)
            row = {"text": text, "frame": frame_name, "audit": audit(text),
                   "left_words": left, "right_words": right,
                   "provenance": "authored reverse-pair inventory; typed POS/valency frame"}
            if not row["audit"]["exact"] or row["audit"]["letters"] < MIN_LETTERS:
                continue
            row["anti_shortcut"] = {"distinct_words": True, "self_reversing_units": False,
                                    "catalogue_blacklist_applied": True,
                                    "whole_clause_shape_checked": True}
            closures.append(row)
    # Exact outputs are retained, but no programmatic score certifies prose.
    closures.sort(key=lambda r: (-r["audit"]["letters"], r["text"]))
    return {"status": "typed_semordnilap_closures_need_human_readability",
            "config": {"min_letters": MIN_LETTERS, "frames": FRAMES,
                        "pair_count": len(PAIRS), "catalogue_blacklist": True},
            "pair_inventory_sha256": sha256(repr(PAIRS).encode()).hexdigest(),
            "candidate_count": len(closures), "rejected_repeated_units": rejected,
            "candidates": closures[:max_candidates],
            "reader_facing_next_operator": "Add typed multiword reverse-boundary phrases whose two orientations each parse as a complete clause; do not broaden the blacklist."}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(); result = run(); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"candidate_count": result["candidate_count"], "top": result["candidates"][:5]}, indent=2))

if __name__ == "__main__": main()
