"""Freeze fresh exact seeds and mirrored phrase-infill moves.

This is a material generator, not a selector over the rejected mirror-pair
bank.  It starts with new short exact closures from the two-ended search, cuts
one whole-word span on the left and its character-mirrored span on the right,
and records the two independent infill contexts.  A later proposer supplies
phrases for the two holes.  Only phrase pairs with exactly reversed normalized
letters can be applied, so exactness is decided mechanically outside the model.

The initial run is deliberately a small feasibility screen.  A successful
intersection is only a valid descendant, not a readable-output result; reader
evaluation remains required before promotion.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.generate import ZipfScorer, build_vocab
from llm_palindrome.overhang import DebtIndex, OverhangAware
from llm_palindrome.paragraphs import is_novel_palindrome
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.validator import is_palindrome, normalize


HOLE_WORD_COUNTS = (1, 2, 3, "whole_left")
INTENTS = (
    "a person notices and responds to a small event",
    "a speaker gives a compact instruction or warning",
    "two connected actions form one short scene",
    "a person makes an observation about an object or place",
)


@dataclass(frozen=True)
class Seed:
    seed_id: str
    search_seed: int
    words: tuple[str, ...]
    normalized: str
    intended_meaning: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def word_offsets(words: Sequence[str]) -> list[tuple[int, int]]:
    """Normalized-letter offsets for each source word."""
    out, at = [], 0
    for word in words:
        letters = normalize(word)
        if not letters:
            raise ValueError(f"non-letter source word: {word!r}")
        out.append((at, at + len(letters)))
        at += len(letters)
    return out


def marked_source(words: Sequence[str], start: int, end: int) -> str:
    """Show a character-level hole while retaining all original word spacing."""
    chunks = []
    at = 0
    for word in words:
        letters = normalize(word)
        pieces = []
        for index, char in enumerate(letters, at):
            if index == start:
                pieces.append("[HOLE]")
            if not (start <= index < end):
                pieces.append(char)
            if index + 1 == end:
                pieces.append("[HOLE]")
        at += len(letters)
        chunks.append("".join(pieces))
    if end == at:
        chunks[-1] += "[HOLE]"
    text = " ".join(chunks)
    # A marker opens and closes the same region; keep its two ends visible.
    return text.replace("[HOLE][HOLE]", "[HOLE]")


def move_for(seed: Seed, words_to_replace: int | str) -> dict:
    """Cut a left whole-word span and the exact character-mirrored right span."""
    offsets = word_offsets(seed.words)
    half_words = len(offsets) // 2
    half_letters = len(seed.normalized) // 2
    if words_to_replace == "whole_left":
        eligible = [index for index, (_, end) in enumerate(offsets) if end <= half_letters]
        if not eligible:
            raise ValueError("seed has no complete word on the left side")
        start_word, end_word = 0, eligible[-1]
    else:
        if not isinstance(words_to_replace, int) or not 1 <= words_to_replace <= half_words:
            raise ValueError("hole size cannot fit in the left half")
        eligible_starts = [start for start in range(len(offsets) - words_to_replace + 1)
                           if offsets[start + words_to_replace - 1][1] <= half_letters]
        if not eligible_starts:
            raise ValueError("hole size has no whole-word span on the left side")
        # Fixed but seed-dependent placement avoids choosing a language-looking span.
        start_word = eligible_starts[(seed.search_seed + words_to_replace - 1) % len(eligible_starts)]
        end_word = start_word + words_to_replace - 1
    left_start, left_end = offsets[start_word][0], offsets[end_word][1]
    total = len(seed.normalized)
    right_start, right_end = total - left_end, total - left_start
    if not (0 <= left_start < left_end <= right_start < right_end <= total):
        raise AssertionError("hole locations must be disjoint mirrored spans")
    return {
        "move_id": f"{seed.seed_id}:w{words_to_replace}",
        "seed_id": seed.seed_id,
        "words_to_replace": words_to_replace,
        "left_start": left_start,
        "left_end": left_end,
        "right_start": right_start,
        "right_end": right_end,
        "left_context": marked_source(seed.words, left_start, left_end),
        "right_context": marked_source(seed.words, right_start, right_end),
        "left_removed": seed.normalized[left_start:left_end],
        "right_removed": seed.normalized[right_start:right_end],
        "intended_meaning": seed.intended_meaning,
    }


def apply_replacements(seed: Seed, move: dict, left: str, right: str) -> str:
    """Apply a bilateral edit and require exact palindrome closure."""
    l = normalize(left)
    r = normalize(right)
    if not l or not r or l != r[::-1]:
        raise ValueError("replacement phrases are not exact reverse lexicalizations")
    s = seed.normalized
    result = (s[:move["left_start"]] + l + s[move["left_end"]:move["right_start"]]
              + r + s[move["right_end"]:])
    if result != result[::-1]:
        raise AssertionError("mirrored replacement broke exactness")
    return result


def intersect(left_candidates: Iterable[str], right_candidates: Iterable[str]) -> list[tuple[str, str]]:
    """Return every normalized reverse match, preserving independent phrases."""
    left = {}
    right = {}
    for phrase in left_candidates:
        key = normalize(phrase)
        if key:
            left.setdefault(key, phrase)
    for phrase in right_candidates:
        key = normalize(phrase)
        if key:
            right.setdefault(key, phrase)
    return [(left[key], right[key[::-1]]) for key in sorted(left) if key[::-1] in right]


def fresh_seeds(count: int, *, seed: int, min_letters: int,
                max_letters: int, vocabulary: int) -> list[Seed]:
    """Create new exact closures; do not source material from the v3 pair bank."""
    tries = WordTries(build_vocab(vocabulary))
    scorer = OverhangAware(ZipfScorer(), DebtIndex(tries), debt_weight=2.0)
    out, seen = [], set()
    # Fresh search seeds are deterministic; the cap is a material-generation
    # budget, not a claim that the generator is exhausted.
    for search_seed in range(seed, seed + max(80, count * 20)):
        words = beam_search(tries, scorer, min_letters=min_letters,
                            beam_width=24, max_steps=160, candidate_limit=100,
                            seed=search_seed)
        text = " ".join(words)
        letters = normalize(text)
        if (not words or not is_palindrome(text) or len(letters) > max_letters
                or letters in seen or not is_novel_palindrome(text)):
            continue
        seen.add(letters)
        seed_id = sha256_bytes(letters.encode())[:16]
        out.append(Seed(seed_id, search_seed, tuple(words), letters,
                        INTENTS[len(out) % len(INTENTS)]))
        if len(out) == count:
            return out
    raise RuntimeError(f"fresh search yielded {len(out)}/{count} novel closures")


def operator_controls() -> dict:
    """Known exact and one-letter-near-match cases validate the intersection."""
    seed = Seed("control", 0, ("step", "on", "no", "pets"), "steponnopets",
                "a warning")
    move = {"left_start": 0, "left_end": 6, "right_start": 6, "right_end": 12}
    positive = intersect(["step on", "other"], ["no pets", "wrong"])
    if positive != [("step on", "no pets")]:
        raise AssertionError("positive control did not intersect")
    if apply_replacements(seed, move, *positive[0]) != seed.normalized:
        raise AssertionError("positive control did not preserve exactness")
    near = intersect(["step on"], ["no pet"])
    if near:
        raise AssertionError("one-letter near-match must not intersect")
    return {"positive": {"left": "step on", "right": "no pets"},
            "near_match_rejected": {"left": "step on", "right": "no pet"}}


def build_materials(*, count: int, seed: int, min_letters: int,
                    max_letters: int, vocabulary: int) -> dict:
    seeds = fresh_seeds(count, seed=seed, min_letters=min_letters,
                        max_letters=max_letters, vocabulary=vocabulary)
    moves = []
    for item in seeds:
        for size in HOLE_WORD_COUNTS:
            try:
                moves.append(move_for(item, size))
            except ValueError:
                continue
    if not moves:
        raise AssertionError("fresh seeds yielded no applicable moves")
    return {
        "status": "frozen_fresh_seed_materials_pending_independent_proposals",
        "design": {
            "seed": seed, "seed_count": count, "min_letters": min_letters,
            "max_letters": max_letters, "vocabulary": vocabulary,
            "hole_word_counts": list(HOLE_WORD_COUNTS),
            "proposal_rule": "left and right lists are generated independently; only exact reverse intersections apply",
            "success_gate": "one novel, mechanically exact bilateral intersection justifies a larger material pilot; it is not readability evidence",
        },
        "input_sha256": {
            "llm_palindrome/generate.py": sha256_file(ROOT / "llm_palindrome" / "generate.py"),
            "llm_palindrome/search.py": sha256_file(ROOT / "llm_palindrome" / "search.py"),
            "data/known_palindromes.json": sha256_file(ROOT / "data" / "known_palindromes.json"),
        },
        "operator_controls": operator_controls(),
        "seeds": [asdict(item) for item in seeds],
        "moves": moves,
    }


def write_materials(out_dir: Path, materials: dict) -> None:
    internal = out_dir / "internal"
    internal.mkdir(parents=True)
    path = internal / "materials.json"
    path.write_text(json.dumps(materials, indent=2) + "\n")
    manifest = {str(path.relative_to(out_dir)): sha256_file(path)
                for path in sorted(out_dir.rglob("*")) if path.is_file()}
    (out_dir / "MANIFEST-SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--min-letters", type=int, default=30)
    parser.add_argument("--max-letters", type=int, default=60)
    parser.add_argument("--vocabulary", type=int, default=12000)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error(f"output path already exists: {args.out_dir}")
    materials = build_materials(count=args.seed_count, seed=args.seed,
                                min_letters=args.min_letters,
                                max_letters=args.max_letters,
                                vocabulary=args.vocabulary)
    write_materials(args.out_dir, materials)
    print(json.dumps({"out_dir": str(args.out_dir), "seeds": len(materials["seeds"]),
                      "moves": len(materials["moves"])}, indent=2))


if __name__ == "__main__":
    main()
