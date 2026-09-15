"""Global POS-shape lattice search for exact English palindromes.

This branch reopens the grammar instead of adding another hand-authored
sentence frame.  Brown contributes only POS-shape counts and word types; no
corpus sentence is copied.  A center-out character search keeps the live POS
shape and unique-content constraints in the state, and a separate ASCII audit
rechecks every closure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from nltk.corpus import brown
from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (
    REPEATABLE_FUNCTION_WORDS,
    mechanical_admission_checks,
    normalize_letters,
)
from llm_palindrome.centerout import centerout_search
from llm_palindrome.search import WordTries
from llm_palindrome.semantic import RankOrderScorer

def independent_ascii_tape(text: str) -> str:
    return "".join(
        ch.casefold() for ch in text
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z")
    )


def inventory(
    *, min_words: int = 8, max_words: int = 15, max_shapes: int = 200,
    words_per_tag: int = 500, min_zipf: float = 3.0,
) -> tuple[tuple[tuple[str, ...], ...], dict[str, tuple[str, ...]]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    shape_counts: Counter[tuple[str, ...]] = Counter()
    for sentence in brown.tagged_sents(tagset="universal"):
        tokens = [
            (word.casefold(), tag)
            for word, tag in sentence
            if word.isascii() and word.isalpha()
        ]
        if min_words <= len(tokens) <= max_words:
            shape_counts[tuple(tag for _, tag in tokens)] += 1
        for word, tag in tokens:
            if word != word[::-1] and zipf_frequency(word, "en") >= min_zipf:
                counts[tag][word] += 1
    shapes = tuple(
        shape for shape, _ in shape_counts.most_common(max_shapes)
        if all(counts.get(tag) for tag in shape)
    )
    roles = set(tag for shape in shapes for tag in shape)
    pools = {
        tag: tuple(word for word, _ in counts[tag].most_common(words_per_tag))
        for tag in roles
    }
    return shapes, pools


def run(
    *, seeds: int = 48, beam: int = 1000, candidate_limit: int = 900,
    max_steps: int = 220, min_letters: int = 39, max_letters: int = 220,
    min_words: int = 8, max_words: int = 15, max_shapes: int = 200,
    words_per_tag: int = 500,
) -> dict:
    shapes, pools = inventory(
        min_words=min_words, max_words=max_words,
        max_shapes=max_shapes, words_per_tag=words_per_tag,
    )
    roles = set(tag for shape in shapes for tag in shape)
    words = sorted(
        {word for tag in roles for word in pools[tag]},
        key=lambda word: (-zipf_frequency(word, "en"), word),
    )
    tries = WordTries(words)
    scorer = RankOrderScorer(words, order_weight=0.0, length_weight=0.2,
                             reuse_weight=-8.0)
    tag_by_word = {
        word: frozenset(tag for tag in roles if word in pools.get(tag, ()))
        for word in words
    }

    @lru_cache(maxsize=500_000)
    def syntax_possible(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
        if len(left) + len(right) > max_words:
            return False
        if not left and not right:
            return True
        for shape in shapes:
            if len(left) + len(right) > len(shape):
                continue
            for split in range(len(left), len(shape) - len(right) + 1):
                if all(
                    slot in tag_by_word.get(word, ())
                    for word, slot in zip(left, shape[split - len(left):split])
                ) and all(
                    slot in tag_by_word.get(word, ())
                    for word, slot in zip(right, shape[split:split + len(right)])
                ):
                    return True
        return False

    def allow_state(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
        content = [
            word for word in left + right
            if word not in REPEATABLE_FUNCTION_WORDS
        ]
        return len(content) == len(set(content)) and syntax_possible(left, right)

    def syntax_complete(words_now: tuple[str, ...]) -> bool:
        return any(
            len(words_now) == len(shape)
            and all(slot in tag_by_word.get(word, ())
                    for word, slot in zip(words_now, shape))
            for shape in shapes
        )

    def allow_closed(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
        words_now = left + right
        if not syntax_complete(words_now):
            return False
        tape = normalize_letters(" ".join(words_now))
        return bool(tape) and tape == tape[::-1] and len(tape) <= max_letters

    records: list[dict] = []
    for seed in range(seeds):
        closed: list[list[str]] = []
        centerout_search(
            tries, scorer, min_letters=min_letters, beam_width=beam,
            candidate_limit=candidate_limit, max_steps=max_steps,
            per_parent=min(beam, 256), seed=seed, max_overhang=42,
            allow_state=allow_state, allow_closed=allow_closed,
            on_closed=lambda candidate: closed.append(candidate),
        )
        for candidate in closed:
            text = " ".join(candidate)
            tape = normalize_letters(text)
            independent = independent_ascii_tape(text)
            checks = mechanical_admission_checks(
                text, min_letters=min_letters, max_letters=max_letters,
            )
            records.append({
                "rendered": text,
                "words": candidate,
                "letters": len(tape),
                "independent_letters": len(independent),
                "independent_ascii_tape": independent,
                "independent_exact": bool(independent)
                and independent == independent[::-1]
                and independent == tape,
                "mechanical_checks": checks,
                "mechanically_eligible": all(checks.values())
                and independent == independent[::-1]
                and independent == tape,
                "seed": seed,
                "reader_status": "unreviewed; programmatic checks do not certify readability",
            })
    unique = {row["independent_ascii_tape"] if "independent_ascii_tape" in row else row["rendered"]: row for row in records}
    return {
        "status": "brown_pos_shape_lattice_complete",
        "config": {
            "seeds": seeds, "beam": beam, "candidate_limit": candidate_limit,
            "max_steps": max_steps, "min_letters": min_letters,
            "max_letters": max_letters, "min_words": min_words,
            "max_words": max_words, "max_shapes": max_shapes,
            "words_per_tag": words_per_tag,
            "syntax_live_state": True, "unique_content_live_state": True,
            "catalogue_text": False, "filler_or_mirror_shell": False,
        },
        "shape_count": len(shapes),
        "pool_counts": {tag: len(words_now) for tag, words_now in pools.items()},
        "shapes": [list(shape) for shape in shapes],
        "records": list(unique.values()),
        "mechanically_eligible": [
            row for row in unique.values() if row["mechanically_eligible"]
        ],
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "lexical_source": "Brown universal POS word types plus wordfreq ranking",
            "intact_corpus_sentences_used": False,
        },
        "next_operator_if_empty": (
            "Carry explicit transitivity and discourse-event roles through the same global POS lattice; "
            "do not widen only the word pool or relax exact/admission gates."
        ),
        "reader_gate": "No row is reader evidence; use intact prose and shuffled controls only after a novel exact survivor.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=48)
    parser.add_argument("--beam", type=int, default=1000)
    parser.add_argument("--candidate-limit", type=int, default=900)
    parser.add_argument("--max-steps", type=int, default=220)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite existing output")
    result = run(seeds=args.seeds, beam=args.beam,
                 candidate_limit=args.candidate_limit,
                 max_steps=args.max_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "records": len(result["records"]),
        "mechanically_eligible": len(result["mechanically_eligible"]),
    }, indent=2))


if __name__ == "__main__":
    main()
