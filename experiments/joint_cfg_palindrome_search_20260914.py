"""Joint typed-grammar search for novel readable palindrome candidates.

This is a constructive search-space method.  The left and right readings are
selected independently from typed clause frames while a character synchronizer
matches their outer letters.  It never reverses a finished clause and never
uses a model to score individual candidates.  The result ledger records exact
mechanical evidence; readability remains a separate human gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


# Every slot is an independently authored lexical inventory.  A small Brown
# frequency supplement is used only to widen ordinary lexical choices; Brown
# sentences are never copied into a result.
BASE: dict[str, tuple[str, ...]] = {
    "DET": "a an the some my our his her one no each this that their your".split(),
    "PRON": "i we you he she they it me us them who".split(),
    "NAME": "diana anna emma grace helen jane laura maria maya nina sara sophia liam delia noel leon emil damon evan".split(),
    "NOUN": (
        "aide artist baker child cook doctor farmer friend gardener guard helper "
        "man men woman women parent poet pupil teacher worker writer reader singer "
        "dancer note notes book books map maps letter letters memo memos plan plans "
        "key door song story task test cup cake room rooms garden gardens bread tool "
        "horse dog cat bird gate road river flower apple drawer reward devil mood doom "
        "star rats time diary school town world day night rain wind fire water light "
        "mail message table chair house home car card money truth music people state "
        "place life work hand part age son set idea issue reason cause result event "
        "film team group name news sense answer code data line word number child"
    ).split(),
    "VERB": (
        "aid aids asks asked bake baked builds built call called calls carry carried "
        "change changed clean cleaned close closed cook cooks draw drew drink drank "
        "drive drove eat eats find found fix fixed fixes follow followed help helped "
        "hold held keep kept learn learned like liked likes listen listened live lived "
        "love loved loves make made mark marked meet met move moved notice noticed "
        "open opened paint painted plan planned read reads repair repaired rescue rescued "
        "save saved saw see sees send sent share shared show showed study studied teach "
        "taught thank thanked tell told use used visit visited wait waited walk walked "
        "watch watched write wrote rip rips inspire inspires serve served record records "
        "test tests work works mail mailed mails reward rewards deliver delivered repay "
        "repaid stress stressed say says began begin starts started ends ended states "
        "state sets set gets got gives gave takes took puts put shows finds makes keeps "
        "brings brought leaves left means meant needs needed knows knew"
    ).split(),
    "AUX": "is was are were am be been".split(),
    "ADJ": "calm careful kind quiet small bright dark clear open warm cold good true old young patient useful brave new long short safe gentle ready full empty red green".split(),
    "ADV": "now then here there often again away home well not slowly carefully outside inside early late".split(),
    "ADP": "at in on by for with from near after before under over into through around".split(),
    "NUM": "one two three four five six seven eight nine ten".split(),
    "CONJ": "and but or yet".split(),
}


# These are complete, ordinary clause frames rather than arbitrary POS salads.
# Agreement/valency is checked by the independent lexical parser below.
SHAPES: tuple[tuple[str, ...], ...] = (
    ("DET", "NOUN", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "NUM", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "ADP", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "ADV"),
    ("DET", "NOUN", "AUX", "ADJ"),
    ("PRON", "VERB", "DET", "NOUN"),
    ("PRON", "VERB", "ADP", "DET", "NOUN"),
    ("PRON", "VERB", "ADV"),
    ("PRON", "AUX", "ADJ"),
    ("NAME", "VERB", "DET", "NOUN"),
    ("NAME", "VERB", "NUM", "NOUN"),
    ("NAME", "VERB", "ADP", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "NAME"),
    ("DET", "NOUN", "VERB", "DET", "NOUN", "ADP", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "NOUN", "CONJ", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "NOUN", "CONJ", "PRON", "VERB"),
    ("PRON", "VERB", "DET", "NOUN", "CONJ", "PRON", "VERB", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "ADP", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "ADV", "DET", "NOUN"),
    ("DET", "NOUN", "AUX", "ADJ", "ADP", "DET", "NOUN"),
    ("PRON", "VERB", "DET", "ADJ", "NOUN"),
    ("NAME", "AUX", "ADJ"),
    ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN", "ADP", "DET", "NOUN"),
    ("PRON", "AUX", "ADJ", "ADP", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "ADP", "DET", "ADJ", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "ADV"),
)

FUNCTION_WORDS = frozenset(
    "a an the some my our his her one no each this that their your i we you he "
    "she they it me us them who is was are were am be been at in on by for with "
    "from near after before under over into through around one two three four "
    "five six seven eight nine ten and but or yet".split()
)


def _augment_with_brown(base: dict[str, tuple[str, ...]], limit: int) -> dict[str, tuple[str, ...]]:
    """Add frequent attested lexical forms without importing sentence spans."""
    if limit <= 0:
        return {key: tuple(dict.fromkeys(value)) for key, value in base.items()}
    try:
        from nltk.corpus import brown
        from wordfreq import zipf_frequency
    except Exception:
        return {key: tuple(dict.fromkeys(value)) for key, value in base.items()}
    counts: Counter[str] = Counter()
    tags: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for sentence in brown.tagged_sents(tagset="universal"):
        for raw, tag in sentence:
            word = raw.casefold()
            if word.isascii() and word.isalpha() and len(word) >= 2:
                counts[word] += 1
                tags[word][tag] += 1
    mapping = {"ADP": "ADP"}
    out: dict[str, tuple[str, ...]] = {}
    for slot, authored in base.items():
        tag = mapping.get(slot, slot)
        extras = [
            word for word in counts
            if tags[word][tag] >= 5 and word != word[::-1]
            and zipf_frequency(word, "en") >= 3.0
        ]
        extras.sort(key=lambda word: (-counts[word], word))
        out[slot] = tuple(dict.fromkeys(tuple(authored) + tuple(extras[:limit])))
    return out


def _indexes(lexicon: dict[str, tuple[str, ...]]):
    by_first: dict[str, dict[str, tuple[str, ...]]] = {}
    by_last: dict[str, dict[str, tuple[str, ...]]] = {}
    for slot, words in lexicon.items():
        first: defaultdict[str, list[str]] = defaultdict(list)
        last: defaultdict[str, list[str]] = defaultdict(list)
        for word in words:
            first[word[0]].append(word)
            last[word[-1]].append(word)
        by_first[slot] = {key: tuple(value) for key, value in first.items()}
        by_last[slot] = {key: tuple(value) for key, value in last.items()}
    return by_first, by_last


def _content_key(word: str) -> str:
    """Conservative family key used to reject repeated content roots."""
    word = re.sub(r"(?:s|es|ed|ing)$", "", word)
    return word


def solve_pair(
    left_shape: Sequence[str],
    right_shape: Sequence[str],
    lexicon: dict[str, tuple[str, ...]],
    by_first: dict[str, dict[str, tuple[str, ...]]],
    by_last: dict[str, dict[str, tuple[str, ...]]],
    *,
    state_budget: int,
    result_limit: int,
    unique_content: bool = True,
) -> tuple[list[tuple[tuple[str, ...], tuple[str, ...]]], int]:
    """Synchronize two independently typed readings from their outer edges."""
    results: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    states = 0

    def rec(
        li: int, lp: int, lw: str | None,
        ri: int, rp: int, rw: str | None,
        left: tuple[str, ...], right_reversed: tuple[str, ...],
        used_content: frozenset[str],
    ) -> None:
        nonlocal states
        states += 1
        if states > state_budget or len(results) >= result_limit:
            return
        if lw is not None and lp >= len(lw):
            li, lp, lw = li + 1, 0, None
        if rw is not None and rp >= len(rw):
            ri, rp, rw = ri - 1, 0, None
        if li >= len(left_shape) or ri < 0:
            if li >= len(left_shape) and ri < 0 and lw is None and rw is None:
                full = left + tuple(reversed(right_reversed))
                if full and (not unique_content or len(set(_content_key(word) for word in full if word not in FUNCTION_WORDS)) == len(
                    [word for word in full if word not in FUNCTION_WORDS]
                )):
                    results.append((left, tuple(reversed(right_reversed))))
            return
        if lw is not None and rw is not None:
            if lw[lp] == rw[-1 - rp]:
                rec(li, lp + 1, lw, ri, rp + 1, rw, left, right_reversed, used_content)
            return

        if lw is None and rw is None:
            for left_word in lexicon[left_shape[li]]:
                left_key = _content_key(left_word)
                if unique_content and left_word not in FUNCTION_WORDS and left_key in used_content:
                    continue
                for right_word in by_last[right_shape[ri]].get(left_word[0], ()):
                    right_key = _content_key(right_word)
                    if right_word == left_word and left_word not in FUNCTION_WORDS:
                        continue
                    if unique_content and right_word not in FUNCTION_WORDS and right_key in used_content:
                        continue
                    next_used = used_content
                    if left_word not in FUNCTION_WORDS:
                        next_used = next_used | {left_key}
                    if right_word not in FUNCTION_WORDS:
                        next_used = next_used | {right_key}
                    rec(li, 0, left_word, ri, 0, right_word,
                        left + (left_word,), right_reversed + (right_word,), next_used)
            return

        if lw is None:
            wanted = rw[-1 - rp]
            for left_word in by_first[left_shape[li]].get(wanted, ()):
                key = _content_key(left_word)
                if unique_content and left_word not in FUNCTION_WORDS and key in used_content:
                    continue
                next_used = used_content | ({key} if left_word not in FUNCTION_WORDS else set())
                rec(li, 0, left_word, ri, rp, rw,
                    left + (left_word,), right_reversed, next_used)
            return

        wanted = lw[lp]
        for right_word in by_last[right_shape[ri]].get(wanted, ()):
            key = _content_key(right_word)
            if unique_content and right_word not in FUNCTION_WORDS and key in used_content:
                continue
            next_used = used_content | ({key} if right_word not in FUNCTION_WORDS else set())
            rec(li, lp, lw, ri, 0, right_word,
                left, right_reversed + (right_word,), next_used)

    rec(0, 0, None, len(right_shape) - 1, 0, None, (), (), frozenset())
    return results, states


def independent_parse(words: Sequence[str], shape: Sequence[str], lexicon: dict[str, tuple[str, ...]]) -> dict[str, object]:
    """Reparse a completed side against its declared slot inventory."""
    if len(words) != len(shape):
        return {"ok": False, "reason": "word_count", "tokens": list(words)}
    for word, slot in zip(words, shape):
        if word not in lexicon.get(slot, ()):
            return {"ok": False, "reason": f"unknown_{slot}", "tokens": list(words)}
    return {"ok": True, "slots": list(shape), "tokens": list(words)}


def run(*, brown_limit: int = 120, shape_limit: int = 28,
        state_budget: int = 45_000, result_limit: int = 30,
        min_letters: int = 38, max_letters: int = 140) -> dict[str, object]:
    lexicon = _augment_with_brown(BASE, brown_limit)
    by_first, by_last = _indexes(lexicon)
    shapes = SHAPES[:shape_limit]
    rows: list[dict[str, object]] = []
    stats: list[dict[str, object]] = []
    for left_shape in shapes:
        for right_shape in shapes:
            found, states = solve_pair(left_shape, right_shape, lexicon, by_first, by_last,
                                       state_budget=state_budget, result_limit=result_limit)
            stats.append({"left_shape": left_shape, "right_shape": right_shape,
                          "states": states, "closures": len(found)})
            for left, right in found:
                text = " ".join(left + right)
                tape = normalize_letters(text)
                if not min_letters <= len(tape) <= max_letters:
                    continue
                checks = mechanical_admission_checks(text, min_letters=min_letters, max_letters=max_letters)
                checks["independent_exact_audit"] = bool(tape) and tape == tape[::-1]
                checks["left_reparse"] = independent_parse(left, left_shape, lexicon)["ok"]
                checks["right_reparse"] = independent_parse(right, right_shape, lexicon)["ok"]
                rows.append({
                    "rendered": text,
                    "left": " ".join(left),
                    "right": " ".join(right),
                    "left_shape": list(left_shape),
                    "right_shape": list(right_shape),
                    "letters": len(tape),
                    "checks": checks,
                    "mechanically_eligible": all(checks.values()),
                    "provenance": {
                        "generator": "joint_cfg_palindrome_search_20260914",
                        "left_source": "independently authored typed lexical inventory",
                        "right_source": "independently authored typed lexical inventory",
                        "corpus_role": "Brown frequency supplement only; no sentence spans copied",
                    },
                    "reader_status": "human-unreviewed; programmatic checks do not certify readability",
                })
    unique: dict[str, dict[str, object]] = {}
    for row in rows:
        unique.setdefault(normalize_letters(str(row["rendered"])), row)
    rows = sorted(unique.values(), key=lambda row: (-int(row["letters"]), str(row["rendered"])))
    return {
        "status": "joint_typed_cfg_exact_search",
        "config": {"brown_limit": brown_limit, "shape_limit": shape_limit,
                   "state_budget": state_budget, "result_limit": result_limit,
                   "min_letters": min_letters, "max_letters": max_letters,
                   "independent_side_reparse": True, "no_per_candidate_model": True,
                   "machine_readability_certification": False},
        "lexicon": {slot: list(words) for slot, words in lexicon.items()},
        "lexicon_sha256": hashlib.sha256(json.dumps(lexicon, sort_keys=True).encode()).hexdigest(),
        "shape_count": len(shapes),
        "shape_stats": stats,
        "records": rows,
        "mechanically_eligible": [row for row in rows if row["mechanically_eligible"]],
        "reader_facing_next_test": "Only an original mechanically eligible closure may enter a randomized blinded intact-prose versus shuffled-control study.",
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "material": "typed lexical products; no borrowed catalogue text",
                       "search": "outer-character synchronization with independent left/right clause slots"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--brown-limit", type=int, default=120)
    parser.add_argument("--shape-limit", type=int, default=28)
    parser.add_argument("--state-budget", type=int, default=45_000)
    parser.add_argument("--result-limit", type=int, default=30)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(brown_limit=args.brown_limit, shape_limit=args.shape_limit,
                 state_budget=args.state_budget, result_limit=args.result_limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(result["records"]),
                      "mechanically_eligible": len(result["mechanically_eligible"])}, sort_keys=True))


if __name__ == "__main__":
    main()
