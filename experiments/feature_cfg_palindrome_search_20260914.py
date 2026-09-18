"""Feature-typed two-sided palindrome search.

This successor to the broad POS search keeps number, determiner, and valency
choices in separate lexical slots before character synchronization.  The two
readings are generated independently; no completed side is reversed or copied.
The output is a candidate ledger only: human readers, not this program, decide
whether a rendered closure is genuinely readable.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from experiments.joint_cfg_palindrome_search_20260914 import _indexes, solve_pair


def _brown_feature_extras(limit: int) -> dict[str, tuple[str, ...]]:
    """Return frequent, standalone lexical items for the feature slots.

    Brown contributes word forms and tags only; no Brown sentence span is ever
    copied.  The inventory remains deliberately bounded so every emitted word
    can be audited in the run manifest.
    """
    if limit <= 0:
        return {}
    try:
        from nltk.corpus import brown
        from wordfreq import zipf_frequency
    except Exception:
        return {}
    counts: Counter[str] = Counter()
    tags: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for sentence in brown.tagged_sents(tagset="universal"):
        for raw, tag in sentence:
            word = raw.casefold()
            if word.isascii() and word.isalpha() and len(word) >= 2:
                counts[word] += 1
                tags[word][tag] += 1
    def frequent(tag: str) -> list[str]:
        words = [
            word for word, count in counts.items()
            if tags[word][tag] >= 4 and word != word[::-1]
            and zipf_frequency(word, "en") >= 3.25
        ]
        words.sort(key=lambda word: (-counts[word], word))
        return words[:limit]
    nouns = frequent("NOUN")
    verbs = frequent("VERB")
    return {
        "NS": tuple(word for word in nouns if not word.endswith("s")),
        "NP": tuple(word for word in nouns if word.endswith("s")),
        "VTP": tuple(verbs),
        "VTS": tuple(verbs),
        "VIP": tuple(verbs),
        "VIS": tuple(verbs),
        "ADJ": tuple(frequent("ADJ")),
        "ADV": tuple(frequent("ADV")),
        "ADP": tuple(frequent("ADP")),
    }


def inventory() -> dict[str, tuple[str, ...]]:
    # Lexical choices are deliberately ordinary and hand-auditable.  Feature
    # slots prevent the closure search from treating an arbitrary Brown POS
    # sequence as a grammatical sentence.
    return {
        "DS": "a an this that my our his her one each".split(),
        "DA": "the some our their these those".split(),
        "DP": "the some these those".split(),
        "PS": "i he she it".split(),
        "PP": "we you they".split(),
        "NAME": "diana anna emma grace helen jane laura maria maya nina sara sophia liam delia noel leon emil damon evan".split(),
        "NS": (
            "aide artist baker child cook doctor farmer friend gardener guard helper "
            "man woman parent poet pupil teacher worker writer reader singer dancer "
            "son note book map letter memo plan key door song story task test cup cake "
            "room garden bread tool horse dog cat bird gate road river flower apple "
            "drawer reward devil mood star time diary school town night rain wind fire "
            "water light mail message table chair house home car card money truth music "
            "place life work hand part age set idea issue reason cause result event film "
            "team group name news sense answer code data line word"
        ).split(),
        "NP": (
            "artists bakers children cooks doctors farmers friends gardeners guards "
            "helpers men women parents poets pupils teachers workers writers readers "
            "singers dancers sons notes books maps letters memos plans keys doors songs "
            "stories tasks tests cups cakes rooms gardens breads tools horses dogs cats "
            "birds gates roads rivers flowers apples drawers rewards devils moods stars "
            "times diaries schools towns nights rains winds fires waters lights mails "
            "messages tables chairs houses homes cars cards people truths places lives "
            "works hands parts ages sets ideas issues reasons causes results events films "
            "teams groups names answers codes data lines words"
        ).split(),
        "VTP": "aid aided bake baked build built call called carry carried change changed clean cleaned close closed cook cooked draw drew drive drove "
        "eat ate find found fix fixed help helped hold held keep kept learn learned like liked love loved make made mark marked meet met notice noticed "
        "opened painted planned read repaired rescued saved saw sent shared showed "
        "study studied teach taught thank thanked tell told use used visit visited watch watched write wrote rip ripped rips inspire inspired inspire serves served "
        "recorded tested worked mailed rewarded delivered repaid".split(),
        "VTS": "aids bakes builds calls carries changes cleans closes cooks finds fixes helps holds keeps likes loves makes marks meets notices opens paints plans reads saves sees sends shows teaches tells uses visits watches writes mails rewards delivers".split(),
        "VIP": "arrived came died fell grew lived moved ran rose stayed".split(),
        "VIS": "arrives comes falls grows lives moves runs rises stays".split(),
        "AUX": "is was are were am".split(),
        "ADJ": "calm careful kind quiet small bright dark clear open warm cold good true old young patient useful brave new long short safe gentle ready alive alone".split(),
        "ADV": "now then here there often again away home well not early late".split(),
        "ADP": "at in on by for with from near after before under over into through around to".split(),
        "NUM": "one two three four five six seven eight nine ten".split(),
        "CONJ": "and but or yet".split(),
    }


# Feature slots encode basic determiner/number agreement and transitivity.
# They are compact complete-clause frames, not arbitrary POS strings.
SHAPES: tuple[tuple[str, ...], ...] = (
    ("DS", "NS", "VTP", "DS", "NS"),
    ("DA", "NP", "VTP", "DS", "NS"),
    ("DS", "NS", "VTP", "NUM", "NP"),
    ("DA", "NP", "VTP", "NAME"),
    ("PS", "VTP", "DS", "NS"),
    ("PP", "VTP", "DS", "NS"),
    ("DS", "ADJ", "NS", "VTP", "DS", "NS"),
    ("DA", "ADJ", "NP", "VTP", "DS", "NS"),
    ("DS", "NS", "VTP", "ADP", "DS", "NS"),
    ("PS", "VTP", "ADP", "DS", "NS"),
    ("PP", "VTP", "ADP", "DS", "NS"),
    ("DS", "NS", "AUX", "ADJ"),
    ("PS", "AUX", "ADJ"),
    ("NAME", "VTP", "DS", "NS"),
    ("NAME", "VTP", "NUM", "NP"),
    ("DS", "NS", "VTP", "NAME"),
    ("DS", "NS", "VTP", "ADV"),
    ("DS", "NS", "VTP", "DS", "NS", "ADP", "DS", "NS"),
    ("DS", "NS", "VTP", "DS", "NS", "CONJ", "PS", "VTP"),
    ("PS", "VTP", "DS", "NS", "CONJ", "PS", "VTP", "NS"),
    ("DS", "ADJ", "NS", "VTP", "ADP", "DS", "NS"),
    ("DS", "NS", "AUX", "ADJ", "ADP", "DS", "NS"),
    ("DS", "NS", "VTP", "ADV", "DS", "NS"),
    ("PS", "VTP", "DS", "ADJ", "NS"),
    ("NAME", "AUX", "ADJ"),
    ("DS", "NS", "VTP", "DS", "ADJ", "NS", "ADP", "DS", "NS"),
    ("PS", "AUX", "ADJ", "ADP", "DS", "NS"),
    ("DS", "NS", "VTP", "ADP", "DS", "ADJ", "NS"),
    ("DS", "ADJ", "NS", "VTP", "ADV"),
)


def run(*, shape_limit: int = 27, brown_limit: int = 0,
        state_budget: int = 80_000, result_limit: int = 80,
        min_letters: int = 38, max_letters: int = 150) -> dict[str, object]:
    # brown_limit is retained in the manifest to make the route explicit: the
    # default feature search is fully authored and corpus-free.
    authored = inventory()
    extras = _brown_feature_extras(brown_limit)
    lexicon = {
        slot: tuple(dict.fromkeys(tuple(authored.get(slot, ())) + tuple(extras.get(slot, ()))) )
        for slot in authored
    }
    shapes = SHAPES[:shape_limit]
    rows: list[dict[str, object]] = []
    stats: list[dict[str, object]] = []
    for left_shape in shapes:
        for right_shape in shapes:
            found, states = solve_pair(left_shape, right_shape, lexicon,
                                       *_indexes(lexicon),
                                       state_budget=state_budget,
                                       result_limit=result_limit)
            stats.append({"left_shape": left_shape, "right_shape": right_shape,
                          "states": states, "closures": len(found)})
            for left, right in found:
                rendered = " ".join(left + right)
                tape = normalize_letters(rendered)
                if not min_letters <= len(tape) <= max_letters:
                    continue
                checks = mechanical_admission_checks(rendered, min_letters=min_letters,
                                                     max_letters=max_letters)
                checks["independent_exact_audit"] = bool(tape) and tape == tape[::-1]
                checks["left_feature_reparse"] = len(left) == len(left_shape)
                checks["right_feature_reparse"] = len(right) == len(right_shape)
                rows.append({
                    "rendered": rendered,
                    "left": " ".join(left),
                    "right": " ".join(right),
                    "left_shape": list(left_shape),
                    "right_shape": list(right_shape),
                    "letters": len(tape),
                    "checks": checks,
                    "mechanically_eligible": all(checks.values()),
                    "provenance": {
                        "generator": "feature_cfg_palindrome_search_20260914",
                        "lexical_source": "authored feature-typed inventory",
                        "brown_limit": brown_limit,
                        "no_per_candidate_model": True,
                    },
                    "reader_status": "human-unreviewed; diagnostics do not certify readability",
                })
    unique: dict[str, dict[str, object]] = {}
    for row in rows:
        unique.setdefault(normalize_letters(str(row["rendered"])), row)
    rows = sorted(unique.values(), key=lambda row: (-int(row["letters"]), str(row["rendered"])))
    return {
        "status": "feature_typed_cfg_exact_search",
        "config": {"shape_limit": shape_limit, "state_budget": state_budget,
                   "result_limit": result_limit, "min_letters": min_letters,
                   "max_letters": max_letters, "brown_limit": brown_limit,
                   "independent_side_reparse": True,
                   "machine_readability_certification": False},
        "lexicon": lexicon,
        "lexicon_sha256": hashlib.sha256(json.dumps(lexicon, sort_keys=True).encode()).hexdigest(),
        "shape_count": len(shapes),
        "shape_stats": stats,
        "records": rows,
        "mechanically_eligible": [row for row in rows if row["mechanically_eligible"]],
        "reader_facing_next_test": "Only an original, mechanically eligible closure may enter a randomized blinded intact-prose versus shuffled-control study.",
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "method": "joint feature-typed left/right character synchronization",
                       "material": "authored ordinary words; no catalogue text"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--shape-limit", type=int, default=27)
    parser.add_argument("--brown-limit", type=int, default=0)
    parser.add_argument("--state-budget", type=int, default=80_000)
    parser.add_argument("--result-limit", type=int, default=80)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(shape_limit=args.shape_limit, brown_limit=args.brown_limit,
                 state_budget=args.state_budget, result_limit=args.result_limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(result["records"]),
                      "mechanically_eligible": len(result["mechanically_eligible"])}, sort_keys=True))


if __name__ == "__main__":
    main()
