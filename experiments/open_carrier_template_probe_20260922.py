"""Remote typed-template probe for longer carriers around the ``s`` seam."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

from open_carrier_brown_probe_20260922 import ORDINARY_TWO, corpus_chunks


FUNCTION = frozenset(
    "a an the this that these those i me we us you he him she her it they them "
    "who which whose and or but if as while when after before because though "
    "of to in on at by for from with without near during is are was were be been "
    "do does did can could will would may might should have has had not no some "
    "any each every either neither another all both few many much more most several".split()
)
BLOCKED = frozenset({"mr", "mrs", "miss", "john", "god", "christ", "yankee"})


def classes(tags: Counter) -> frozenset[str]:
    keys = set(tags)
    out = set()
    if any(tag.startswith("nn") for tag in keys): out.add("N")
    if any(tag == "vb" for tag in keys): out.add("V")
    if any(tag.startswith("jj") for tag in keys): out.add("A")
    if any(tag.startswith("rb") or tag in {"ql", "qlp"} for tag in keys): out.add("R")
    if any(tag.startswith("pp") or tag.startswith("wp") for tag in keys): out.add("P")
    if any(tag in {"at", "dt", "dti", "dts", "dtx", "abn", "abx", "cd"} for tag in keys): out.add("D")
    if any(tag in {"in", "to", "rp"} for tag in keys): out.add("I")
    return frozenset(out)


P_PATTERNS = (
    "V", "VN", "VDN", "VAN", "VDAN", "VINDN",
    "PV", "PVDN", "PVAN", "PVDAN", "PVINDN",
    "DN", "DAN", "DNV", "DNVDN", "DNVAN", "DNVINDN",
    "NV", "NVDN", "NVAN", "NVINDN",
    "RV", "RVDN", "RDNV", "RDNVDN",
    "DN V", "DN VDN", "VDN V", "VDN VDN",
)


def pattern_ok(words: tuple[str, ...], word_classes: dict[str, frozenset[str]]) -> list[str]:
    # Spaces in a pattern denote a licensed punctuation boundary.
    hits = []
    for raw in P_PATTERNS:
        pattern = raw.replace(" ", "")
        if len(pattern) != len(words):
            continue
        if all(symbol in word_classes[word] for symbol, word in zip(pattern, words)):
            hits.append(raw)
    # Preserve the incumbent's independently typed finding/directive carrier.
    if len(words) == 3 and words[0] == "no" and "N" in word_classes[words[1]] \
            and "V" in word_classes[words[2]]:
        hits.append("NO_N V")
    return hits


def make_segmenter(vocabulary: set[str], *, max_words: int = 7):
    terminal = ""
    trie = {}
    for word in vocabulary:
        node = trie
        for char in word:
            node = node.setdefault(char, {})
        node[terminal] = word

    def segment(target: str):
        found = []
        def visit(offset: int, words: tuple[str, ...]):
            if offset == len(target):
                if 2 <= len(words) <= max_words:
                    found.append(words)
                return
            if len(words) >= max_words:
                return
            node = trie
            for end in range(offset, len(target)):
                node = node.get(target[end])
                if node is None:
                    break
                word = node.get(terminal)
                if word is not None:
                    visit(end + 1, words + (word,))
        visit(0, ())
        return found
    return segment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("brown_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--min-count", type=int, default=5)
    args = parser.parse_args()

    chunks = list(corpus_chunks(args.brown_root))
    counts = Counter()
    tags = defaultdict(Counter)
    proper = set()
    for _source, _line, chunk in chunks:
        for word, tag in chunk:
            counts[word] += 1
            tags[word][tag] += 1
            if tag.startswith("np"):
                proper.add(word)
    word_classes = {word: classes(word_tags) for word, word_tags in tags.items()}
    vocabulary = {
        word for word, count in counts.items()
        if count >= args.min_count and word not in proper and word not in BLOCKED
        and word.isascii() and word.isalpha()
        and (len(word) > 1 or word in {"a", "i"})
        and (len(word) != 2 or word in ORDINARY_TWO)
        and word_classes[word]
    }
    segment = make_segmenter(vocabulary)
    ranked = lambda symbol, start="": tuple(
        word for word in sorted(vocabulary, key=lambda word: (-counts[word], word))
        if symbol in word_classes[word] and word.startswith(start)
    )[:args.max_per_class]
    banks = {symbol: ranked(symbol) for symbol in "NVADRPI"}
    banks["SV"] = ranked("V", "s")
    banks["SD"] = ranked("D", "s")
    banks["SP"] = ranked("P", "s")
    banks["SR"] = ranked("R", "s")

    # Q begins with s by construction.  These are complete finite clauses,
    # imperatives, noun phrases, or discourse continuations.
    q_templates = (
        ("SV", "D", "N"), ("SV", "A", "N"), ("SV", "D", "A", "N"),
        ("SV", "N", "I", "D", "N"), ("SV", "D", "N", "I", "D", "N"),
        ("SD", "N", "V"), ("SD", "A", "N", "V"),
        ("SP", "V", "D", "N"), ("SP", "V", "A", "N"),
        ("SR", "P", "V"), ("SR", "D", "N", "V"),
    )
    matches = []
    q_tested = 0
    seen = set()
    for template in q_templates:
        domains = [banks[symbol] for symbol in template]
        # Keep the widest patterns bounded while retaining common words.
        cap = {3: 120, 4: 35, 5: 15, 6: 10}[len(template)]
        domains = [domain[:cap] for domain in domains]
        for q_words in itertools.product(*domains):
            q_tested += 1
            if len(set(w for w in q_words if w not in FUNCTION)) != len(
                    [w for w in q_words if w not in FUNCTION]):
                continue
            q_tape = "".join(q_words)
            if not 13 <= len(q_tape) <= 37:
                continue
            p_tape = q_tape[::-1][:-1]
            for p_words in segment(p_tape):
                patterns = pattern_ok(p_words, word_classes)
                if not patterns:
                    continue
                content = [w for w in p_words + q_words if w not in FUNCTION]
                if len(content) != len(set(content)):
                    continue
                key = (p_words, q_words)
                if key in seen:
                    continue
                seen.add(key)
                score = sum(math.log1p(counts[word]) for word in p_words + q_words)
                matches.append({
                    "p_words": list(p_words), "q_words": list(q_words),
                    "p_pattern": patterns[0], "q_template": list(template),
                    "p_tape": p_tape, "q_tape": q_tape,
                    "equation_holds": q_tape[::-1] == p_tape + "s",
                    "full_letters_with_k2": 2 * len(p_tape) + 20,
                    "frequency_score": score,
                    "word_counts": {word: counts[word] for word in p_words + q_words},
                })
    matches.sort(key=lambda row: (
        -row["full_letters_with_k2"], -row["frequency_score"],
        row["p_words"], row["q_words"],
    ))
    payload = {
        "probe": "open-carrier-template-probe-20260922",
        "equation": "reverse(T(Q)) = T(P) + s",
        "stats": {
            "vocabulary": len(vocabulary), "q_templates": len(q_templates),
            "q_products_tested": q_tested, "matches": len(matches),
            "longer_than_42": sum(row["full_letters_with_k2"] > 42 for row in matches),
        },
        "bank_sizes": {key: len(value) for key, value in banks.items()},
        "matches": matches[:1000],
        "provenance": {
            "source": "typed word banks derived from Brown counts and tags",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "proper_tagged_words_excluded": True,
            "finished_phrase_reversal_used_for_generation": False,
            "equation_checked_during_joint template/segmentation search": True,
        },
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    for row in matches[:80]:
        print(row["full_letters_with_k2"], row["p_pattern"],
              "P=", " ".join(row["p_words"]), "| Q=", " ".join(row["q_words"]))


if __name__ == "__main__":
    main()
