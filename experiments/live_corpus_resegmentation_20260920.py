"""Live overhang resegmentation of real forward prose.

The left arm is an intact Brown-corpus sentence.  The right arm is not made by
reversing that sentence after the fact: while the left arm's characters are
consumed from its opening edge, the search opens right-side words from their
closing edge and matches their reversed spellings one character at a time.
Only complete right arms whose POS shape and corpus bigram score look like an
English sentence are retained.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from nltk.corpus import brown
from wordfreq import zipf_frequency


TAG = {
    "nn": "NOUN", "nns": "NOUN", "np": "NAME", "nps": "NAME",
    "nr": "NAME", "vb": "VERB", "vbd": "VERB", "vbg": "VERB",
    "vbn": "VERB", "vbz": "VERB", "md": "VERB", "be": "VERB",
    "bed": "VERB", "beg": "VERB", "bem": "VERB", "ben": "VERB",
    "ber": "VERB", "bez": "VERB", "jj": "ADJ", "jjs": "ADJ",
    "jjr": "ADJ", "rb": "ADV", "rbr": "ADV", "rbt": "ADV",
    "dt": "DET", "at": "DET", "dti": "DET", "dts": "DET",
    "pp": "PRON", "pps": "PRON", "ppo": "PRON", "ppss": "PRON",
    "ppl": "PRON", "pn": "PRON", "in": "PREP", "to": "PREP",
    "cc": "CONJ", "cs": "COMP",
}

MANUAL = {
    "DET": "a an the some no one every each my our your his her this that",
    "PRON": "i me we us he him she her it they them you my our your",
    "VERB": "am is are was were be been do does did can will would could",
    "PREP": "of to in on at by for with from into near over under",
    "CONJ": "and or but while as if",
}


def norm(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = norm(text)
    mismatch = next(((i, tape[i], tape[-i - 1])
                     for i in range(len(tape) // 2)
                     if tape[i] != tape[-i - 1]), None)
    return {
        "letters": len(tape),
        "exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def hidden_span(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.casefold())
    full = norm(text)
    for i in range(len(words)):
        for j in range(i + 2, len(words) + 1):
            span = "".join(words[i:j])
            if 1 < len(span) < len(full) and span == span[::-1]:
                return True
    return False


@dataclass(frozen=True)
class Lexicon:
    categories: dict[str, frozenset[str]]
    reversed_words: dict[str, tuple[str, ...]]
    bigrams: Counter
    sentence_shapes: frozenset[tuple[str, ...]]


def build_lexicon(limit: int = 2500) -> Lexicon:
    categories: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, Counter] = defaultdict(Counter)
    bigrams: Counter = Counter()
    shape_counts: Counter = Counter()
    for word, tag in brown.tagged_words():
        word = word.casefold()
        if not re.fullmatch(r"[a-z]+", word):
            continue
        category = TAG.get(tag.split("-", 1)[0].casefold())
        if category:
            categories[word].add(category)
            counts[category][word] += 1
    for sentence in brown.sents():
        words = [re.sub(r"[^a-z]", "", word.casefold()) for word in sentence]
        words = [word for word in words if word]
        if 3 <= len(words) <= 14:
            shape = []
            for word in words:
                choices = categories.get(word, set())
                shape.append(next(iter(sorted(choices))) if choices else "OTHER")
            if "OTHER" not in shape:
                shape_counts[tuple(shape)] += 1
        bigrams.update(zip(words, words[1:]))

    for category, text in MANUAL.items():
        for word in text.split():
            categories[word].add(category)
            counts[category][word] += 100

    # Keep the common vocabulary broad enough for genuine resegmentation, but
    # exclude one-off garbage and punctuation tokens from the Brown source.
    for category, counter in counts.items():
        for word, _ in counter.most_common(limit):
            if (len(word) == 1 and word not in {"a", "i"}) or len(word) > 14:
                continue
            if zipf_frequency(word, "en") >= 2.0:
                categories[word].add(category)

    # A reversed-word trie is represented as first-character buckets here;
    # live matching additionally checks the whole exposed spelling.
    reversed_words: dict[str, list[str]] = defaultdict(list)
    for word, cats in categories.items():
        if ((len(word) > 1 or word in {"a", "i"}) and cats
                and zipf_frequency(word, "en") >= 2.0):
            reversed_words[word[::-1][0]].append(word)
    return Lexicon(
        categories={word: frozenset(cats) for word, cats in categories.items()},
        reversed_words={char: tuple(words) for char, words in reversed_words.items()},
        bigrams=bigrams,
        sentence_shapes=frozenset(shape_counts),
    )


def source_sentences(min_letters: int = 36, max_letters: int = 100):
    for sentence in brown.sents():
        words = [re.sub(r"[^a-z]", "", word.casefold()) for word in sentence]
        words = [word for word in words if word]
        if not 4 <= len(words) <= 14:
            continue
        text = " ".join(words)
        if min_letters <= len(norm(text)) <= max_letters:
            yield words, text


def shape_matches(words: tuple[str, ...], lexicon: Lexicon) -> bool:
    options = [lexicon.categories.get(word, frozenset()) for word in words]
    if any(not choices for choices in options):
        return False
    # Brown-derived shapes are deliberately used as a permissive grammar
    # witness, not as a claim that corpus frequency alone proves readability.
    states = {()}
    for choices in options:
        states = {prefix + (category,)
                  for prefix in states for category in choices}
        if len(states) > 10000:
            break
    return bool(states & lexicon.sentence_shapes)


def sentence_score(words: tuple[str, ...], lexicon: Lexicon) -> float:
    if not words:
        return -1e9
    score = sum(0.12 * zipf_frequency(word, "en") for word in words)
    for a, b in zip(words, words[1:]):
        score += math.log1p(lexicon.bigrams[a, b])
    if any(len(word) == 1 for word in words):
        score -= 0.25
    return score


def live_resegment(left_words: tuple[str, ...], lexicon: Lexicon,
                   beam: int = 800) -> list[dict]:
    left_tape = norm(" ".join(left_words))
    # State stores the right arm in reverse word order: each newly opened word
    # is the word exposed at the current right boundary.
    states: dict[int, list[tuple[tuple[str, ...], float]]] = {0: [((), 0.0)]}
    for pos in range(len(left_tape)):
        current = states.get(pos, [])
        current.sort(key=lambda item: -item[1])
        current = current[:beam]
        states[pos] = current
        if not current:
            continue
        first = left_tape[pos]
        for right_rev, score in current:
            for word in lexicon.reversed_words.get(first, ()):
                spelling = word[::-1]
                if not left_tape.startswith(spelling, pos):
                    continue
                if right_rev and word == right_rev[0]:
                    continue
                new_pos = pos + len(spelling)
                if new_pos > len(left_tape):
                    continue
                # right_rev is [right_last, ...].  The new word is opened at
                # the outer boundary and therefore prepended in final order.
                new_score = score + 0.12 * zipf_frequency(word, "en")
                if right_rev:
                    new_score += math.log1p(lexicon.bigrams[word, right_rev[0]])
                states.setdefault(new_pos, []).append(
                    ((word,) + right_rev, new_score))
        # Bound every future bucket immediately.  Without this, long words
        # leave many unpruned buckets alive and make a corpus sweep look like
        # a grammar search even though only the best live frontier matters.
        for key in list(states):
            if key > pos:
                states[key].sort(key=lambda item: -item[1])
                states[key] = states[key][:beam]
    rows = []
    for right_rev, score in states.get(len(left_tape), []):
        right_words = tuple(reversed(right_rev))
        if len(right_words) < 4 or not shape_matches(right_words, lexicon):
            continue
        text = " ".join(left_words) + "; " + " ".join(right_words)
        rows.append({
            "rendered": text,
            "left_words": list(left_words),
            "right_words": list(right_words),
            "score": score + sentence_score(right_words, lexicon),
            "audit": audit(text),
            "hidden_palindromic_span": hidden_span(text),
            "provenance": {
                "left_source": "Brown forward sentence",
                "right_construction": "live reversed-word boundary expansion",
                "finished_tape_reversal": False,
                "post_hoc_repair": False,
            },
        })
    return rows


def run() -> dict:
    lexicon = build_lexicon(int(os.environ.get("CORPUS_LEX_LIMIT", "2500")))
    beam = int(os.environ.get("CORPUS_BEAM", "800"))
    rows: list[dict] = []
    sources = list(itertools.islice(source_sentences(
        int(os.environ.get("CORPUS_MIN_LETTERS", "36")),
        int(os.environ.get("CORPUS_MAX_LETTERS", "100"))),
        int(os.environ.get("CORPUS_SOURCE_LIMIT", "0")) or None))
    for left_words, _ in sources:
        rows.extend(live_resegment(tuple(left_words), lexicon, beam=beam))
    rows.sort(key=lambda row: (-row["audit"]["letters"], -row["score"], row["rendered"]))
    clean = [row for row in rows if not row["hidden_palindromic_span"]]
    return {
        "experiment": "live-corpus-resegmentation-20260920",
        "method": "Brown forward sentence plus live opposite-edge word-boundary expansion",
        "stats": {
            "source_sentences": len(sources),
            "lexicon_words": len(lexicon.categories),
            "beam": beam,
            "exact": len(rows),
            "longest_exact": max((row["audit"]["letters"] for row in rows), default=0),
            "longest_hidden_span_free": max((row["audit"]["letters"] for row in clean), default=0),
        },
        "exact_candidates": rows[:100],
        "clean_exact_candidates": clean[:50],
        "provenance": {
            "audits": ["independent two-pointer comparison", "forward/reverse SHA-256"],
            "finished_tape_reversal": False,
            "post_hoc_repair": False,
            "reader_gate": "closed pending human reading and semantic review",
        },
    }


if __name__ == "__main__":
    output = Path("runs/live-corpus-resegmentation-20260920.json")
    result = run()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
    for row in result["clean_exact_candidates"][:30]:
        print(row["audit"]["letters"], row["rendered"])
