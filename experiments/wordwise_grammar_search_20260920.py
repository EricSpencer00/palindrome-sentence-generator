"""Search wordwise semordnilap paths under two independent POS templates.

This is a small diagnostic: each left word is paired with its exact reverse,
but both resulting sides are required to match separately scored grammar
templates.  It is deliberately stricter than a raw reversible-word list and
is not a promotion route unless an intact reader-worthy sentence survives.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from nltk.corpus import brown
from wordfreq import zipf_frequency


TAG = {
    "dt": "DET", "at": "DET", "dti": "DET", "dts": "DET",
    "pp": "PRON", "pps": "PRON", "ppo": "PRON", "ppss": "PRON",
    "ppl": "PRON", "pn": "PRON", "in": "PREP", "to": "PREP",
    "cc": "CONJ", "cs": "COMP", "jj": "ADJ", "jjs": "ADJ",
    "jjr": "ADJ", "rb": "ADV", "rbr": "ADV", "rbt": "ADV",
    "nn": "NOUN", "nns": "NOUN", "np": "NAME", "nps": "NAME",
    "nr": "NAME", "vb": "VERB", "vbd": "VERB", "vbg": "VERB",
    "vbn": "VERB", "vbz": "VERB", "md": "VERB", "be": "VERB",
    "bed": "VERB", "beg": "VERB", "bem": "VERB", "ben": "VERB",
    "ber": "VERB", "bez": "VERB",
}


BASE_TEMPLATES = [
    ["PRON", "VERB"],
    ["PRON", "VERB", "DET", "NOUN"],
    ["PRON", "VERB", "NOUN"],
    ["DET", "NOUN", "VERB"],
    ["DET", "NOUN", "VERB", "DET", "NOUN"],
    ["DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"],
    ["DET", "NOUN", "VERB", "ADV"],
    ["NAME", "VERB", "DET", "NOUN"],
    ["NAME", "VERB", "ADV"],
    ["DET", "NOUN", "VERB", "PREP", "DET", "NOUN"],
    ["DET", "ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"],
    ["PRON", "VERB", "ADV", "VERB", "DET", "NOUN"],
    ["DET", "NOUN", "VERB", "COMP", "DET", "NOUN"],
]


def build() -> tuple[dict[str, set[str]], dict[str, str], Counter, Counter]:
    categories: dict[str, set[str]] = {}
    bigrams: Counter = Counter()
    starts: Counter = Counter()
    for word, tag in brown.tagged_words():
        word = word.casefold()
        if not re.fullmatch(r"[a-z]+", word):
            continue
        category = TAG.get(tag.split("-", 1)[0].casefold())
        if category:
            categories.setdefault(word, set()).add(category)
    for sentence in brown.sents():
        words = [re.sub(r"[^a-z]", "", word.casefold()) for word in sentence]
        words = [word for word in words if word]
        if words:
            starts[words[0]] += 1
            bigrams.update(zip(words, words[1:]))
    reverse = {}
    for word in categories:
        other = word[::-1]
        if (other in categories and len(word) >= 2
                and max(zipf_frequency(word, "en"),
                        zipf_frequency(other, "en")) >= 3.0):
            reverse[word] = other
    return categories, reverse, bigrams, starts


def search(limit: int = 500, keep: int = 100) -> list[dict]:
    categories, reverse, bigrams, starts = build()
    templates = list(BASE_TEMPLATES)
    for first in BASE_TEMPLATES:
        for second in BASE_TEMPLATES:
            if len(first) + 1 + len(second) <= 12:
                templates.append(first + ["CONJ"] + second)
    options: dict[tuple[str, str], list[str]] = defaultdict(list)
    for word, other in reverse.items():
        for left_cat in categories[word]:
            for right_cat in categories[other]:
                options[left_cat, right_cat].append(word)

    def link(a: str, b: str) -> float:
        return math.log1p(bigrams[a, b]) - 0.15

    def phrase_score(words: list[str]) -> float:
        return (sum(link(a, b) for a, b in zip(words, words[1:]))
                + 0.1 * sum(zipf_frequency(word, "en") for word in words))

    rows: list[dict] = []
    for left_template in templates:
        for right_template in templates:
            if len(left_template) != len(right_template):
                continue
            states: list[tuple[tuple[str, ...], float]] = [((), 0.0)]
            for index, left_category in enumerate(left_template):
                right_category = right_template[-1 - index]
                words = options[left_category, right_category]
                if not words:
                    states = []
                    break
                next_states = []
                for chosen, score in states:
                    for word in words:
                        other = reverse[word]
                        if word in chosen or other in chosen:
                            continue
                        new = chosen + (word,)
                        left_score = (link(chosen[-1], word)
                                      if chosen else math.log1p(starts[word]))
                        right = [reverse[item] for item in new[::-1]]
                        next_states.append((new, score + left_score
                                            + 0.1 * zipf_frequency(word, "en")
                                            + 0.5 * phrase_score(right)))
                next_states.sort(key=lambda item: -item[1])
                states = next_states[:limit]
                if not states:
                    break
            for chosen, score in states[:keep]:
                right = [reverse[item] for item in chosen[::-1]]
                letters = sum(map(len, chosen + tuple(right)))
                if letters >= 39:
                    rows.append({"letters": letters, "score": score,
                                 "left": list(chosen), "right": right,
                                 "left_template": left_template,
                                 "right_template": right_template})
    rows.sort(key=lambda row: (-row["score"], -row["letters"], row["left"]))
    seen = set()
    unique = []
    for row in rows:
        text = " ".join(row["left"]) + "; " + " ".join(row["right"])
        if text in seen:
            continue
        seen.add(text)
        row["rendered"] = text + "."
        unique.append(row)
    return unique[:keep]


if __name__ == "__main__":
    rows = search()
    print({"rows": len(rows), "max_letters": max((row["letters"] for row in rows), default=0)})
    for row in rows[:100]:
        print(row["letters"], row["rendered"])
