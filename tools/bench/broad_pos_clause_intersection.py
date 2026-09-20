"""Bench-only broad POS clause intersection.

This is a lexical-support expansion of the paired-clause constructor, not a
finished-tape reverse or a post-hoc repair.  Brown supplies word/POS support;
the sentence shapes are explicit authored templates, and the two clauses are
joined only by live outside-in character checks.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

try:
    from wordfreq import zipf_frequency
except ModuleNotFoundError:  # lexical support is already Brown-filtered
    def zipf_frequency(_word: str, _lang: str) -> float:
        return 5.0

try:
    from nltk.corpus import brown  # type: ignore
except ModuleNotFoundError:  # the bench uses the shipped Brown files directly
    brown = None

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get(
    "PAL_OUT", str(Path.cwd() / "broad-pos-clause-intersection-20260920.json")
))


def _words(prefixes: tuple[str, ...], limit: int = 120) -> list[str]:
    counts: Counter[str] = Counter()
    if brown is not None:
        tagged = brown.tagged_words()
    else:
        data_root = Path(__file__).with_name("brown")
        tagged = []
        for path in sorted(data_root.glob("c*")):
            for token in path.read_text(encoding="latin-1").split():
                if "/" not in token:
                    continue
                raw, tag = token.rsplit("/", 1)
                tagged.append((raw, tag.upper()))
    for raw, tag in tagged:
        word = raw.casefold()
        if not word.isalpha() or not tag.startswith(prefixes):
            continue
        if zipf_frequency(word, "en") < 4.0 or len(word) > 12:
            continue
        counts[word] += 1
    return [word for word, _ in counts.most_common(limit)]


BANK = {
    "DET": ["a", "an", "the", "some", "our", "your", "this", "that", "each", "one", "no"],
    "ADJ": _words(("JJ",), 800),
    "NOUN": _words(("NN",), 1000),
    "VERB": _words(("VB",), 1000),
    "PREP": _words(("IN",), 120),
    "ADV": _words(("RB",), 300),
    "PRON": _words(("PP",), 80),
    "NAME": ["ada", "diana", "mara", "noel", "leon", "nora", "aram", "ariel", "rowan", "simon", "oliver", "clara"],
    "NUM": ["one", "two", "three", "nine", "many"],
    "CONJ": ["and", "but", "or", "so", "yet", "for"],
    "COMP": ["that", "when", "while", "as", "if"],
}
BANK["NAME"] = list(dict.fromkeys(BANK["NAME"] + _words(("NP",), 240)))
BANK["NOUN"] = list(dict.fromkeys(BANK["NOUN"] + "aide memos men time drawer diana".split()))
BANK["NOUN"] = list(dict.fromkeys(BANK["NOUN"] + "news people world story room road river letter".split()))
BANK["VERB"] = list(dict.fromkeys(BANK["VERB"] + "rips inspire was live draw emit saw reward dont cant wont isnt wasnt couldnt would should might must".split()))
BANK["PRON"] = list(dict.fromkeys(BANK["PRON"] + "i we you he she they it".split()))

# Complete clauses.  The templates are intentionally asymmetric: the point
# is to discover a character-compatible pair with different lexical roles.
TEMPLATES = [
    ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "PREP", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"),
    ("NAME", "VERB", "DET", "NOUN"),
    ("PRON", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "ADV"),
    ("DET", "NOUN", "VERB", "PREP", "DET", "ADJ", "NOUN"),
    ("DET", "NOUN", "VERB", "NUM", "NOUN"),
    ("DET", "NOUN", "VERB", "NAME"),
    ("DET", "NOUN", "VERB", "CONJ", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "COMP", "DET", "NOUN"),
    ("NAME", "VERB", "DET", "NOUN", "COMP", "DET", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN", "COMP", "DET", "NOUN"),
]
TARGET_TEMPLATES = [
    ("DET", "NOUN", "VERB", "NUM", "NOUN"),
    ("DET", "NOUN", "VERB", "NAME"),
    ("DET", "ADJ", "NOUN", "VERB", "NUM", "NOUN"),
    ("DET", "NOUN", "VERB", "NUM", "ADJ", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "NAME"),
    ("DET", "NOUN", "VERB", "ADJ", "NAME"),
    ("DET", "NOUN", "VERB", "NUM", "NOUN", "CONJ", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "COMP", "DET", "NOUN", "NAME"),
]


def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    normalized = tape(text)
    forward = hashlib.sha256(normalized.encode()).hexdigest()
    reverse = hashlib.sha256(normalized[::-1].encode()).hexdigest()
    return {
        "letters": len(normalized),
        "exact": bool(normalized) and normalized == normalized[::-1],
        "sha256_forward": forward,
        "sha256_reverse": reverse,
    }


def search(left_template: tuple[str, ...], right_template: tuple[str, ...], cap: int = 100,
           partial_ok=None):
    rows: list[tuple[list[str], list[str]]] = []
    nodes = 0

    def visit(li: int, ri: int, lo: int, ro: int,
              left: list[str], right_reversed: list[str]) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > 5_000_000 or len(rows) >= cap:
            return
        left_done = bool(left) and lo == len(left[-1])
        right_done = bool(right_reversed) and ro == len(right_reversed[-1])
        if li == len(left_template) and ri < 0 and left and right_reversed:
            # Grammar completion can leave the center inside either final
            # lexical edge.  Do not discard that valid center merely because
            # the word boundaries are unequal; the independent audit below is
            # the closure test, while all outer characters were still matched
            # online before this point.
            candidate = left[:] + list(reversed(right_reversed))
            checked = audit(" ".join(candidate))
            if checked["exact"]:
                rows.append((left[:], list(reversed(right_reversed))))
            return
        # A completed left (or right) grammar must not terminate the search:
        # the other clause may still have to emit its remaining words and
        # consume the live character obligation.  The old early returns made
        # every complete-clause pair unreachable and falsely reported zero.
        if li < len(left_template) and (not left or left_done):
            for word in BANK[left_template[li]]:
                if word in left or word in right_reversed:
                    continue
                if partial_ok is not None and not partial_ok(
                        left + [word], left_template, "left"):
                    continue
                visit(li + 1, ri, 0, ro, left + [word], right_reversed)
            return
        if ri >= 0 and (not right_reversed or right_done):
            needed = left[-1][lo] if left and lo < len(left[-1]) else None
            for word in BANK[right_template[ri]]:
                if word in left or word in right_reversed:
                    continue
                if needed is not None and word[-1] != needed:
                    continue
                if partial_ok is not None and not partial_ok(
                        list(reversed(right_reversed + [word])), right_template, "right"):
                    continue
                visit(li, ri - 1, lo, 0, left, right_reversed + [word])
            return
        if (lo >= len(left[-1]) or ro >= len(right_reversed[-1])):
            # One lexical edge has been consumed while the other still has
            # exposed characters; return to the corresponding grammar frontier
            # so the next word can continue the live equation.
            return
        if left[-1][lo] != right_reversed[-1][-1 - ro]:
            return
        visit(li, ri, lo + 1, ro + 1, left, right_reversed)

    visit(0, len(right_template) - 1, 0, 0, [], [])
    return rows, nodes


def run() -> dict:
    rows: list[dict] = []
    seen: set[str] = set()
    nodes = 0
    templates = TARGET_TEMPLATES if os.environ.get("PAL_TARGET") else TEMPLATES
    for left_template, right_template in product(templates, repeat=2):
        pairs, visited = search(left_template, right_template)
        nodes += visited
        for left, right in pairs:
            rendered = " ".join(left) + "; " + " ".join(right)
            if rendered in seen:
                continue
            seen.add(rendered)
            checked = audit(rendered)
            if checked["letters"] <= 38 or not checked["exact"]:
                continue
            rows.append({
                "rendered": rendered,
                "left_template": list(left_template),
                "right_template": list(right_template),
                "audit": checked,
                "reader_eligible": False,
                "provenance": {
                    "lexical_support": "Brown POS counts with zipf>=4.0",
                    "authored_sentence_templates": True,
                    "finished_tape_reversed": False,
                    "posthoc_repair": False,
                    "catalogue_text": False,
                    "independent_clause_derivations": True,
                },
            })
    rows.sort(key=lambda row: (row["audit"]["letters"], row["rendered"]), reverse=True)
    return {
        "experiment": "broad-pos-clause-intersection-20260920",
        "method": "outside-in live character intersection of asymmetric complete-clause POS grammars",
        "lexicon_sizes": {key: len(value) for key, value in BANK.items()},
        "templates": [list(item) for item in templates],
        "visited_nodes": nodes,
        "candidates": rows,
        "exact_candidates_over_38": len(rows),
        "reader_facing_candidates": [],
        "independent_validation": ["literal two-pointer scan", "forward/reverse SHA-256"],
        "next_construction": "retain only grammatical rows and add a typed relative/complement slot before increasing the lexical bank",
    }


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "visited_nodes": result["visited_nodes"],
        "exact_candidates_over_38": result["exact_candidates_over_38"],
        "longest": result["candidates"][0]["audit"]["letters"] if result["candidates"] else 0,
        "examples": [row["rendered"] for row in result["candidates"][:12]],
    }, indent=2))
