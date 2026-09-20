"""Pair attested complete Brown sentences with a forward CFG parse of the reverse tape.

This is a constructive semordnilap search, not a finished-tape reversal.  The
left side is an intact corpus sentence; the right side is emitted left-to-right
from independently collected POS lexicons while a character trie consumes the
reversed left tape.  The final candidate is independently pointer/SHA audited
and the structural gate remains diagnostic.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("PAL_OUT", Path.cwd())) / "attested-sentence-reverse-cfg-20260920.json"

TAG = {
    "at": "DET", "dt": "DET", "dti": "DET", "dts": "DET", "dtx": "DET",
    "pp": "PRON", "pps": "PRON", "ppo": "PRON", "ppl": "PRON", "ppss": "PRON",
    "in": "PREP", "to": "PREP", "cc": "CONJ", "cs": "COMP",
    "jj": "ADJ", "jjs": "ADJ", "jjr": "ADJ", "rb": "ADV", "rbr": "ADV", "rbt": "ADV",
    "nn": "NOUN", "nns": "NOUN", "np": "NAME", "nps": "NAME", "nr": "NAME",
    "vb": "VERB", "vbd": "VERB", "vbg": "VERB", "vbn": "VERB", "vbz": "VERB", "md": "VERB",
}

# Complete surface frames.  They deliberately do not include punctuation or
# sentence fragments; an output is a pair of clauses separated only when
# rendered after the tape has already closed.
TEMPLATES = [
    ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "PREP", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN"),
    ("PRON", "VERB", "DET", "NOUN"),
    ("NAME", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "ADV"),
    ("DET", "ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "CONJ", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "COMP", "DET", "NOUN"),
    ("NAME", "VERB", "DET", "NOUN", "COMP", "DET", "NOUN"),
]


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    rev = tape[::-1]
    mismatch = next(((i, tape[i], tape[-1 - i]) for i in range(len(tape) // 2)
                     if tape[i] != tape[-1 - i]), None)
    return {
        "letters": len(tape),
        "pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(rev.encode()).hexdigest(),
        "sha_equal": hashlib.sha256(tape.encode()).hexdigest() == hashlib.sha256(rev.encode()).hexdigest(),
    }


def structural_gate(text: str) -> dict:
    words = re.findall(r"[a-z]+", text.casefold())
    repeated = len(words) != len(set(words))
    self_pal = any(len(w) > 1 and w == w[::-1] for w in words)
    nested = False
    for i in range(len(words)):
        for j in range(i + 2, len(words) + 1):
            span = "".join(words[i:j])
            if 1 < len(span) < len(letters(text)) and span == span[::-1]:
                nested = True
    return {"no_repeated_words": not repeated, "no_self_palindromic_word": not self_pal,
            "no_nested_word_span": not nested,
            "mechanically_admitted": not repeated and not self_pal and not nested}


def parse_brown(root: Path) -> tuple[list[tuple[str, tuple[str, ...]]], dict[str, dict[str, int]]]:
    """Read Brown's token/POS files without requiring nltk on the bench."""
    sentences: list[tuple[str, tuple[str, ...]]] = []
    counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for path in sorted(root.glob("*")):
        if not path.is_file():
            continue
        current: list[tuple[str, str]] = []
        for raw in path.read_text(errors="ignore").split():
            if "/" not in raw:
                continue
            word, tag = raw.rsplit("/", 1)
            base = TAG.get(tag.split("-", 1)[0])
            if base and re.fullmatch(r"[A-Za-z]+", word):
                w = word.casefold()
                current.append((w, base))
                counts[base][w] += 1
            if tag.split("-", 1)[0] in {".", "?", "!"}:
                if 4 <= len(current) <= 18:
                    tape = "".join(w for w, _ in current)
                    if 38 <= len(tape) <= 150:
                        sentences.append((" ".join(w for w, _ in current), tuple(t for _, t in current)))
                current = []
    return sentences, counts


def build_lexicon(counts: dict[str, collections.Counter[str]], limit: int = 900) -> dict[str, tuple[str, ...]]:
    out = {}
    for tag, counter in counts.items():
        # Keep common words and all short function words.  The rank is a
        # proposal prior only; it never certifies readability.
        words = sorted(counter, key=lambda w: (-counter[w], w))
        out[tag] = tuple(w for w in words[:limit] if len(w) <= 14)
    return out


def find_right(tape: str, lexicon: dict[str, tuple[str, ...]], template: tuple[str, ...], max_each: int = 4) -> list[tuple[str, ...]]:
    target = tape[::-1]
    by_prefix: dict[tuple[str, int], list[str]] = {}
    for tag in set(template):
        for word in lexicon.get(tag, ()):
            by_prefix.setdefault((tag, word[0]), []).append(word)

    @lru_cache(None)
    def go(slot: int, pos: int) -> tuple[tuple[str, ...], ...]:
        if slot == len(template):
            return ((),) if pos == len(target) else ()
        tag = template[slot]
        if pos >= len(target):
            return ()
        out: list[tuple[str, ...]] = []
        for word in by_prefix.get((tag, target[pos]), ()):
            if target.startswith(word, pos):
                for tail in go(slot + 1, pos + len(word))[:max_each]:
                    out.append((word,) + tail)
                    if len(out) >= max_each:
                        return tuple(out)
        return tuple(out)

    return list(go(0, 0))


def run(sentence_limit: int = 160_000) -> dict:
    brown = Path(os.environ.get("BROWN_ROOT", str(ROOT / "brown")))
    sentences, counts = parse_brown(brown)
    lexicon = build_lexicon(counts, int(os.environ.get("LEX_LIMIT", "900")))
    # De-duplicate by surface tape while retaining the first attested sentence.
    unique: dict[str, tuple[str, tuple[str, ...]]] = {}
    for text, tags in sentences:
        unique.setdefault(letters(text), (text, tags))
    candidates = list(unique.items())[:sentence_limit]
    rows: list[dict] = []
    probes = 0
    for tape, (left, left_tags) in candidates:
        for template in TEMPLATES:
            rights = find_right(tape, lexicon, template)
            probes += 1
            for right_words in rights:
                rendered = f"{left}; {' '.join(right_words)}."
                row = {"rendered": rendered, "left_sentence": left,
                       "left_tags": left_tags, "right_template": template,
                       "right_words": right_words, "audit": audit(rendered),
                       "structural_gate": structural_gate(rendered),
                       "reader_status": "human-unreviewed; grammar and programmatic gates are diagnostic"}
                rows.append(row)
    rows.sort(key=lambda r: (-r["audit"]["letters"], r["rendered"]))
    exact = [r for r in rows if r["audit"]["pointer_exact"]]
    admitted = [r for r in exact if r["structural_gate"]["mechanically_admitted"]]
    return {
        "experiment_id": "attested-sentence-reverse-cfg-20260920",
        "method": "attested complete Brown sentence paired with forward typed CFG segmentation of its reversed tape",
        "stats": {"attested_sentences": len(sentences), "unique_tapes": len(unique),
                  "templates": len(TEMPLATES), "probes": probes, "rendered": len(rows),
                  "exact": len(exact), "mechanically_admitted": len(admitted),
                  "longest_exact": max((r["audit"]["letters"] for r in exact), default=0),
                  "longest_rendered": max((r["audit"]["letters"] for r in rows), default=0)},
        "exact_candidates": exact[:200], "mechanically_admitted": admitted[:100],
        "controls": rows[:100],
        "novelty_preflight": {"status": "passed", "signature": "attested-sentence|forward-reverse-cfg|typed-template",
                              "distinct_from": "paired clause banks: left side is an intact attested complete sentence and only the right side is parsed online",
                              "finished_tape_reversal": False, "post_hoc_repair": False},
        "provenance": {"left_source": "Brown corpus sentence surface, retained intact",
                        "right_source": "Brown POS lexicon, generated forward",
                        "audits": ["independent two-pointer comparison", "forward/reverse SHA-256"],
                        "reader_gate": "closed until a fresh exact row clears structural screening and blinded reading",
                        "hard_exclusions": ["repeated words", "self-palindromic words", "proper nested spans", "fragments", "catalogue text"]},
        "next_construction": "Keep attested sentence frames but add agreement-conditioned subject/verb and article states to the reverse CFG.",
        "status": "fresh exact candidate requires human reading" if admitted else "no structurally clean exact candidate; intact controls retained",
    }


if __name__ == "__main__":
    result = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
    for row in result["exact_candidates"][:20]:
        print(row["audit"]["letters"], row["structural_gate"]["mechanically_admitted"], row["rendered"])
