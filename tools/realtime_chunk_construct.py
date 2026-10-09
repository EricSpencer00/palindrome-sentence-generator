"""Run the existing saved-chunk constructors without the retired serving API.

Standard library only. Refrain repeats each outer input exactly once; paired
nests distinct generated mirror halves under the existing hierarchy guards.
Neither mode claims reader validation or newly discovered source sentences.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_palindrome.hierarchy import (sentence_centres, sentence_pairs,
                                      select_sentence_pairs)
from llm_palindrome.paragraphs import harvest, refrain, render
from llm_palindrome.spelling import spell
from llm_palindrome.themes import (content_words, is_question,
                                   order_for_refrain, trim_to_theme)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def letter_tape(text: str) -> str:
    """Reject unsupported alphabetic characters before normalizing ASCII."""
    if any(character.isalpha() and not character.isascii() for character in text):
        raise ValueError("unsupported non-ASCII alphabetic character")
    return "".join(character.lower() for character in text
                   if character.isascii() and character.isalpha())


def exact_audit(text: str) -> dict:
    try:
        tape = letter_tape(text)
    except ValueError as exc:
        return {"supported_ascii_letters": False, "exact": False,
                "reason": str(exc), "letters": 0}
    two_pointer = bool(tape) and all(tape[i] == tape[-1 - i]
                                    for i in range(len(tape) // 2))
    return {"supported_ascii_letters": True, "nonempty": bool(tape),
            "letters": len(tape), "exact": two_pointer and tape == tape[::-1],
            "two_pointer_exact": two_pointer,
            "normalized_sha256": sha256(tape.encode("ascii")),
            "reversed_sha256": sha256(tape[::-1].encode("ascii")),
            "surface_sha256": sha256(text.encode("utf-8"))}


def theme_text(text: str) -> str:
    # This is only an internal theme key; output keeps the original wording.
    return "".join(c.lower() for c in text
                   if c.isascii() and (c.isalpha() or c.isspace()))


class ChunkConstructor:
    """Load and compile the existing material once, then make warm requests."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self._source_hashes: dict[str, str] = {}
        self._counts: dict[str, int] = {}
        self._excluded: list[dict] = []
        self._chunks: dict[str, dict] = {}
        self._bank: list[dict] = []
        self._curated: set[str] = set()

        centres_path = "data/centres.json"
        centres = self._read(centres_path)
        self._counts["curated_centres_input"] = len(centres)
        for index, text in enumerate(centres):
            row = self._add(text, "catalogue", centres_path, index, {"text": text})
            if row is not None:
                self._curated.add(row["tape"])

        canon_path = "data/canon_spelled.json"
        canon = self._read(canon_path)
        self._counts["catalogue_input"] = len(canon)
        for key, text in canon.items():
            self._add(text, "catalogue", canon_path, key,
                      {"catalogue_key": key, "text": text})

        reddit_path = "data/reddit-palindromes/2026-10-09/posts.json"
        reddit = self._read(reddit_path)
        posts = reddit["rows"]
        self._counts["reddit_input"] = len(posts)
        self._reddit_context = {k: deepcopy(v) for k, v in reddit.items() if k != "rows"}
        for index, record in enumerate(posts):
            self._add(record["text"], "reddit", reddit_path, index, record)

        bank_path = "data/v3_bank.json"
        bank = self._read(bank_path)
        self._counts["v3_bank_input"] = len(bank)
        for index, record in enumerate(bank):
            kind = "project_generated" if record["source"] == "generated" else "catalogue"
            row = self._add(record["text"], kind, bank_path, index, record)
            if row is not None and kind == "project_generated":
                # The frozen v3 material is already space-delimited word units.
                words = theme_text(record["text"]).split()
                self._bank.append({"words": words, "source": kind,
                                   "chunk": row, "record": deepcopy(record)})

        brown_path = "tools/polaris/payload/brown.json.gz"
        brown = self._read(brown_path, compressed=True)
        self._table = {word: frozenset(tags) for word, tags in brown["table"].items()}
        self._shapes = {tuple(shape) for shape in brown["shapes"]}
        self._trigrams = {tuple(key.split("|")): value
                          for key, value in brown["trigrams"].items()}
        raw_pairs = []
        seen_halves: set[str] = set()
        self._pair_sources: dict[tuple[str, str], dict] = {}
        for row in self._bank:
            got = harvest([" ".join(row["words"])]).pairs
            if not got:
                continue
            left, right = got[0]
            keys = (letter_tape(" ".join(left)), letter_tape(" ".join(right)))
            if keys[0] == keys[1] or any(key in seen_halves for key in keys):
                continue
            seen_halves.update(keys)
            raw_pairs.append((left, right, row["source"]))
            self._pair_sources[(" ".join(left), " ".join(right))] = row["chunk"]
        self._pairs = sentence_pairs(raw_pairs, self._table, self._shapes, self._trigrams)
        self._centres = sentence_centres(self._bank, self._table, self._shapes, self._trigrams)
        self._counts.update(exact_distinct_chunks=len(self._chunks),
                            curated_exact_chunks=len(self._curated),
                            generated_exact_rows=len(self._bank),
                            nondegenerate_generated_pairs=len(raw_pairs),
                            sentence_pairs=len(self._pairs),
                            sentence_centres=len(self._centres),
                            excluded_input_rows=len(self._excluded))

    def _read(self, relative_path: str, compressed: bool = False):
        value = (self.root / relative_path).read_bytes()
        self._source_hashes[relative_path] = sha256(value)
        return json.loads(gzip.decompress(value) if compressed else value)

    def _add(self, text, kind, path, index, metadata):
        audit = exact_audit(text)
        lineage = {"source_kind": kind, "input_path": path,
                   "input_sha256": self._source_hashes[path], "input_index": index,
                   "original_text": text, "original_text_sha256": sha256(text.encode("utf-8")),
                   "metadata": deepcopy(metadata)}
        if not audit["exact"]:
            self._excluded.append({**lineage, "audit": audit})
            return None
        tape = letter_tape(text)
        if tape in self._chunks:
            self._chunks[tape]["lineage"].append(lineage)
        else:
            self._chunks[tape] = {"text": text, "tape": tape,
                                  "chunk_id": audit["normalized_sha256"],
                                  "lineage": [lineage]}
        return self._chunks[tape]

    @property
    def inventory(self) -> dict:
        return {"counts": dict(self._counts), "source_hashes": dict(self._source_hashes),
                "dependencies": {"python": "standard library", "packages": [],
                                 "modules": ["llm_palindrome.hierarchy", "llm_palindrome.paragraphs",
                                             "llm_palindrome.spelling", "llm_palindrome.themes"]},
                "excluded_inputs": deepcopy(self._excluded),
                "reddit_collection": deepcopy(self._reddit_context)}

    def construct(self, *, mode="refrain", target_letters=500, units=3,
                  anchor=None, seed=0) -> dict:
        if mode not in ("refrain", "paired"):
            raise ValueError("mode must be refrain or paired")
        if not isinstance(target_letters, int) or target_letters < 1:
            raise ValueError("target_letters must be a positive integer")
        if not isinstance(units, int) or units < 1:
            raise ValueError("units must be a positive integer")
        if anchor is not None:
            anchor = anchor.lower().strip()
            if not anchor.isascii() or not anchor.isalpha():
                raise ValueError("anchor must be one ASCII word")
        request = {"mode": mode, "target_letters": target_letters,
                   "units": units, "anchor": anchor, "seed": seed}
        if mode == "refrain":
            chunks, stats, reason = self._refrain(target_letters, units, anchor, seed)
        else:
            chunks, stats, reason = self._paired(target_letters, units, anchor, seed)
        text = "\n".join(chunk["text"] for chunk in chunks)
        audit = exact_audit(text)
        tapes = [letter_tape(chunk["text"]) for chunk in chunks]
        counts = Counter(tapes)
        adjacent = any(a == b for a, b in zip(tapes, tapes[1:]))
        if chunks and (not audit["exact"] or audit["letters"] > target_letters
                       or adjacent or max(counts.values()) > (2 if mode == "refrain" else 1)):
            raise AssertionError("chunk construction invariant failed")
        sources = sorted({source["source_kind"] for chunk in chunks for source in chunk["lineage"]})
        return {"status": "constructed" if chunks else "no_construction", "reason": reason,
                "request": request, "text": text, "letters": audit["letters"],
                "undershoot_letters": target_letters - audit["letters"],
                "exactness": audit, "chunks": chunks, "layout": deepcopy(chunks),
                "chunk_count": len(chunks), "distinct_chunks": len(counts),
                "max_chunk_uses": max(counts.values(), default=0),
                "adjacent_duplicate_chunks": adjacent, "source_kinds": sources,
                "source_hashes": dict(self._source_hashes),
                "novel": False if any(s in sources for s in ("catalogue", "reddit")) else None,
                "novelty": {"new_source_chunks": False, "reuses_existing_material": bool(chunks),
                            "new_composition_novelty": "not evaluated"},
                "human_readability": "not evaluated", "selection": stats}

    def _occurrence(self, row, text, slot, role, structural_role, **extra):
        return {"slot": slot, "role": role, "structural_role": structural_role,
                "text": text, "chunk_id": sha256(letter_tape(text).encode("ascii")),
                "letters": len(letter_tape(text)), "lineage": deepcopy(row["lineage"]), **extra}

    def _refrain(self, target, units, anchor, seed):
        pool = [row for row in self._chunks.values() if len(row["tape"]) <= target]
        if anchor:
            pool = [row for row in pool if anchor in content_words(theme_text(row["text"]))]
        eligible_total = len(pool)
        curated = [row for row in pool if row["tape"] in self._curated]
        pool = curated or pool
        if not pool:
            return [], {}, "no eligible centre fits the target and anchor"
        by_theme = {theme_text(row["text"]): row for row in pool}
        theme = trim_to_theme(list(by_theme), anchor=anchor)
        random.Random(seed).shuffle(theme)
        ordered = order_for_refrain(theme)
        centre = by_theme[ordered[-1]]
        remaining = target - len(centre["tape"])
        selected = []
        for key in ordered[:-1]:
            if len(selected) >= units - 1:
                break
            row = by_theme[key]
            if 2 * len(row["tape"]) <= remaining:
                selected.append(row)
                remaining -= 2 * len(row["tape"])
        selected.append(centre)
        sequence = refrain([row["text"] for row in selected])
        rows = selected + list(reversed(selected[:-1]))
        def presented(row):
            original = row["text"]
            # Public post titles remain literal, including their punctuation.
            if row["lineage"][0]["source_kind"] == "reddit" or original.endswith((".", "?", "!")):
                return original
            written = spell(original.split())
            if is_question(theme_text(original)) and written.endswith("."):
                written = written[:-1] + "?"
            if letter_tape(written) != letter_tape(original):
                raise AssertionError("presentation changed source letters")
            return written

        chunks = [self._occurrence(row, presented(row), index,
                                  "opener" if index < len(selected) - 1 else
                                  "turn" if index == len(selected) - 1 else "return",
                                  "left" if index < len(selected) - 1 else
                                  "centre" if index == len(selected) - 1 else "right",
                                  input_chunk_text=text,
                                  return_of=(len(sequence) - 1 - index)
                                  if index >= len(selected) else None)
                  for index, (row, text) in enumerate(zip(rows, sequence))]
        if letter_tape("\n".join(c["text"] for c in chunks)) != letter_tape("\n".join(sequence)):
            raise AssertionError("refrain presentation changed the source sequence")
        return chunks, {"algorithm": "themes.trim_to_theme + order_for_refrain + paragraphs.refrain",
                        "units_semantics": "maximum distinct whole chunks",
                        "eligible_total_indexed_chunks": eligible_total,
                        "curated_eligible_chunks": len(curated),
                        "eligible_chunks": len(pool), "theme_chunks": len(theme),
                        "curated_preferred": bool(curated),
                        "explicit_return": True}, None

    def _paired(self, target, units, anchor, seed):
        centres = [row for row in self._centres
                   if len(letter_tape(" ".join(row["words"]))) <= target
                   and (not anchor or anchor in content_words(" ".join(row["words"])))]
        if not centres:
            return [], {}, "no sentence centre fits the target and anchor"
        rng = random.Random(seed)
        pairs = list(self._pairs)
        rng.shuffle(pairs)
        centre = rng.choice(centres)
        centre_tape = letter_tape(" ".join(centre["words"]))
        pairs = [pair for pair in pairs if all(letter_tape(" ".join(half)) != centre_tape
                                              for half in pair[:2])]
        selected, stats = select_sentence_pairs(pairs, centre["words"], target,
                                                max_pairs=units)
        selected_pairs = [(pair["left"], pair["right"]) for pair in selected]
        rendered = render(selected_pairs, centre["words"])
        chunks = []
        for index, pair in enumerate(selected):
            source = self._pair_sources[(" ".join(pair["left"]), " ".join(pair["right"]))]
            chunks.append(self._occurrence(source, spell(pair["left"]), index,
                                           "opener", "left", pair=index))
        chunks.append(self._occurrence(centre["chunk"], spell(centre["words"]),
                                       len(chunks), "turn", "centre"))
        for index, pair in reversed(list(enumerate(selected))):
            source = self._pair_sources[(" ".join(pair["left"]), " ".join(pair["right"]))]
            chunks.append(self._occurrence(source, spell(pair["right"]), len(chunks),
                                           "return", "right", pair=index, return_of=index))
        if letter_tape(rendered) != letter_tape("\n".join(c["text"] for c in chunks)):
            raise AssertionError("layout changed the existing renderer's letters")
        return chunks, {"algorithm": "hierarchy.select_sentence_pairs + paragraphs.render",
                        "guard_policy": "historical hierarchy control",
                        "global_caps_define_useful_reuse": False,
                        "units_semantics": "maximum mirror pairs", "pairs": len(selected),
                        "guards": {"max_bigram_uses": 2, "max_template_uses": 2,
                                   "max_content_word_uses": 3}, "guard_stats": stats}, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("refrain", "paired"), default="refrain")
    parser.add_argument("--anchor", default=None)
    parser.add_argument("--target-letters", type=int, default=500)
    parser.add_argument("--units", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        constructor = ChunkConstructor()
        result = constructor.construct(mode=args.mode, target_letters=args.target_letters,
                                       units=args.units, anchor=args.anchor, seed=args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    result["inventory"] = constructor.inventory
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
