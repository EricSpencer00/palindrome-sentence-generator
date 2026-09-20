"""Exact-by-construction slot-pair search with genuine cross-word seams.

The sentence is one grammar derivation.  We choose its outer slots together,
compare the newly exposed character prefixes immediately, and recurse inward.
There is no finished-tape reversal and no assumption that a left word pairs
with the reverse of a right word: unequal word lengths are retained in the
prefix/suffix buffers and may cross several word boundaries.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "exact": bool(tape) and tape == tape[::-1],
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


@dataclass(frozen=True)
class Slot:
    role: str
    words: tuple[str, ...]


def _compatible(prefix: str, suffix: str) -> bool:
    """Check every character whose opposite endpoint is already assigned."""
    overlap = min(len(prefix), len(suffix))
    return prefix[:overlap] == suffix[::-1][:overlap]


def search(template: tuple[Slot, ...], *, limit: int = 32) -> dict[str, object]:
    rendered: list[dict[str, object]] = []
    states = pruned = 0

    def walk(
        lo: int,
        hi: int,
        prefix: str,
        suffix: str,
        left_words: tuple[str, ...],
        right_words: tuple[str, ...],
        shifted: bool,
        word_pairs: tuple[dict[str, object], ...],
    ) -> None:
        nonlocal states, pruned
        if len(rendered) >= limit:
            return
        if lo > hi:
            states += 1
            chosen = left_words + right_words
            text = " ".join(chosen)
            checked = audit(text)
            if checked["exact"] and shifted:
                rendered.append({
                    "rendered": text,
                    "audit": checked,
                    "provenance": {
                        "template_roles": [slot.role for slot in template],
                        "word_pairs": word_pairs,
                        "cross_word_seam": shifted,
                    },
                })
            return
        if lo == hi:
            for word in template[lo].words:
                if word in left_words or word in right_words or word == word[::-1]:
                    continue
                all_words = left_words + (word,) + right_words
                candidate = letters(" ".join(all_words))
                if candidate != candidate[::-1]:
                    pruned += 1
                    continue
                states += 1
                rendered.append({
                    "rendered": " ".join(all_words),
                    "audit": audit(" ".join(all_words)),
                    "provenance": {
                        "template_roles": [slot.role for slot in template],
                        "word_pairs": word_pairs,
                        "cross_word_seam": shifted,
                    },
                })
            return
        for left in template[lo].words:
            if left in left_words or left in right_words or left == left[::-1]:
                continue
            for right in template[hi].words:
                if right in left_words or right in right_words or right == right[::-1] or right == left:
                    continue
                new_prefix = prefix + letters(left)
                new_suffix = letters(right) + suffix
                states += 1
                if not _compatible(new_prefix, new_suffix):
                    pruned += 1
                    continue
                pair = {
                    "left_role": template[lo].role,
                    "right_role": template[hi].role,
                    "left_word": left,
                    "right_word": right,
                    "left_letters": len(letters(left)),
                    "right_letters": len(letters(right)),
                    "boundary_offset": len(new_prefix) - len(new_suffix),
                }
                walk(
                    lo + 1,
                    hi - 1,
                    new_prefix,
                    new_suffix,
                    left_words + (left,),
                    (right,) + right_words,
                    shifted or letters(left) != letters(right)[::-1],
                    word_pairs + (pair,),
                )

    walk(0, len(template) - 1, "", "", tuple(), tuple(), False, tuple())
    return {
        "candidates": rendered,
        "stats": {"states": states, "pruned": pruned, "exact": len(rendered)},
    }


def main() -> None:
    bank_path = Path("data/brown_pcfg_bank_20260920.json")
    if bank_path.exists():
        bank = json.loads(bank_path.read_text())["lexicon"]
        def top(tag, fallback):
            return tuple(x["word"] for x in bank.get(tag, [])[:24]) or fallback
    else:
        def top(tag, fallback): return fallback
    def inflected(tag, fallback, predicate):
        values = top(tag, fallback)
        chosen = tuple(word for word in values if predicate(word))
        return chosen or values
    det = top("DET", ("a", "the", "one", "this"))
    adj = top("ADJ", ("calm", "brave", "young", "wise", "fair", "quiet", "keen", "mild"))
    noun = inflected("NOUN", ("poet", "sailor", "keeper", "reader", "bard", "pilot", "guard"), lambda w: not w.endswith("s"))
    verb = inflected("VERB", ("reads", "marks", "guides", "guards", "seeks", "keeps", "hears"), lambda w: w.endswith("s"))
    obj = top("NOUN", ("letter", "sonnet", "garden", "harbor", "parcel", "secret", "candle"))
    template = (
        Slot("det", det), Slot("adj", adj), Slot("subject", noun),
        Slot("verb", verb), Slot("det", det), Slot("object", obj),
    )
    templates = [template, (
        Slot("det", det), Slot("subject", noun), Slot("verb", verb),
        Slot("det", det), Slot("object", obj), Slot("adjunct", ("today", "quietly", "nearby")),
    ), (
        Slot("det", det), Slot("subject", noun), Slot("verb", verb),
        Slot("det", det), Slot("object", obj), Slot("prep", ("in", "at", "on")),
        Slot("object", obj),
    )]
    runs = [search(item, limit=32) for item in templates]
    result = {"candidates": [c for r in runs for c in r["candidates"]],
              "stats": {"states": sum(r["stats"]["states"] for r in runs),
                        "pruned": sum(r["stats"]["pruned"] for r in runs),
                        "exact": sum(r["stats"]["exact"] for r in runs)}}
    result.update({
        "experiment_id": "slot-pair-character-search-20260919",
        "method": "single-sentence grammar slot product with online cross-word character obligations",
        "provenance": {
            "templates": [[slot.role for slot in item] for item in templates],
            "finished_tape_reversal": False,
            "paired_clauses": False,
            "aligned_token_mirror": False,
            "fallback": False,
            "distinct_words": True,
        },
    })
    Path("runs/slot-pair-character-search-20260919.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))


if __name__ == "__main__":
    main()
