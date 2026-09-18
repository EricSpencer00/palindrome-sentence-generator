"""Bounded search for fresh mirror-pair sentences with an in-word centre.

This is an experiment, not part of the serving controller.  It scans the
checked ``mirror_pairs`` inventory and re-audits each rendered pair from its
letters.  A row is useful even when rejected: the result is an inspectable
search frontier rather than a claim that every lexical closure is prose.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
DEFAULT_BOUND = 256


def normalize(text: str) -> str:
    return "".join(c for c in text.lower() if "a" <= c <= "z")


def _words(text: str) -> tuple[str, ...]:
    return tuple(WORD_RE.findall(text.lower()))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _word_order_mirror(words: tuple[str, ...]) -> bool:
    n = len(words)
    return n > 1 and n % 2 == 0 and words[: n // 2] == words[n // 2 :][::-1]


def _proper_multiword_palindromic_spans(words: tuple[str, ...]) -> list[dict[str, object]]:
    spans = []
    for start in range(len(words)):
        for end in range(start + 2, len(words) + 1):
            text = " ".join(words[start:end])
            # The complete sentence is expected to be palindromic; only a
            # *proper* subspan is a forbidden self-palindromic shortcut.
            if (start, end) != (0, len(words)) and normalize(text) == normalize(text)[::-1]:
                spans.append({"start": start, "end": end, "text": text})
    return spans


def audit_rendered(left: str, right: str, *, catalogue: set[str], lexicon: set[str]) -> dict[str, object]:
    rendered = f"{left} {right}".strip()
    tape = normalize(rendered)
    words = _words(rendered)
    center = len(tape) // 2
    offsets = []
    cursor = 0
    for word in words:
        offsets.append((cursor, cursor + len(word)))
        cursor += len(word)
    center_inside_word = any(start < center < end for start, end in offsets)
    spans = _proper_multiword_palindromic_spans(words)
    repeated = len(set(words)) != len(words)
    checks = {
        "exact_letter_palindrome": bool(tape) and tape == tape[::-1],
        "center_inside_word": center_inside_word,
        "no_proper_multiword_palindromic_span": not spans,
        "not_word_order_mirror": not _word_order_mirror(words),
        "no_repeated_content": not repeated,
        "all_words_in_lexicon": all(w in lexicon for w in words),
        "catalogue_absent": tape not in catalogue,
    }
    return {
        "rendered": rendered,
        "left": left,
        "right": right,
        "normalized": tape,
        "sha256": _sha(tape),
        "letters": len(tape),
        "center_offset": center,
        "words": list(words),
        "proper_multiword_palindromic_spans": spans,
        "checks": checks,
        "rejection_codes": [k for k, ok in checks.items() if not ok],
    }


def _independent_two_pointer(tape: str) -> bool:
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            return False
        left += 1
        right -= 1
    return bool(tape)


def run(*, bound: int = DEFAULT_BOUND) -> dict[str, object]:
    pairs = json.loads((ROOT / "data" / "mirror_pairs.json").read_text())
    lexicon = set((ROOT / "data" / "lexicon.txt").read_text().split())
    catalogue = set(json.loads((ROOT / "data" / "known_palindromes.json").read_text()))
    rows = []
    for index, pair in enumerate(pairs[: max(0, bound)], 1):
        left, right = " ".join(pair["left"]), " ".join(pair["right"])
        row = audit_rendered(left, right, catalogue=catalogue, lexicon=lexicon)
        tape = row["normalized"]
        row.update({
            "pair_index": index,
            "provenance": "data/mirror_pairs.json -> rendered left/right -> fresh lexical audit",
            "independent_two_pointer_exact": _independent_two_pointer(tape),
            "independent_sha_exact": _sha(tape) == _sha(tape[::-1]),
            "source_pair_sha256": _sha(json.dumps(pair, sort_keys=True)),
        })
        rows.append(row)
    exact = [r for r in rows if not r["rejection_codes"]]
    return {
        "status": "complete",
        "config": {"bound": bound},
        "candidate_count": len(rows),
        "exact_count": len(exact),
        "rendered_rows": rows,
        "exact_candidates": exact,
        "failure_repair": {"failure": "no qualifying center-boundary row", "repair": "expand bound or author fresh lexical pairs"},
        "catalogue_sha256": _sha((ROOT / "data" / "known_palindromes.json").read_bytes().decode()),
        "lexicon_sha256": _sha("\n".join(sorted(lexicon))),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
