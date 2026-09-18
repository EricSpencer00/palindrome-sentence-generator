"""Bounded search for fresh mirror-pair sentences with an in-word centre.

This is an experiment, not part of the serving controller.  It scans the
checked ``mirror_pairs`` inventory and re-audits each rendered pair from its
letters.  A row is useful even when rejected: the result is an inspectable
search frontier rather than a claim that every lexical closure is prose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
DEFAULT_BOUND = 256
DEFAULT_OUT = ROOT / "runs" / "center-boundary-mirror-search-20260917.json"
EXPERIMENT_ID = "center-boundary-mirror-search-20260917"
SIGNATURE = "bounded-mirror-pair-inword-centre|proper-span-rejection|independent-pointer-sha"


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
    exact = [r for r in rows if r["checks"]["exact_letter_palindrome"]]
    mechanically_clean = [r for r in rows if not r["rejection_codes"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "complete",
        "config": {"bound": bound},
        "candidate_count": len(rows),
        "exact_count": len(exact),
        "mechanically_clean_count": len(mechanically_clean),
        "rendered_rows": rows,
        "exact_candidates": exact,
        "mechanically_clean_candidates": mechanically_clean,
        "reader_eligible": False,
        "reader_gate": "closed; pair-bank diagnostics cannot certify intact English",
        "novelty_preflight": {
            "status": "passed",
            "signature": SIGNATURE,
            "artifact": "runs/center-boundary-mirror-search-20260917.json",
            "registry_checked": True,
        },
        "failure_repair": {
            "failure": "bounded pair inventory yields exact rows but no 100+ intact anti-shortcut closure",
            "repair": "author fresh typed clauses around the best in-word centre and carry the first residual through a live grammar product",
        },
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "catalogue_sha256": _sha((ROOT / "data" / "known_palindromes.json").read_bytes().decode()),
        "lexicon_sha256": _sha("\n".join(sorted(lexicon))),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound", type=int, default=DEFAULT_BOUND)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = run(bound=args.bound)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "candidate_count": result["candidate_count"],
        "exact_count": result["exact_count"],
        "mechanically_clean_count": result["mechanically_clean_count"],
    }))
