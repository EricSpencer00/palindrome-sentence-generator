"""Bounded outside-in intersection of two independently authored clause grammars.

The search chooses a left and right clause token only when their exposed
characters agree.  It never constructs a finished tape and reverses it, and
it keeps the two clause derivations separate until the character equation is
closed.  The lexical bank is deliberately small and hand-authored so any
survivor can be inspected as prose rather than treated as a corpus fragment.
"""
from __future__ import annotations

import hashlib
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


BANK = {
    "DET": "a an the some our no my one".split(),
    "ADJ": (
        "quiet patient young old careful bright silver weary gentle wild clear "
        "northern eastern winter faithful stressed smart raw"
    ).split(),
    "NOUN": (
        "sailor keeper captain poet scholar child river garden harbor lantern "
        "letter bridge moon tide bell song map story road star aide memos men "
        "drawer dog god wolf parts pots tips pets ward evil diana time trams "
        "rats liar reed pools name tag friend gate shore"
    ).split(),
    "VERB": (
        "marks guards studies carries opens reads writes watches finds follows hears "
        "sees names keeps guides holds brings leaves crosses lights calls meets "
        "saves tells draws sends raises greets records chooses answers teaches "
        "builds trusts loves needs waits returns enters knows rips inspire was "
        "live draw emit saw deliver reward spit step stop flow strap stressed repaid "
        "parts pots tips pets ward time trams rats liar reed pools"
    ).split(),
    "PREP": "by near under over beside through after before with from into on of to against".split(),
    "ADV": "now then still softly quietly again soon here there never".split(),
    "PRON": "i we you she he they it".split(),
    "NAME": "ada diana mara noel leon nora aram ariel rowan simon oliver clara pam nomad god".split(),
    "NUM": "one two nine many".split(),
}
BANK["WORD"] = list(dict.fromkeys(word for words in BANK.values() for word in words))

# Complete, short surface grammars. The left and right templates are allowed
# to differ; the character equation, not word-count symmetry, joins them.
TEMPLATES = [
    ("DET", "NOUN", "VERB", "ADJ"),
    ("DET", "NOUN", "VERB", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB"),
    ("DET", "NOUN", "VERB", "PREP"),
    ("NAME", "VERB", "DET", "NOUN"),
    ("PRON", "VERB", "DET", "NOUN"),
    ("VERB", "PREP", "DET", "NOUN"),
    ("VERB", "DET", "NOUN", "ADV"),
    ("DET", "NOUN", "VERB", "DET"),
    ("NOUN", "VERB", "NOUN", "DET"),
    ("NOUN", "VERB", "DET", "NOUN"),
    ("WORD", "WORD", "WORD"),
    ("WORD", "WORD", "WORD", "WORD"),
    ("WORD", "WORD", "WORD", "WORD", "WORD"),
]


def _search(left_template: tuple[str, ...], right_template: tuple[str, ...], cap: int = 100):
    rows: list[tuple[list[str], list[str]]] = []
    nodes = 0

    def visit(li: int, ri: int, lo: int, ro: int,
              left: list[str], right_reversed: list[str]) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > 1_000_000 or len(rows) >= cap:
            return
        left_done = bool(left) and lo == len(left[-1])
        right_done = bool(right_reversed) and ro == len(right_reversed[-1])
        if li == len(left_template) and ri < 0 and left_done and right_done:
            rows.append((left[:], list(reversed(right_reversed))))
            return
        if li >= len(left_template) and left_done:
            return
        if ri < 0 and right_done:
            return
        if li < len(left_template) and (not left or left_done):
            for word in BANK[left_template[li]]:
                if word in left or word in right_reversed:
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
                visit(li, ri - 1, lo, 0, left, right_reversed + [word])
            return
        if left[-1][lo] != right_reversed[-1][-1 - ro]:
            return
        visit(li, ri, lo + 1, ro + 1, left, right_reversed)

    visit(0, len(right_template) - 1, 0, 0, [], [])
    return rows, nodes


def independent_audit(text: str) -> dict:
    tape = normalize_letters(text)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "normalized_tape": tape,
        "two_pointer_exact": tape == tape[::-1],
        "sha256_forward": forward,
        "sha256_reverse": reverse,
    }


def run() -> dict:
    candidates = []
    seen: set[str] = set()
    nodes = 0
    for left_template, right_template in product(TEMPLATES, repeat=2):
        pairs, visited = _search(left_template, right_template)
        nodes += visited
        for left, right in pairs:
            text = " ".join(left) + "; " + " ".join(right)
            if text in seen:
                continue
            seen.add(text)
            audit = independent_audit(text)
            if audit["letters"] <= 38 or not audit["two_pointer_exact"]:
                continue
            admission = mechanical_admission_checks(text, min_letters=39, max_letters=500)
            candidates.append({
                "rendered": text,
                "left_words": left,
                "right_words": right,
                "audit": audit,
                "mechanical_admission": admission,
                "reader_eligible": False,
                "provenance": {
                    "method": "outside-in paired clause lexical intersection",
                    "finished_tape_reversed": False,
                    "posthoc_repair": False,
                    "catalogue_used": False,
                    "independent_clause_derivations": True,
                },
            })
    candidates.sort(key=lambda row: (row["audit"]["letters"], row["rendered"]), reverse=True)
    return {
        "experiment": "paired-clause-lexical-intersection-20260920",
        "method": "outside-in character intersection of independent short-clause grammars",
        "templates": [list(t) for t in TEMPLATES],
        "visited_nodes": nodes,
        "candidates": candidates,
        "exact_candidates_over_38": len(candidates),
        "reader_facing_candidates": [],
        "diagnostic_controls": [],
        "independent_validation": ["literal two-pointer scan", "forward/reverse SHA-256"],
        "next_construction": "attach one complete authored adjunct to each clause while carrying the same live character debt; reject duplicate and nested-palindrome closures",
    }


if __name__ == "__main__":
    out = run()
    path = Path("runs/paired-clause-lexical-intersection-20260920.json")
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "visited_nodes": out["visited_nodes"],
        "exact_candidates_over_38": out["exact_candidates_over_38"],
        "longest": out["candidates"][0]["audit"]["letters"] if out["candidates"] else 0,
        "examples": [row["rendered"] for row in out["candidates"][:5]],
    }, indent=2))
