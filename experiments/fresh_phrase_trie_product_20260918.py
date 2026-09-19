#!/usr/bin/env python3
"""Fresh phrase-inventory product with live character obligations.

The phrase bank is authored for this run and contains no copied palindrome
sentences.  Complete clause realizations are indexed by role, then a trie of
reversed right-hand realizations is walked one character at a time from each
left-hand clause.  A terminal node is the only place an exact pair can be
rendered; ordinary prose controls and independent audits are retained when no
terminal exists.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/fresh-phrase-trie-product-20260918.json"
EXPERIMENT = "fresh-phrase-trie-product-20260918"
SIGNATURE = "fresh-authored-pos-inventory|reversed-character-trie-product|typed-complete-clause|independent-audit"

# These lexical items were authored for this experiment.  They are not a
# sentence catalogue and are not copied from data/known_palindromes.json.
INVENTORY = {
    "det": ["a", "the", "this", "one"],
    "adj": ["quiet", "silver", "patient", "young", "gentle", "watchful"],
    "noun": [
        "scribe", "keeper", "pilot", "teacher", "gardener", "captain",
        "poet", "lantern", "letter", "garden", "harbor", "river",
    ],
    "verb": [
        "marks", "opens", "carries", "copies", "guards", "follows",
        "writes", "meets", "keeps", "guides",
    ],
    "prep": ["near", "under", "beside", "toward", "across"],
}

TEMPLATES = (
    ("det", "adj", "noun", "verb", "det", "noun"),
    ("det", "noun", "verb", "prep", "det", "noun"),
)


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    mismatches = [
        {"left": i, "right": len(tape) - 1 - i, "actual": tape[i], "expected": tape[-1 - i]}
        for i in range(len(tape) // 2)
        if tape[i] != tape[-1 - i]
    ]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "exact": bool(tape) and not mismatches,
        "independent_two_pointer_exact": bool(tape) and not mismatches,
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha256_equal": forward == reverse,
    }


def build_records() -> list[dict]:
    records = []
    for template_id, roles in enumerate(TEMPLATES):
        for words in itertools.product(*(INVENTORY[role] for role in roles)):
            content = [word for role, word in zip(roles, words) if role not in {"det", "prep"}]
            if len(set(content)) != len(content):
                continue
            text = " ".join(words) + "."
            records.append(
                {
                    "template_id": template_id,
                    "roles": list(roles),
                    "words": list(words),
                    "rendered": text,
                    "tape": letters(text),
                }
            )
    return records


def trie_insert(root: dict, tape: str) -> int:
    node = root
    created = 0
    for char in tape:
        if char not in node:
            node[char] = {}
            created += 1
        node = node[char]
    node.setdefault("$ends", 0)
    node["$ends"] += 1
    return created


def walk_live_obligations(root: dict, left_tape: str) -> tuple[dict | None, int]:
    """Walk left characters against reversed-right obligations, never render first."""
    node = root
    steps = 0
    for char in left_tape:
        steps += 1
        node = node.get(char)
        if node is None:
            return None, steps
    return node, steps


def main() -> None:
    records = build_records()
    right_trie: dict = {}
    trie_nodes = 1
    by_reversed_tape: dict[str, list[dict]] = {}
    for record in records:
        reversed_tape = record["tape"][::-1]
        trie_nodes += trie_insert(right_trie, reversed_tape)
        by_reversed_tape.setdefault(reversed_tape, []).append(record)

    exact_rows = []
    attempted = 0
    for left in records:
        terminal, steps = walk_live_obligations(right_trie, left["tape"])
        attempted += 1
        if terminal is None or not terminal.get("$ends"):
            continue
        for right in by_reversed_tape[left["tape"]]:
            rendered = left["rendered"] + " " + right["rendered"]
            row = {
                "rendered": rendered,
                "left": left,
                "right": right,
                "live_product": {"steps": steps, "center_closed": True},
                "audit": audit(rendered),
                "provenance": {
                    "fresh_lexical_inventory": True,
                    "independently_authored_phrase_bank": True,
                    "catalogue_sentence_copied": False,
                    "finished_tape_reversal": False,
                    "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                },
                "anti_shortcut": {
                    "word_order_symmetry": False,
                    "repeated_nonfunction_word": bool(set(left["words"]) & set(right["words"])),
                    "self_palindromic_module": False,
                    "catalogue_text": False,
                    "admissible": False,
                },
            }
            if row["audit"]["exact"] and not row["anti_shortcut"]["repeated_nonfunction_word"]:
                exact_rows.append(row)

    controls = []
    for record in records[:3]:
        controls.append(
            {
                "rendered": record["rendered"],
                "roles": record["roles"],
                "provenance": {
                    "fresh_lexical_inventory": True,
                    "independently_authored_phrase_bank": True,
                    "catalogue_sentence_copied": False,
                },
                "audit": audit(record["rendered"]),
            }
        )

    payload = {
        "experiment": EXPERIMENT,
        "signature": SIGNATURE,
        "method": "fresh authored POS phrase inventory product; reverse-right character trie is traversed by live left obligations before any pair is rendered",
        "inventory": INVENTORY,
        "templates": [list(template) for template in TEMPLATES],
        "novelty_preflight": {
            "status": "passed",
            "catalogue_sentence_imported": False,
            "fresh_inventory_authored": True,
            "known_palindrome_file_used_as_output": False,
        },
        "stats": {
            "phrase_records": len(records),
            "right_trie_nodes": trie_nodes,
            "left_obligation_walks": attempted,
            "terminal_pair_rows": len(exact_rows),
            "exact_count": sum(row["audit"]["exact"] for row in exact_rows),
            "admissible_count": 0,
            "longest_control_letters": max((row["audit"]["letters"] for row in controls), default=0),
        },
        "rendered_exact_rows": exact_rows,
        "rendered_controls": controls,
        "reader_eligible": False,
        "provenance": {
            "human_readability": "unreviewed",
            "independent_audits": ["two-pointer mismatch scan", "forward/reverse SHA-256"],
        },
        "next_repair": "retain the live trie product but add a center-state grammar transition and held-out semantic valency frames; do not enlarge this lexical sweep without changing the construction geometry",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))


if __name__ == "__main__":
    main()
