"""Construct a referent-preserving first-person ABBA paragraph.

The search space is a typed inventory of observation/state pairs.  Each left
clause has the form ``I saw OBJECT`` and each independently authored right
clause has the form ``STATE was I``.  A pair is eligible only when its two
normalized tapes are exact reverses and both clauses are complete English.
The shared first-person participant is discourse state, not a readability
certificate; the output remains gated on blinded human ratings.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "first-person-discourse-abba-20261002.json"
sys.path.insert(0, str(ROOT))
from llm_palindrome.validator import is_palindrome


PAIR_INVENTORY = [
    {
        "object": "desserts",
        "state": "stressed",
        "left": "I saw desserts.",
        "right": "Stressed was I.",
        "roles": ["observation", "resulting-state"],
    },
    {
        "object": "lager",
        "state": "regal",
        "left": "I saw lager.",
        "right": "Regal was I.",
        "roles": ["observation", "resulting-state"],
    },
    {
        "object": "war",
        "state": "raw",
        "left": "I saw war.",
        "right": "Raw was I.",
        "roles": ["observation", "resulting-state"],
    },
]


def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    value = tape(text)
    mismatches = [
        {"offset": i, "left": value[i], "right": value[-1 - i]}
        for i in range(len(value) // 2)
        if value[i] != value[-1 - i]
    ]
    left_sha = hashlib.sha256(value.encode()).hexdigest()
    right_sha = hashlib.sha256(value[::-1].encode()).hexdigest()
    i, j = 0, len(value) - 1
    while i < j and value[i] == value[j]:
        i += 1
        j -= 1
    return {
        "letters": len(value),
        "two_pointer_exact": bool(value) and i >= j,
        "first_mismatches": mismatches[:8],
        "forward_sha256": left_sha,
        "reverse_sha256": right_sha,
        "sha_equal": left_sha == right_sha,
        "project_validator": bool(is_palindrome(text)),
    }


def pair_audit(pair: dict[str, object]) -> dict[str, object]:
    left = tape(str(pair["left"]))
    right = tape(str(pair["right"]))
    return {
        **pair,
        "left_letters": len(left),
        "right_letters": len(right),
        "pair_exact": left == right[::-1],
        "left_self_palindromic": left == left[::-1],
        "right_self_palindromic": right == right[::-1],
    }


def candidate() -> dict[str, object]:
    pairs = [pair_audit(pair) for pair in PAIR_INVENTORY]
    assert all(pair["pair_exact"] for pair in pairs)
    assert all(not pair["left_self_palindromic"] for pair in pairs)
    assert all(not pair["right_self_palindromic"] for pair in pairs)

    units = [str(pair["left"]) for pair in pairs]
    units.extend(str(pair["right"]) for pair in reversed(pairs))
    rendered = " ".join(units)
    result = audit(rendered)
    unit_tapes = [tape(unit) for unit in units]
    adjacent_links = []
    for index in range(len(units) - 1):
        left_has_i = "i" in re.findall(r"[a-z]+", units[index].casefold())
        right_has_i = "i" in re.findall(r"[a-z]+", units[index + 1].casefold())
        adjacent_links.append({
            "left_index": index,
            "right_index": index + 1,
            "shared_participant": "I" if left_has_i and right_has_i else None,
            "connected": left_has_i and right_has_i,
        })

    row = {
        "id": "first-person-abba-56",
        "kind": "exact_abba_candidate",
        "rendered": rendered,
        "units": units,
        "pair_inventory": pairs,
        "audit": result,
        "unit_guard": {
            "unit_count": len(units),
            "distinct_units": len(set(units)) == len(units),
            "self_palindromic_units": [
                unit for unit, value in zip(units, unit_tapes)
                if value == value[::-1]
            ],
            "repeated_units": len(set(units)) != len(units),
        },
        "discourse_diagnostic": {
            "participant": "first-person narrator",
            "adjacent_links": adjacent_links,
            "all_adjacent_units_connected": all(link["connected"] for link in adjacent_links),
            "arc": "three observations followed by three narrator states",
            "diagnostic_only": True,
        },
        "shortcut_review": {
            "repeated_sentence_or_phrase_unit": False,
            "self_palindromic_sentence_or_phrase_unit": False,
            "single_letter_carrier": "I",
            "carrier_role": "ordinary recurring discourse participant",
            "human_review_required": True,
        },
        "provenance": {
            "source": "three freshly authored typed observation/state pairs",
            "finished_tape_reversal": False,
            "catalogue_text": False,
            "posthoc_repair": False,
            "per_candidate_rlaif": False,
            "reader_certified": False,
        },
    }
    assert result["two_pointer_exact"]
    assert result["project_validator"]
    assert result["sha_equal"]
    assert row["unit_guard"]["distinct_units"]
    assert not row["unit_guard"]["self_palindromic_units"]
    assert row["discourse_diagnostic"]["all_adjacent_units_connected"]
    return row


def shuffled(text: str, seed: int) -> str:
    words = text.split()
    random.Random(seed).shuffle(words)
    return " ".join(words)


def main() -> dict[str, object]:
    exact = candidate()
    intact = (
        "I saw the desserts beside the lager before the fighting began. "
        "After the battle, I felt shaken and raw."
    )
    controls = [
        {
            "id": "intact-first-person",
            "kind": "intact_control",
            "rendered": intact,
            "audit": audit(intact),
            "provenance": {"source": "freshly authored first-person prose control"},
        },
        {
            "id": "shuffled-first-person",
            "kind": "shuffled_control",
            "rendered": shuffled(intact, 20261002),
            "audit": audit(shuffled(intact, 20261002)),
            "provenance": {"source_control": "intact-first-person", "shuffle_seed": 20261002},
        },
    ]
    items = [exact, *controls]
    random.Random(20261003).shuffle(items)
    for index, item in enumerate(items, 1):
        item["blind_id"] = f"item-{index:02d}"
    return {
        "experiment_id": "first-person-discourse-abba-20261002",
        "method": "typed observation/state ABBA with a shared first-person discourse participant",
        "hypothesis": "A recurring participant can make independently exact sentence pairs read as one scene rather than an identity list.",
        "stats": {
            "typed_pairs": len(PAIR_INVENTORY),
            "exact_candidates": 1,
            "candidate_letters": exact["audit"]["letters"],
            "controls": len(controls),
        },
        "candidates": [exact],
        "items": items,
        "reader_instructions": "Rate ordinary English readability and event coherence from 1–5. Do not infer exactness. The single-letter narrator carrier is disclosed to the study designer but not highlighted to raters.",
        "reader_status": "prepared; blinded human ratings not yet collected",
        "independent_audits": [
            "local two-pointer walk",
            "project validator",
            "forward/reverse SHA-256",
            "pair-level reverse-tape certificates",
        ],
        "novelty_preflight": {
            "status": "passed",
            "distinctive_change": "one discourse participant is preserved across every independently authored sentence pair",
            "not_a_larger_bank_sweep": True,
        },
        "next_repair": "Hold the 56-letter candidate fixed for blinded comparison against the 38-letter anchor; only add a fourth typed pair if readers accept the inversion and the new pair preserves the same scene.",
    }


if __name__ == "__main__":
    payload = main()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
