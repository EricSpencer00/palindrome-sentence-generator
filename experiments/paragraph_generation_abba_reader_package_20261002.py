"""Prepare a blinded reader packet for paragraph-level ABBA candidates.

This is an evaluation artifact, not another search sweep.  It makes the
paragraph hypothesis concrete: each candidate is an ordered sequence of
distinct, independently rendered prose units with a live ABBA seam
certificate.  The packet keeps exactness separate from readability and pairs
the candidates with intact and shuffled prose controls.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.validator import is_palindrome

OUT = ROOT / "runs" / "paragraph-generation-abba-reader-20261002.json"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    reverse = tape[::-1]
    mismatches = [
        {"offset": i, "left": tape[i], "right": tape[-1 - i]}
        for i in range(len(tape) // 2)
        if tape[i] != tape[-1 - i]
    ]
    forward_sha = hashlib.sha256(tape.encode()).hexdigest()
    reverse_sha = hashlib.sha256(reverse.encode()).hexdigest()
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and not mismatches,
        "first_mismatches": mismatches[:8],
        "forward_sha256": forward_sha,
        "reverse_sha256": reverse_sha,
        "sha_equal": forward_sha == reverse_sha,
        "project_validator": bool(is_palindrome(text)),
    }


def unit_guard(units: list[str]) -> dict[str, object]:
    tapes = [letters(unit) for unit in units]
    return {
        "unit_count": len(units),
        "distinct_units": len(set(units)) == len(units),
        "self_palindromic_units": [
            unit for unit, tape in zip(units, tapes) if tape and tape == tape[::-1]
        ],
        "word_order_mirror": [
            units[i]
            for i in range(len(units))
            if units[i] == units[-1 - i]
        ],
    }


def seam_certificate(units: list[str]) -> dict[str, object]:
    """Check the ABBA pairing without constructing a reversed surface."""
    pairs = []
    half = len(units) // 2
    for i in range(half):
        left, right = units[i], units[-1 - i]
        left_tape, right_tape = letters(left), letters(right)
        pairs.append({
            "left_index": i,
            "right_index": len(units) - 1 - i,
            "left_unit": left,
            "right_unit": right,
            "left_letters": len(left_tape),
            "right_letters": len(right_tape),
            "pair_exact": left_tape == right_tape[::-1],
        })
    return {"pairs": pairs, "all_pairs_exact": all(p["pair_exact"] for p in pairs)}


def load_candidate(path: str, rendered: str, units: list[str], label: str) -> dict[str, object]:
    text = " ".join(units)
    assert text == rendered, f"candidate surface changed for {label}"
    return {
        "id": label,
        "kind": "exact_abba_candidate",
        "rendered": text,
        "units": units,
        "audit": audit(text),
        "unit_guard": unit_guard(units),
        "seam_certificate": seam_certificate(units),
        "source_artifact": path,
        "provenance": {
            "independently_authored_units": True,
            "finished_tape_reversal": False,
            "catalogue_text": False,
            "repeated_units": False,
            "reader_certified": False,
        },
    }


def shuffled_words(text: str, seed: int) -> str:
    words = text.split()
    rng = random.Random(seed)
    rng.shuffle(words)
    return " ".join(words)


def main() -> dict[str, object]:
    # These are the two existing exact ABBA constructions.  This packet does
    # not promote either one: it is the first reader-facing comparison of the
    # topology, with exactness already independently rechecked above.
    candidates = [
        load_candidate(
            "runs/grammar-abba-paragraph-20260928.json",
            "Nora, I saw evil. Noel, I saw war. Raw was I, Leon. Live was I, Aron.",
            [
                "Nora, I saw evil.",
                "Noel, I saw war.",
                "Raw was I, Leon.",
                "Live was I, Aron.",
            ],
            "abba-46",
        ),
        load_candidate(
            "runs/vocative-abba-clause-search-20260927.json",
            "Nora, I saw evil. Noel, I saw war. Mara, I saw God. Dog was I, Aram. Raw was I, Leon. Live was I, Aron.",
            [
                "Nora, I saw evil.",
                "Noel, I saw war.",
                "Mara, I saw God.",
                "Dog was I, Aram.",
                "Raw was I, Leon.",
                "Live was I, Aron.",
            ],
            "abba-68",
        ),
    ]
    assert all(c["audit"]["two_pointer_exact"] for c in candidates)
    assert all(c["audit"]["project_validator"] for c in candidates)
    assert all(c["seam_certificate"]["all_pairs_exact"] for c in candidates)
    assert all(not c["unit_guard"]["self_palindromic_units"] for c in candidates)

    base_controls = [
        {
            "id": "intact-mountain",
            "kind": "intact_control",
            "rendered": (
                "At first light, the pilot checked the weather before leaving the mountain lodge. "
                "On the trail, a guide traded spare matches for a compass. "
                "Beyond the pass, clouds gathered above a silent ravine. "
                "At the river, a ranger swapped dry gloves for a lantern."
            ),
        },
        {
            "id": "intact-archive",
            "kind": "intact_control",
            "rendered": (
                "Mara opened the archive before dawn. She found a weathered map beside the window "
                "and marked the road that led toward the harbor. By evening, the team packed the charts and started home."
            ),
        },
    ]
    controls = []
    for control in base_controls:
        control["audit"] = audit(control["rendered"])
        control["provenance"] = {"source": "freshly authored ordinary prose control"}
        controls.append(control)
        seed = 20261002 + len(controls)
        shuffled = shuffled_words(control["rendered"], seed)
        controls.append({
            "id": control["id"].replace("intact", "shuffled"),
            "kind": "shuffled_control",
            "rendered": shuffled,
            "audit": audit(shuffled),
            "provenance": {"source_control": control["id"], "shuffle_seed": seed},
        })

    items = candidates + controls
    rng = random.Random(20261002)
    rng.shuffle(items)
    for index, item in enumerate(items):
        item["blind_id"] = f"item-{index + 1:02d}"

    return {
        "experiment_id": "paragraph-generation-abba-reader-20261002",
        "method": "reader packet for independently authored paragraph-level ABBA seams",
        "hypothesis": "ABBA generation topology can preserve a paragraph-like event progression only if each mirrored generation is independently authored and its live character seam closes.",
        "stats": {
            "exact_abba_candidates": len(candidates),
            "candidate_lengths": [c["audit"]["letters"] for c in candidates],
            "controls": len(controls),
            "items": len(items),
        },
        "candidates": candidates,
        "items": items,
        "random_seed": 20261002,
        "reader_instructions": (
            "Rate each item for ordinary English readability and event coherence, 1–5. "
            "Do not infer or rate exactness. The packet contains intact prose and word-shuffled controls; order is blinded."
        ),
        "reader_status": "prepared; human ratings not yet collected",
        "next_test": "Run the packet with blinded human readers, then author multi-sentence generation units at the first residual frontier rather than expanding the existing clause bank.",
        "independent_audits": ["local two-pointer comparison", "project validator", "forward/reverse SHA-256", "ABBA pair seam certificates"],
    }


if __name__ == "__main__":
    payload = main()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
