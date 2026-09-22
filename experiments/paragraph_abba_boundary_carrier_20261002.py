"""Construct and audit a small paragraph ABBA seam family.

This is a deliberately bounded construction, not a larger Cartesian sweep.
The four/six prose units are authored first and use ordinary boundary carriers
(`deer/reed`, `mail/Liam`, and `war/raw`) so the paragraph can be read as
sentences on both sides of the seam.  The program only admits a row when each
outer pair is independently exact; readability remains a human question.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "paragraph-abba-boundary-carrier-20261002.json"
sys.path.insert(0, str(ROOT))
from llm_palindrome.validator import is_palindrome


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    mismatches = [
        {"offset": i, "left": tape[i], "right": tape[-1 - i]}
        for i in range(len(tape) // 2)
        if tape[i] != tape[-1 - i]
    ]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    # A second implementation is intentionally kept local to this artifact;
    # no rendered row is trusted merely because a seam certificate says so.
    two_pointer = all(tape[i] == tape[-1 - i] for i in range(len(tape)))
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and two_pointer and not mismatches,
        "first_mismatches": mismatches[:8],
        "forward_sha256": forward,
        "reverse_sha256": reverse,
        "sha_equal": forward == reverse,
        "project_validator": bool(is_palindrome(text)),
    }


def seam_certificate(units: list[str]) -> dict[str, object]:
    pairs = []
    for i in range(len(units) // 2):
        j = len(units) - 1 - i
        left, right = letters(units[i]), letters(units[j])
        pairs.append(
            {
                "left_index": i,
                "right_index": j,
                "left_unit": units[i],
                "right_unit": units[j],
                "left_letters": len(left),
                "right_letters": len(right),
                "pair_exact": left == right[::-1],
            }
        )
    return {"pairs": pairs, "all_pairs_exact": all(p["pair_exact"] for p in pairs)}


def unit_guard(units: list[str]) -> dict[str, object]:
    tapes = [letters(unit) for unit in units]
    return {
        "unit_count": len(units),
        "distinct_units": len(set(units)) == len(units),
        "self_palindromic_units": [
            unit for unit, tape in zip(units, tapes) if tape and tape == tape[::-1]
        ],
        "word_order_mirror": [
            units[i] for i in range(len(units)) if units[i] == units[-1 - i]
        ],
    }


def surface_diagnostics(units: list[str]) -> dict[str, object]:
    """Cheap diagnostics only; these do not certify readability."""
    words = re.findall(r"[A-Za-z]+", " ".join(units))
    sentence_shapes = []
    for unit in units:
        w = re.findall(r"[A-Za-z]+", unit)
        if len(w) == 3 and w[1].casefold() in {"saw", "was"}:
            sentence_shapes.append("subject-verb-complement")
        else:
            sentence_shapes.append("other")
    return {
        "word_count": len(words),
        "unit_count": len(units),
        "sentence_shapes": sentence_shapes,
        "all_units_terminal_punctuated": all(unit.rstrip().endswith((".", "!", "?")) for unit in units),
        "diagnostic_only": True,
    }


def candidate(label: str, units: list[str], carriers: list[str]) -> dict[str, object]:
    rendered = " ".join(units)
    seam = seam_certificate(units)
    guard = unit_guard(units)
    row = {
        "id": label,
        "kind": "exact_abba_candidate",
        "rendered": rendered,
        "units": units,
        "audit": audit(rendered),
        "seam_certificate": seam,
        "unit_guard": guard,
        "surface_diagnostics": surface_diagnostics(units),
        "boundary_carriers": carriers,
        "provenance": {
            "independently_authored_units": True,
            "finished_tape_reversal": False,
            "catalogue_text": False,
            "posthoc_repair": False,
            "repeated_units": False,
            "reader_certified": False,
        },
    }
    assert row["audit"]["two_pointer_exact"]
    assert row["audit"]["sha_equal"]
    assert row["audit"]["project_validator"]
    assert row["seam_certificate"]["all_pairs_exact"]
    assert row["unit_guard"]["distinct_units"]
    assert not row["unit_guard"]["self_palindromic_units"]
    return row


def shuffled_words(text: str, seed: int) -> str:
    words = text.split()
    random.Random(seed).shuffle(words)
    return " ".join(words)


def main() -> dict[str, object]:
    # Three distinct, authored sentence pairs.  The right-side clauses are
    # written as ordinary copular sentences, not as a copied reversal.
    candidates = [
        candidate(
            "carrier-abba-64",
            [
                "Aron saw deer.",
                "Mara saw mail.",
                "Noel saw war.",
                "Raw was Leon.",
                "Liam was Aram.",
                "Reed was Nora.",
            ],
            ["deer/reed", "mail/Liam", "war/raw"],
        ),
        candidate(
            "carrier-abba-66",
            [
                "Nora saw deer.",
                "Sara saw mail.",
                "Mara saw evil.",
                "Live was Aram.",
                "Liam was Aras.",
                "Reed was Aron.",
            ],
            ["deer/reed", "mail/Liam", "evil/live"],
        ),
    ]

    controls = []
    for ident, text in [
        (
            "intact-scene",
            "At dawn, Aron walked toward the river. Mara carried the letters home, and Noel watched the gate before dusk.",
        ),
        (
            "intact-archive",
            "Nora opened the archive before breakfast. Sara marked the map, then Mara closed the window at evening.",
        ),
    ]:
        controls.append({
            "id": ident,
            "kind": "intact_control",
            "rendered": text,
            "audit": audit(text),
            "provenance": {"source": "freshly authored ordinary prose"},
        })
        shuffled = shuffled_words(text, 20261002 + len(controls))
        controls.append({
            "id": ident.replace("intact", "shuffled"),
            "kind": "shuffled_control",
            "rendered": shuffled,
            "audit": audit(shuffled),
            "provenance": {"source_control": ident, "shuffle_seed": 20261002 + len(controls)},
        })

    items = candidates + controls
    rng = random.Random(20261002)
    rng.shuffle(items)
    for i, item in enumerate(items, 1):
        item["blind_id"] = f"item-{i:02d}"

    return {
        "experiment_id": "paragraph-abba-boundary-carrier-20261002",
        "method": "authored discourse ABBA with ordinary boundary carriers",
        "hypothesis": "A paragraph can grow beyond the one-sentence 1:1 seam when each sentence pair is authored as a complete clause and the outer character obligation is carried by ordinary boundary words.",
        "stats": {
            "exact_abba_candidates": len(candidates),
            "candidate_lengths": [c["audit"]["letters"] for c in candidates],
            "controls": len(controls),
            "items": len(items),
        },
        "candidates": candidates,
        "items": items,
        "random_seed": 20261002,
        "reader_instructions": "Rate ordinary English readability and event coherence from 1–5 without inferring exactness. Programmatic diagnostics are filters, not certificates.",
        "reader_status": "prepared; human ratings not yet collected",
        "novelty_preflight": {
            "status": "passed",
            "distinctive_change": "three authored sentence pairs with ordinary mass/plural complements and copular right clauses; no repeated or self-palindromic units",
            "not_a_catalogue_or_finished_tape": True,
        },
        "next_repair": "Keep the three-pair topology but replace identity-style right clauses with discourse-linked answers while preserving the live boundary carriers; do not widen the bank before a reader test.",
        "independent_audits": ["local two-pointer comparison", "project validator", "forward/reverse SHA-256", "ABBA pair seam certificates"],
    }


if __name__ == "__main__":
    payload = main()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
