"""Generate exact four-paragraph ABBA scenes from typed object/state pairs.

The generator chooses object/predicate pairs before rendering.  Each left
observation and right state are ordinary complete clauses whose normalized
tapes are reverses.  Pairs are partitioned into two nonempty paragraph spans,
so the whole surface is A / B / B' / A' without making any paragraph or
sentence self-palindromic.  Programmatic diagnostics expose repetition and
listiness; only blinded readers may certify readability.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "hierarchical-paragraph-abba-20261002.json"
sys.path.insert(0, str(ROOT))
from llm_palindrome.validator import is_palindrome
from llm_palindrome.admission import mechanical_admission_checks


PAIR_BANK = {
    "lager": {"state": "regal", "object_type": "mass noun", "state_type": "adjective"},
    "desserts": {"state": "stressed", "object_type": "plural noun", "state_type": "participle"},
    "trams": {"state": "smart", "object_type": "plural noun", "state_type": "adjective"},
    "guns": {"state": "snug", "object_type": "plural noun", "state_type": "adjective"},
    "war": {"state": "raw", "object_type": "mass noun", "state_type": "adjective"},
    "gums": {"state": "smug", "object_type": "plural noun", "state_type": "adjective"},
}


def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    value = tape(text)
    i, j = 0, len(value) - 1
    while i < j and value[i] == value[j]:
        i += 1
        j -= 1
    forward = hashlib.sha256(value.encode()).hexdigest()
    reverse = hashlib.sha256(value[::-1].encode()).hexdigest()
    return {
        "letters": len(value),
        "two_pointer_exact": bool(value) and i >= j,
        "first_mismatch": None if i >= j else {"offset": i, "left": value[i], "right": value[j]},
        "project_validator": bool(is_palindrome(text)),
        "forward_sha256": forward,
        "reverse_sha256": reverse,
        "sha_equal": forward == reverse,
    }


def clauses(objects: list[str]) -> tuple[list[str], list[str]]:
    left = [f"I saw {obj}." for obj in objects]
    right = [f"{PAIR_BANK[obj]['state'].capitalize()} was I." for obj in reversed(objects)]
    return left, right


def build(candidate_id: str, outer: list[str], inner: list[str], role: str) -> dict[str, object]:
    a, a_prime = clauses(outer)
    b, b_prime = clauses(inner)
    paragraphs = [" ".join(a), " ".join(b), " ".join(b_prime), " ".join(a_prime)]
    units = [*a, *b, *b_prime, *a_prime]
    rendered = "\n\n".join(paragraphs)
    paragraph_pairs = [
        {"left": "A", "right": "A-prime", "exact_reverse": tape(paragraphs[0]) == tape(paragraphs[3])[::-1]},
        {"left": "B", "right": "B-prime", "exact_reverse": tape(paragraphs[1]) == tape(paragraphs[2])[::-1]},
    ]
    pair_rows = []
    for obj in [*outer, *inner]:
        left = f"I saw {obj}."
        right = f"{PAIR_BANK[obj]['state'].capitalize()} was I."
        pair_rows.append({"object": obj, **PAIR_BANK[obj], "left": left, "right": right,
                          "exact_reverse": tape(left) == tape(right)[::-1]})
    result = audit(rendered)
    row = {
        "id": candidate_id,
        "role": role,
        "rendered": rendered,
        "paragraphs": paragraphs,
        "units": units,
        "topology": "A / B / B-prime / A-prime",
        "paragraph_pair_certificates": paragraph_pairs,
        "typed_pair_certificates": pair_rows,
        "audit": result,
        "unit_guard": {
            "distinct_units": len(set(units)) == len(units),
            "repeated_units": len(set(units)) != len(units),
            "self_palindromic_units": [u for u in units if tape(u) == tape(u)[::-1]],
            "self_palindromic_paragraphs": [p for p in paragraphs if tape(p) == tape(p)[::-1]],
        },
        "discourse_diagnostic": {
            "shared_participant": "first-person narrator in every clause",
            "arc": "observations in A/B, state response in B-prime/A-prime",
            "repeated_template": True,
            "listiness_risk": "high",
            "diagnostic_only": True,
        },
        "provenance": {
            "source": "fresh typed object/state grammar",
            "finished_tape_reversal": False,
            "catalogue_borrowing": False,
            "posthoc_character_repair": False,
            "per_candidate_rlaif": False,
            "reader_certified": False,
            "word_order_symmetry": True,
            "repeated_phrase_scaffold": True,
        },
    }
    assert result["two_pointer_exact"] and result["project_validator"] and result["sha_equal"]
    assert all(pair["exact_reverse"] for pair in paragraph_pairs)
    assert all(pair["exact_reverse"] for pair in pair_rows)
    assert row["unit_guard"]["distinct_units"]
    assert not row["unit_guard"]["self_palindromic_units"]
    assert not row["unit_guard"]["self_palindromic_paragraphs"]
    checks = mechanical_admission_checks(rendered, min_letters=30, max_letters=2000)
    row["mechanical_admission"] = {
        "checks": checks,
        "admitted": all(checks.values()),
        "blocking_failures": sorted(key for key, value in checks.items() if not value),
    }
    assert not row["mechanical_admission"]["admitted"]
    assert not checks["not_word_order_symmetry"]
    assert not checks["no_repeated_nontrivial_unit"]
    return row


def shuffle_words(text: str, seed: int) -> str:
    words = text.split()
    random.Random(seed).shuffle(words)
    return " ".join(words)


def main() -> dict[str, object]:
    reader_candidate = build(
        "paragraph-abba-reader-90",
        outer=["lager", "desserts"],
        inner=["trams", "guns", "war"],
        role="exact paragraph-topology control; mechanically rejected before reader study",
    )
    length_stress = build(
        "paragraph-abba-length-106",
        outer=["lager", "desserts", "gums"],
        inner=["trams", "guns", "war"],
        role="exact length stress control; mechanically rejected before reader study",
    )
    intact = (
        "I saw lager and desserts near the tram stop. Then I saw guns and war. "
        "The scene left me raw and stressed, though I tried to remain smart."
    )
    packet = [
        {"id": reader_candidate["id"], "kind": "exact_candidate", "rendered": reader_candidate["rendered"]},
        {"id": length_stress["id"], "kind": "exact_candidate", "rendered": length_stress["rendered"]},
        {"id": "intact-scene-control", "kind": "intact_control", "rendered": intact},
        {"id": "shuffled-scene-control", "kind": "shuffled_control",
         "rendered": shuffle_words(intact, 20261002), "shuffle_seed": 20261002},
    ]
    random.Random(20261007).shuffle(packet)
    for index, item in enumerate(packet, 1):
        item["blind_id"] = f"paragraph-item-{index:02d}"
    return {
        "experiment_id": "hierarchical-paragraph-abba-20261002",
        "method": "typed object/state generation with exact hierarchical A/B/B-prime/A-prime paragraph seams",
        "scaling_rule": "each accepted non-self object/state reverse pair adds one observation and one state clause; exactness is invariant under adding a fresh pair to A or B",
        "scaling_limit": "linear composition is currently bounded by the small reader-plausible pair bank; phrase mining must expand the bank without lowering grammar quality",
        "candidates": [reader_candidate, length_stress],
        "stats": {
            "typed_pairs_available": len(PAIR_BANK),
            "exact_topology_controls": 2,
            "reader_candidates": 0,
            "shorter_control_letters": reader_candidate["audit"]["letters"],
            "longest_letters": length_stress["audit"]["letters"],
        },
        "diagnostic_packet": packet,
        "reader_packet": [],
        "reader_instructions": "In randomized order, rate ordinary English readability, coherence, and listiness from 1-5. Do not show exactness labels. Programmatic diagnostics are not readability evidence.",
        "reader_status": "closed: both exact rows fail central anti-shortcut admission for boundary-aligned word symmetry and repeated phrase scaffolds",
        "novelty_preflight": {
            "status": "passed",
            "distinctive_change": "paragraph boundaries are first-class mirrored spans, not punctuation pasted onto a one-sentence tape",
            "not_a_larger_duplicate_sweep": True,
        },
        "next_operator": "replace the repeated saw/was units with varied grammatical frames whose exact character seams cross word boundaries; do not expand this mirrored-pair family",
    }


if __name__ == "__main__":
    payload = main()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    for row in payload["candidates"]:
        print("\n" + row["id"] + "\n" + row["rendered"])
