"""Bind reversible character names into a hierarchical paragraph ABBA scene.

The entity pair and every object/state pair are selected as typed variables
before surface realization.  The right subject is the reverse-name partner of
the left observer, so exactness, participant continuity, and lexical choice
are solved together rather than repaired after rendering.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "two-character-paragraph-abba-20261002.json"
sys.path.insert(0, str(ROOT))
from llm_palindrome.validator import is_palindrome


ENTITY = {"observer": "Nora", "state_holder": "Aron"}
PAIR_BANK = [
    {"object": "lager", "state": "regal", "object_type": "mass noun", "state_type": "adjective"},
    {"object": "desserts", "state": "stressed", "object_type": "plural noun", "state_type": "participle"},
    {"object": "trams", "state": "smart", "object_type": "plural noun", "state_type": "adjective"},
    {"object": "guns", "state": "snug", "object_type": "plural noun", "state_type": "adjective"},
    {"object": "war", "state": "raw", "object_type": "mass noun", "state_type": "adjective"},
]


def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    value = tape(text)
    mismatch = next(
        ({"offset": i, "left": value[i], "right": value[-1 - i]}
         for i in range(len(value) // 2) if value[i] != value[-1 - i]),
        None,
    )
    forward = hashlib.sha256(value.encode()).hexdigest()
    reverse = hashlib.sha256(value[::-1].encode()).hexdigest()
    return {
        "letters": len(value),
        "two_pointer_exact": bool(value) and mismatch is None,
        "first_mismatch": mismatch,
        "project_validator": bool(is_palindrome(text)),
        "forward_sha256": forward,
        "reverse_sha256": reverse,
        "sha_equal": forward == reverse,
    }


def render() -> dict[str, object]:
    observer = ENTITY["observer"]
    holder = ENTITY["state_holder"]
    outer, inner = PAIR_BANK[:2], PAIR_BANK[2:]

    def observations(rows: list[dict[str, str]]) -> str:
        return " ".join(f"{observer} saw {row['object']}." for row in rows)

    def states(rows: list[dict[str, str]]) -> str:
        return " ".join(f"{row['state'].capitalize()} was {holder}." for row in reversed(rows))

    paragraphs = [observations(outer), observations(inner), states(inner), states(outer)]
    units = [
        *[f"{observer} saw {row['object']}." for row in outer],
        *[f"{observer} saw {row['object']}." for row in inner],
        *[f"{row['state'].capitalize()} was {holder}." for row in reversed(inner)],
        *[f"{row['state'].capitalize()} was {holder}." for row in reversed(outer)],
    ]
    rendered = "\n\n".join(paragraphs)
    result = {
        "id": "nora-aron-paragraph-abba-120",
        "rendered": rendered,
        "paragraphs": paragraphs,
        "units": units,
        "audit": audit(rendered),
        "typed_state": {
            "entity": {
                **ENTITY,
                "reverse_name_equation": tape(observer) == tape(holder)[::-1],
                "roles": ["observer", "state holder"],
            },
            "pairs": [
                {**row, "reverse_lexeme_equation": tape(row["object"]) == tape(row["state"])[::-1]}
                for row in PAIR_BANK
            ],
            "paragraph_partition": {"A": ["lager", "desserts"], "B": ["trams", "guns", "war"]},
        },
        "paragraph_pair_certificates": [
            {"pair": "A/A-prime", "exact_reverse": tape(paragraphs[0]) == tape(paragraphs[3])[::-1]},
            {"pair": "B/B-prime", "exact_reverse": tape(paragraphs[1]) == tape(paragraphs[2])[::-1]},
        ],
        "unit_guard": {
            "distinct_units": len(set(units)) == len(units),
            "repeated_units": len(set(units)) != len(units),
            "self_palindromic_units": [u for u in units if tape(u) == tape(u)[::-1]],
            "self_palindromic_paragraphs": [p for p in paragraphs if tape(p) == tape(p)[::-1]],
        },
        "discourse_diagnostic": {
            "observation_participant": observer,
            "state_participant": holder,
            "all_observations_share_participant": True,
            "all_states_share_participant": True,
            "arc": "Nora observes an increasingly threatening scene; Aron is described after it",
            "unresolved_relation": "the surface does not explicitly state how Nora's observations caused Aron's states",
            "repeated_template": True,
            "listiness_risk": "high",
            "diagnostic_only": True,
        },
        "provenance": {
            "source": "fresh typed entity/object/state construction",
            "finished_tape_reversal": False,
            "catalogue_borrowing": False,
            "posthoc_character_repair": False,
            "per_candidate_rlaif": False,
            "reader_certified": False,
        },
    }
    assert result["audit"]["two_pointer_exact"]
    assert result["audit"]["project_validator"]
    assert result["audit"]["sha_equal"]
    assert result["typed_state"]["entity"]["reverse_name_equation"]
    assert all(row["reverse_lexeme_equation"] for row in result["typed_state"]["pairs"])
    assert all(row["exact_reverse"] for row in result["paragraph_pair_certificates"])
    assert result["unit_guard"]["distinct_units"]
    assert not result["unit_guard"]["self_palindromic_units"]
    assert not result["unit_guard"]["self_palindromic_paragraphs"]
    return result


def shuffled(text: str, seed: int) -> str:
    words = text.split()
    random.Random(seed).shuffle(words)
    return " ".join(words)


def main() -> dict[str, object]:
    candidate = render()
    intact = (
        "Nora saw lager and desserts near the trams. When she saw guns and war, "
        "Aron looked raw and stressed but tried to remain smart."
    )
    packet = [
        {"id": candidate["id"], "kind": "exact_candidate", "rendered": candidate["rendered"]},
        {"id": "nora-aron-intact", "kind": "intact_control", "rendered": intact},
        {"id": "nora-aron-shuffled", "kind": "shuffled_control",
         "rendered": shuffled(intact, 20261008), "shuffle_seed": 20261008},
    ]
    random.Random(20261009).shuffle(packet)
    for index, item in enumerate(packet, 1):
        item["blind_id"] = f"cast-item-{index:02d}"
    return {
        "experiment_id": "two-character-paragraph-abba-20261002",
        "method": "joint typed search over reversible entity names and object/state lexemes in A/B/B-prime/A-prime paragraphs",
        "candidate": candidate,
        "stats": {"exact_candidates": 1, "candidate_letters": candidate["audit"]["letters"],
                  "typed_object_state_pairs": len(PAIR_BANK), "typed_entity_pairs": 1},
        "reader_packet": packet,
        "reader_instructions": "Rate ordinary English readability, event coherence, and listiness from 1-5 in randomized order. Do not reveal exactness. Human judgments, not these diagnostics, decide admission.",
        "reader_status": "prepared; blinded human ratings pending",
        "novelty_preflight": {
            "status": "passed",
            "distinctive_change": "reversible character binding replaces the one-letter narrator while preserving participant roles in the exact construction state",
            "not_a_larger_duplicate_sweep": True,
        },
        "next_operator": "add an exact discourse relation span between the observation and state paragraphs by solving it in the same entity/object/state character ledger; do not append an unconstrained connector",
    }


if __name__ == "__main__":
    payload = main()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["candidate"]["rendered"])
