"""Repair one repeated shell via a reversed-predicate valency switch."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.validator import is_palindrome, normalize


PARENT = ROOT / "runs" / "incumbent-568-asymmetric-residual-solos-20260922.json"
OUT = ROOT / "runs" / "incumbent-586-repeated-shell-valency-repair-20260922.json"
PARENT_ID = "asymmetric-noel-leon-solos-586"
PARENT_SHA256 = "d7cf3db4224acc9042f7f4badd48dfbd69c56f0f554ed9d52f053ef5f197ab41"
EXPECTED_SHA256 = "142f34541db73cb8a7b9c7fa96cb34abec9fd4bea5d3003b6482d476f6e7569b"
OLD_LEFT = "A tub? He maps Aron."
NEW_LEFT = "A tub? He raps, Aron."
OLD_RIGHT = "Nora, spam."
NEW_RIGHT = "Nora, spar!"


def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def two_pointer_exact(value: str) -> bool:
    left, right = 0, len(value) - 1
    while left < right:
        if value[left] != value[right]:
            return False
        left += 1
        right -= 1
    return bool(value)


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent_row = parent_payload["candidate"]
    assert parent_row["id"] == PARENT_ID
    parent_rendered = str(parent_row["rendered"])
    parent_tape = tape(parent_rendered)
    assert len(parent_tape) == 586 and parent_tape == parent_tape[::-1]
    assert hashlib.sha256(parent_tape.encode("ascii")).hexdigest() == PARENT_SHA256
    assert parent_rendered.count(OLD_LEFT) == 1
    assert parent_rendered.count(OLD_RIGHT) == 1

    rendered = parent_rendered.replace(OLD_LEFT, NEW_LEFT, 1)
    rendered = rendered.replace(OLD_RIGHT, NEW_RIGHT, 1)
    candidate_tape = tape(rendered)
    result_sha = hashlib.sha256(candidate_tape.encode("ascii")).hexdigest()
    reverse_sha = hashlib.sha256(candidate_tape[::-1].encode("ascii")).hexdigest()
    exact = two_pointer_exact(candidate_tape)
    assert len(candidate_tape) == 586
    assert exact and result_sha == reverse_sha == EXPECTED_SHA256
    assert is_palindrome(rendered) and normalize(rendered) == candidate_tape
    assert parent_tape.count("atubhemaps") == 2
    assert candidate_tape.count("atubhemaps") == 1
    assert tape("maps") == tape("spam")[::-1]
    assert tape("raps") == tape("spar")[::-1]

    return {
        "experiment_id": "incumbent-586-repeated-shell-valency-repair-20260922",
        "method": "paired lexical valency switch at one repeated 568-lineage shell",
        "status": "exact_same_length_repair_variant_with_inherited_prose_debt",
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "id": PARENT_ID,
            "letters": 586,
            "sha256": PARENT_SHA256,
            "568_lineage_parent": {
                "artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
                "id": "outer-causal-scene-568-working-incumbent",
                "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
            },
        },
        "candidate": {
            "id": "asymmetric-noel-leon-solos-586-one-shell-repaired",
            "rendered": rendered,
            "audit": {
                "letters": len(candidate_tape),
                "independent_two_pointer_exact": exact,
                "project_validator_exact": bool(is_palindrome(rendered)),
                "project_normalizer_agrees": normalize(rendered) == candidate_tape,
                "sha256_forward": result_sha,
                "sha256_reverse": reverse_sha,
                "sha_equal": result_sha == reverse_sha,
            },
            "length_delta": 0,
            "repetition_delta": {
                "normalized_shell": "atubhemaps",
                "before": parent_tape.count("atubhemaps"),
                "after": candidate_tape.count("atubhemaps"),
            },
            "local_lexical_equation": {
                "old_left": tape("maps"),
                "old_right": tape("spam"),
                "new_left": tape("raps"),
                "new_right": tape("spar"),
                "reverse_left_equals_right": tape("raps") == tape("spar")[::-1],
                "cross_clause_residual_persisted": False,
            },
            "repair": {
                "before_left": OLD_LEFT,
                "after_left": NEW_LEFT,
                "before_reflected_support": OLD_RIGHT,
                "after_reflected_support": NEW_RIGHT,
                "interpretation": "He raps is intransitive; Aron is a vocative. Nora, spar! is an imperative.",
            },
            "provenance": {
                "source_is_generated_parent": True,
                "borrowed_or_catalogue_text": False,
                "punctuation_changed_letter_tape": False,
                "finished_tape_reversal": False,
                "full_shell_replacement": False,
            },
            "repair_debt": {
                "one_repeated_A_tub_shell_remains": True,
                "inherited_proper_palindromic_spans": True,
                "rough_prose": True,
                "human_readability_certified": False,
                "effect": "preserve as a same-length repair branch, not a readability claim",
            },
        },
        "novelty_preflight": {
            "status": "passed_before_this_candidate",
            "checked_with_ignore_rules_disabled": ["runs/", "experiments/", "docs/", "data/"],
            "queries": ["He raps, Aron", "Nora, spar", "rapsaron", "noraspar"],
            "prior_exact_clause_collisions": 0,
            "prior_operator_overlap": "distinct from previous complete-clause shell lattices: only one verb and its reverse-word support are edited",
        },
        "stats": {
            "exact_children": 1,
            "same_length_repair_variants": 1,
            "letters": len(candidate_tape),
            "repeated_shell_instances_removed": 1,
            "backtracks": 0,
        },
        "next_construction": {
            "target": "the remaining A tub? He maps Nora shell",
            "operator": "model the query fragment and answer as one typed dialogue move with shared valency state, keeping a nonempty residual open across the question boundary",
            "avoid": ["whole-clause reverse-trie graft", "fixed mirrored-window replacement", "reused event-clause banks"],
            "reader_test": "after the remaining repeated shell is repaired, use blinded randomized reading against intact-prose and shuffled controls; no programmatic score can certify readability",
        },
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["candidate"]["rendered"])


if __name__ == "__main__":
    main()
