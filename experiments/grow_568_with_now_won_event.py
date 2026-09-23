#!/usr/bin/env python3
"""Grow the verified 568-letter tape with a live now/won event pair.

This is a lineage-specific construction: insert ``now`` before the existing
Nora-delivers-maps clause, and its reverse ``won`` after the mirrored
Spam-is-reviled/Aron clause.  Grammar and character ownership are recorded in
the run artifact; this script makes no claim of human readability.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT_REL = "runs/incumbent-560-outer-causal-scene-20261002.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
OUTPUT_REL = "runs/incumbent-568-now-won-event-growth-20260923.json"
LEFT_INSERTION = "now"
RIGHT_INSERTION = "won"


def letters(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.lower()))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def main() -> None:
    parent_path = ROOT / PARENT_REL
    parent_record = json.loads(parent_path.read_text())
    parent_text = parent_record["rows"][0]["rendered"]
    parent_tape = letters(parent_text)

    assert len(parent_tape) == 568
    assert sha256(parent_tape) == PARENT_SHA256
    assert parent_tape == parent_tape[::-1]

    left_source = "Nora delivers maps."
    right_source = "Spam's reviled, Aron. Aidan's drawer"
    left_replacement = "Now, Nora delivers maps."
    right_replacement = "Spam's reviled; Aron won. Aidan's drawer"
    assert parent_text.count(left_source) == 1
    assert parent_text.count(right_source) == 1

    child_text = parent_text.replace(left_source, left_replacement, 1)
    child_text = child_text.replace(right_source, right_replacement, 1)
    child_tape = letters(child_text)

    left_cursor = 48
    right_cursor = 520
    mirrored_clause_length = 16
    assert parent_tape[left_cursor : left_cursor + mirrored_clause_length] == "noradeliversmaps"
    assert parent_tape[right_cursor - mirrored_clause_length : right_cursor] == "spamsreviledaron"
    assert parent_tape[left_cursor : left_cursor + mirrored_clause_length][::-1] == parent_tape[
        right_cursor - mirrored_clause_length : right_cursor
    ]
    assert LEFT_INSERTION == RIGHT_INSERTION[::-1]

    # Independently pin the earlier failed cursor report to this exact parent.
    assert parent_tape[167:170] == "saw"
    assert parent_tape[167:169] == "sa" and parent_tape[169:170] == "w"
    assert parent_tape[398:401] == "was"
    assert parent_tape[398:399] == "w" and parent_tape[399:401] == "as"
    failed_left = "".join(re.findall(r"[a-z]", "Mara saw that Leon sa".lower()))
    failed_right = "".join(re.findall(r"[a-z]", "As Noel, that was Aram.".lower()))
    failed_expected_right = failed_left[::-1]
    failed_mismatches = [
        {"offset": i, "expected": a, "observed": b}
        for i, (a, b) in enumerate(zip(failed_expected_right, failed_right))
        if a != b
    ]
    assert len(failed_left) == len(failed_right) == 17
    assert failed_expected_right == "asnoeltahtwasaram"
    assert failed_right == "asnoelthatwasaram"
    assert failed_mismatches == [
        {"offset": 7, "expected": "a", "observed": "h"},
        {"offset": 8, "expected": "h", "observed": "a"},
    ]

    expected_tape = (
        parent_tape[:left_cursor]
        + LEFT_INSERTION
        + parent_tape[left_cursor:right_cursor]
        + RIGHT_INSERTION
        + parent_tape[right_cursor:]
    )
    assert child_tape == expected_tape
    assert len(child_tape) == len(parent_tape) + len(LEFT_INSERTION) + len(RIGHT_INSERTION)

    # Independent exactness checks: normalized-string equality and a direct
    # mirrored-character walk, intentionally using separate implementations.
    regex_exact = child_tape == child_tape[::-1]
    mirrored_mismatches = [
        i for i in range(len(child_tape) // 2) if child_tape[i] != child_tape[-1 - i]
    ]
    assert regex_exact
    assert not mirrored_mismatches

    failed_seam = {
        "status": "no_candidate_semantic_attachment_obstruction",
        "parent_letter_cursors": {"left": 169, "right": 399},
        "parent_local_tape": {
            "left_token": "saw",
            "left_cut": "sa|w",
            "left_residual": "w",
            "right_token": "was",
            "right_cut": "w|as",
            "right_residual": "as",
        },
        "bounded_completion_reported_by": "Luna agent fresh_composition",
        "reported_local_equation": {
            "left": "Mara saw that Leon sa",
            "right": "As Noel, that was Aram.",
            "normalized_left": "marasawthatleonsa",
            "reverse_of_normalized_left": "asnoeltahtwasaram",
            "normalized_right": failed_right,
            "equation_exact": False,
            "first_character_mismatches": failed_mismatches,
        },
        "obstruction": (
            "The probe report's claimed equation was not exact: the reverse requires 'taht' "
            "where the proposed English clause has 'that' (two swapped characters). "
            "It also did not close a complete clause pair: "
            "Leon was simultaneously assigned an embedded-subject and inherited-continuation role, "
            "and the right-side 'As Noel, that was Aram' had no licensed predicate attachment. "
            "No full child was emitted."
        ),
        "next_action": "switch to a different 568 seam and an event-frame operator",
    }

    reward_drawer_probe = {
        "status": "bounded_lexical_infix_no_nonempty_pair",
        "parent_letter_cursors": {"left": 37, "right": 531},
        "open_tokens": {
            "left": {"surface": "rewards", "cut": "r|ewards", "residual_after_cut": "ewards", "suffix_owner": "third-person-singular verb inflection"},
            "mirrored_right": {"surface": "drawer", "cut": "drawe|r", "residual_after_cut": "r", "adjacent_matching_s_owner": "Aidan's possessive"},
        },
        "bounded_operator": "insert a nonempty infix X after left 'r' and reverse(X) after mirrored 'drawe', requiring both resulting strings to be intact English words",
        "lexicon": {
            "host": "hst-bench",
            "path": "/home/eric/words_alpha.txt",
            "sha256": "3ed0c94610d8bcf7c11bbb49c56aa49c7234d32b66824df91f554169e572da48",
            "word_count": 370105,
            "left_forms_with_nonempty_infix": ["rerewards", "romewards"],
            "paired_nonempty_infixes": [],
            "empty_infix_control": ["rewards", "drawer"],
        },
        "cursor_obstruction": "The only lexical closure found was the incumbent's unchanged rewards/drawer pair. For the two nonempty left forms, the mirrored drawe+reverse(X)+r form was absent; the boundary's s is already split across verb inflection and possessive ownership.",
        "pivot": "abandon this word-infix operator at [37,531]; seek a different event/grammar seam rather than enlarging this lexical family",
    }

    artifact = {
        "experiment_id": "incumbent-568-now-won-event-growth-20260923",
        "method": (
            "At the aligned clause seam, add a present-time adjunct 'Now' to the "
            "Nora-delivers-maps frame and pair its exact reverse 'won' with the "
            "independent Aron event in the mirrored clause. Track the two insertion "
            "cursors and the subject/tense/valency owner for each emitted word."
        ),
        "working_status": "audit_only_novelty_collision_not_promoted",
        "parent": {
            "artifact": PARENT_REL,
            "normalized_letter_length": len(parent_tape),
            "normalized_letter_sha256": sha256(parent_tape),
        },
        "prior_bounded_attempt": failed_seam,
        "distinct_seam_attempt": reward_drawer_probe,
        "construction": {
            "source_cursors": {"left_before_phrase": left_cursor, "right_after_mirror": right_cursor},
            "left_phrase_before": "Nora delivers maps",
            "right_mirror_before": "Spam's reviled, Aron",
            "left_emission": {
                "surface": "Now",
                "letters": LEFT_INSERTION,
                "owner": "temporal adjunct of singular-subject present-tense Nora delivers maps",
            },
            "right_emission": {
                "surface": "won",
                "letters": RIGHT_INSERTION,
                "owner": "past-tense intransitive predicate of singular subject Aron",
            },
            "paired_residual": {
                "before": ["now", "won"],
                "character_pairs": [["n", "n"], ["o", "o"], ["w", "w"]],
                "closure": "now == reverse(won)",
                "unresolved_lexical_residuals": [],
            },
            "surface_replacements": [
                {"before": left_source, "after": left_replacement},
                {"before": right_source, "after": right_replacement},
            ],
            "novelty_preflight": {
                "exact_surface_pair_found_in_prior_runs": False,
                "searches": [
                    "Now, Nora delivers maps",
                    "Aron won",
                    "spamsreviledaronwon",
                ],
                "scope_note": "The exact inserted surface pair was absent, but the post-generation ledger audit found that this complete-sentence seam had already been used; this child is retained only as a replay audit.",
            },
        },
        "post_generation_novelty_audit": {
            "collision": True,
            "prior_experiment_id": "incumbent-672-discourse-linked-reverse-chain-20260922",
            "prior_run_artifact": "runs/incumbent-672-discourse-linked-reverse-chain-20260922.json",
            "prior_same_seam": [48, 520],
            "prior_child_length": 672,
            "decision": "do not promote the 574-letter child or describe the seam as new; preserve this as an audit replay and pivot",
        },
        "candidate": {
            "rendered": child_text,
            "normalized_letter_length": len(child_tape),
            "normalized_letter_sha256": sha256(child_tape),
            "growth_over_parent_letters": len(child_tape) - len(parent_tape),
            "independent_exact_checks": {
                "regex_normalize_and_reverse": regex_exact,
                "direct_mirrored_character_walk": not mirrored_mismatches,
                "mismatch_count": len(mirrored_mismatches),
                "source_plus_two_owned_insertions": child_tape == expected_tape,
            },
            "shortcut_and_readability_status": {
                "borrowed_catalogue_text": False,
                "word_order_only_symmetry": False,
                "repeated_or_self_palindromic_units": "inherited parent repetition remains repair debt",
                "human_readability_certified": False,
                "known_local_readability": "The added clauses are grammatical in isolation; the full inherited text remains rough and is not reader-certified.",
            },
        },
    }
    output_path = ROOT / OUTPUT_REL
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"artifact": OUTPUT_REL, "length": len(child_tape), "sha256": sha256(child_tape), "rendered": child_text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
