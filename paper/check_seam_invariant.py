"""Replay one saved asymmetric seam edit and audit its algebraic invariant.

This is a bounded check of a specific 568-to-630 construction, not a search
over the repository's historical candidate bank.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "paper" / "seam_invariant_results.json"
RUN_PATH = ROOT / "runs" / "luna6-god-dog-live-residual-growth-20260923.json"
PARENT_PATH = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
EXPECTED_PARENT_SHA = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
EXPECTED_CHILD_SHA = "017d5e73f11339204b3f343fa66e63b5c4674afd01c5ea823e36999879a2ac11"


def normalize(text: str) -> str:
    return "".join(char.lower() for char in text if char.isascii() and char.isalpha())


def outside_in(tape: str) -> dict[str, object]:
    left, right = 0, len(tape) - 1
    while left < right and tape[left] == tape[right]:
        left += 1
        right -= 1
    exact = left >= right
    return {
        "exact": exact,
        "letters": len(tape),
        "matched_outer_pairs": left,
        "first_mismatch": None if exact else {
            "left_offset": left,
            "right_offset": right,
            "left": tape[left],
            "right": tape[right],
        },
    }


def raw_after_letters(text: str, count: int) -> int:
    seen = 0
    for index, char in enumerate(text):
        if char.isascii() and char.isalpha():
            seen += 1
            if seen == count:
                return index + 1
    if count == 0:
        return 0
    raise ValueError(f"surface contains fewer than {count} letters")


def binary_strings(max_length: int) -> list[str]:
    return [
        "".join(chars)
        for length in range(max_length + 1)
        for chars in itertools.product("ab", repeat=length)
    ]


def exhaustive_algebra_audit() -> dict[str, object]:
    """Exhaust equal-length insertions; separately expose unequal-length scope.

    If |U|=|V|, the center is unchanged, so the equation Uq=q reverse(V) is
    necessary and sufficient. For unequal insertion lengths the palindrome's
    center can shift, so the equation remains sufficient but is not necessary.
    """
    outer_contexts = binary_strings(2)
    residuals = binary_strings(2)
    insertions = binary_strings(3)
    centers = [value for value in binary_strings(4) if value == value[::-1]]

    equivalence_cases = 0
    false_positive = 0
    false_negative = 0
    for outer, q, left_insert, center, right_insert in itertools.product(
        outer_contexts, residuals, insertions, centers, insertions
    ):
        if len(left_insert) != len(right_insert):
            continue
        predicted = left_insert + q == q + right_insert[::-1]
        whole = (
            outer + left_insert + q + center + right_insert + q[::-1]
            + outer[::-1]
        )
        observed = whole == whole[::-1]
        equivalence_cases += 1
        false_positive += int(predicted and not observed)
        false_negative += int(observed and not predicted)

    context_cases = 0
    context_errors = 0
    for outer, center in itertools.product(outer_contexts, centers):
        context_cases += 1
        wrapped = outer + center + outer[::-1]
        context_errors += int(wrapped != wrapped[::-1])

    unequal_counterexample = {
        "A": "",
        "q": "a",
        "C": "",
        "U": "",
        "V": "a",
    }
    counterexample_tape = (
        unequal_counterexample["A"] + unequal_counterexample["U"]
        + unequal_counterexample["q"] + unequal_counterexample["C"]
        + unequal_counterexample["V"] + unequal_counterexample["q"][::-1]
        + unequal_counterexample["A"][::-1]
    )
    counterexample_equation = (
        unequal_counterexample["U"] + unequal_counterexample["q"]
        == unequal_counterexample["q"] + unequal_counterexample["V"][::-1]
    )
    unequal_insertions = sum(
        len(left) != len(right)
        for left, right in itertools.product(insertions, insertions)
    )
    checked = equivalence_cases + context_cases
    return {
        "alphabet": "{a,b}",
        "domains": {
            "A_strings_length_0_to_2": len(outer_contexts),
            "q_strings_length_0_to_2": len(residuals),
            "C_palindromes_length_0_to_4": len(centers),
            "U_strings_length_0_to_3": len(insertions),
            "V_strings_length_0_to_3": len(insertions),
        },
        "full_unrestricted_cartesian_size": (
            len(outer_contexts) * len(residuals) * len(centers)
            * len(insertions) * len(insertions)
        ),
        "checks_performed": {
            "equal_length_U_V_full_tape_equivalence_cases_over_A_q_U_V_C": equivalence_cases,
            "outer_center_context_cases_over_A_C": context_cases,
            "total": checked,
        },
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "invalid_outer_center_contexts": context_errors,
        "unequal_length_insertions_per_q_C_A_context": unequal_insertions,
        "unequal_length_counterexample": {
            **unequal_counterexample,
            "full_tape": counterexample_tape,
            "full_tape_exact": counterexample_tape == counterexample_tape[::-1],
            "equation_holds": counterexample_equation,
            "shows_iff_fails_when_lengths_differ": (
                counterexample_tape == counterexample_tape[::-1]
                and not counterexample_equation
            ),
        },
        "all_checks_passed": (
            false_positive == false_negative == context_errors == 0
            and counterexample_tape == counterexample_tape[::-1]
            and not counterexample_equation
        ),
        "scope_note": (
            "The iff claim is checked only for equal-length U and V. The "
            "equation is sufficient at unequal lengths, but not necessary. "
            "This verifies algebra only, not Englishness, novelty, or readability."
        ),
    }


def grammar_inventory_census() -> dict[str, object]:
    """Count the finite 672-search clause and reverse-compatible-pair bank."""
    sys.path.insert(0, str(ROOT))
    from experiments.incumbent_672_discourse_linked_reverse_chain_20260922 import (
        ENTITIES,
        PREDICATES,
        iter_clauses,
    )

    clauses = list(iter_clauses())
    filtered = [
        clause for clause in clauses
        if clause.subject.surface != clause.object.surface
        and clause.tape != clause.tape[::-1]
    ]
    pair_count = sum(
        left.tape == right.tape[::-1]
        for left in filtered
        for right in filtered
    )
    self_subject_object = sum(
        clause.subject.surface == clause.object.surface for clause in clauses
    )
    self_palindromic = sum(clause.tape == clause.tape[::-1] for clause in clauses)
    return {
        "source": "experiments/incumbent_672_discourse_linked_reverse_chain_20260922.py",
        "entity_count": len(ENTITIES),
        "predicate_count": len(PREDICATES),
        "raw_clause_count": len(clauses),
        "clauses_with_subject_equal_object": self_subject_object,
        "self_palindromic_clauses": self_palindromic,
        "clauses_after_both_filters": len(filtered),
        "ordered_reverse_compatible_clause_pairs": pair_count,
        "scope_note": (
            "A per-clause inventory census only. It does not count connected "
            "four-clause chains, historical novelty gates, or readability."
        ),
    }


def edit_ablations(parent: str, left_raw: int, right_raw: int,
                   left_insert: str, right_insert: str) -> dict[str, object]:
    def splice(left: str, right: str) -> str:
        return (
            parent[:left_raw] + left + parent[left_raw:right_raw]
            + right + parent[right_raw + 1:]
        )

    full = splice(left_insert, right_insert)
    left_only = parent[:left_raw] + left_insert + parent[left_raw:]
    right_only = splice("", right_insert)
    one_char_removed = splice(left_insert[:-1], right_insert)
    punctuation_variant = "".join(
        char if char.isascii() and char.isalpha() or char.isspace() else ","
        for char in full
    )
    variants = {
        "full_paired_edit": full,
        "left_only": left_only,
        "right_only": right_only,
        "one_left_inserted_character_removed": one_char_removed,
        "letter_preserving_punctuation_change": punctuation_variant,
    }

    result: dict[str, object] = {}
    for name, rendered in variants.items():
        tape = normalize(rendered)
        audit = outside_in(tape)
        result[name] = {
            "letters": len(tape),
            "exact": audit["exact"],
            "first_mismatch": audit["first_mismatch"],
            "sha256": hashlib.sha256(tape.encode("ascii")).hexdigest(),
        }
    return result


def main() -> dict[str, object]:
    sys.path.insert(0, str(ROOT))
    from llm_palindrome.validator import is_palindrome
    from experiments.luna6_god_dog_live_residual_growth_20260923 import (
        LEFT_CUT,
        LEFT_INSERT,
        RIGHT_CUT,
        RIGHT_INSERT,
    )

    saved = json.loads(RUN_PATH.read_text())
    parent_payload = json.loads(PARENT_PATH.read_text())
    parent = next(
        row["rendered"] for row in parent_payload["rows"]
        if row["id"] == "outer-causal-scene-568-working-incumbent"
    )
    parent_tape = normalize(parent)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if len(parent_tape) != 568 or parent_sha != EXPECTED_PARENT_SHA:
        raise AssertionError("pinned 568 parent identity changed")
    if not outside_in(parent_tape)["exact"] or not is_palindrome(parent):
        raise AssertionError("pinned parent failed independent exact validation")

    left_raw = raw_after_letters(parent, LEFT_CUT)
    right_raw = raw_after_letters(parent, RIGHT_CUT)
    if (left_raw, right_raw) != (20, 758):
        raise AssertionError("source cursor/raw-cursor mapping changed")
    rendered = (
        parent[:left_raw] + LEFT_INSERT + parent[left_raw:right_raw]
        + RIGHT_INSERT + parent[right_raw + 1:]
    )
    if rendered != saved["rendered_full_text"]:
        raise AssertionError("replayed surface differs from saved 630-letter artifact")

    tape = normalize(rendered)
    a = parent_tape[:LEFT_CUT]
    q = parent_tape[LEFT_CUT:LEFT_CUT + 4]
    center = parent_tape[LEFT_CUT + 4:RIGHT_CUT]
    right_q = parent_tape[RIGHT_CUT:RIGHT_CUT + 4]
    suffix = parent_tape[RIGHT_CUT + 4:]
    u, v = normalize(LEFT_INSERT), normalize(RIGHT_INSERT)
    equation_holds = u + q == q + v[::-1]
    splice_closes = (
        a == suffix[::-1]
        and right_q == q[::-1]
        and center == center[::-1]
        and equation_holds
    )
    if (q, len(center), right_q) != ("nora", 528, "aron"):
        raise AssertionError("saved parent does not match requested partial-word seam")

    independent_audit = outside_in(tape)
    hash_forward = hashlib.sha256(tape.encode("ascii")).hexdigest()
    hash_reverse = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    project_audit = bool(is_palindrome(rendered))
    exact = independent_audit["exact"] and project_audit and hash_forward == hash_reverse
    if (len(tape) != 630 or hash_forward != EXPECTED_CHILD_SHA
            or not splice_closes or not exact):
        raise AssertionError("replayed splice failed its equation or full-tape audits")
    saved_seam = saved["live_seam"]
    if (saved_seam["left_emission"] != u
            or saved_seam["right_emission"] != v
            or saved_seam["left_residual"] != q
            or saved_seam["retained_middle_letters"] != len(center)):
        raise AssertionError("saved live-seam trace differs from replayed tape")

    algebra = exhaustive_algebra_audit()
    if not algebra["all_checks_passed"]:
        raise AssertionError("bounded algebra audit found a counterexample")
    ablations = edit_ablations(parent, left_raw, right_raw, LEFT_INSERT, RIGHT_INSERT)
    grammar_census = grammar_inventory_census()
    result = {
        "check_id": "asymmetric-568-to-630-live-residual-splice",
        "sources": {
            "generator": "experiments/luna6_god_dog_live_residual_growth_20260923.py",
            "saved_run": str(RUN_PATH.relative_to(ROOT)),
            "parent_run": str(PARENT_PATH.relative_to(ROOT)),
        },
        "parent": {
            "normalized_letters": len(parent_tape),
            "sha256": parent_sha,
        },
        "splice": {
            "normalized_cursors": [LEFT_CUT, RIGHT_CUT],
            "raw_cursors": [left_raw, right_raw],
            "A_letters": len(a),
            "q": q,
            "C_letters": len(center),
            "right_residual": right_q,
            "U": u,
            "U_letters": len(u),
            "V": v,
            "V_letters": len(v),
            "equal_inserted_letter_lengths": len(u) == len(v),
            "equation": "U + q = q + reverse(V)",
            "equation_holds": equation_holds,
            "outer_and_center_invariants_hold": a == suffix[::-1] and center == center[::-1],
            "full_splice_closes": splice_closes,
        },
        "child": {
            "normalized_letters": len(tape),
            "growth_over_parent": len(tape) - len(parent_tape),
            "sha256": hash_forward,
            "outside_in_exact": independent_audit["exact"],
            "first_mismatch": independent_audit["first_mismatch"],
            "project_validator_exact": project_audit,
            "forward_reverse_hashes_equal": hash_forward == hash_reverse,
            "exact": exact,
        },
        "bounded_algebra_audit": algebra,
        "bounded_672_clause_inventory_census": grammar_census,
        "ablations": ablations,
        "interpretation": (
            "The paired edit is mechanically sufficient at this saved seam. "
            "This check does not test whether the inserted clauses are natural, "
            "coherent, novel, or readable."
        ),
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
