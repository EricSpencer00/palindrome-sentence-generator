"""Carry the 54-letter NP residual into bounded clause continuations.

This is a strict follow-up to ``seed_np_cross_role_intersection_20260922``.
The exact 54-letter control is retained byte-for-byte, while added material
is restricted to repository-lexicon words.  Two small operators try distinct
cross-role continuations at the live ``memohero + m = m + reverse(morehome)``
seam.  A failed row is useful only when its opposing cursor, residual, and
grammar obstruction are persisted; no rejected center-pair row is reused.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks
from llm_palindrome.lexicon import is_real_word, load_lexicon

ID = "seed-np-cross-role-residual-continuation-20260922"
OUT = ROOT / "runs" / f"{ID}.json"

CONTROL = "An aide rips nine memo-hero memos. Some more home men inspire Diana."
CONTROL_SHA256 = "2f88268e3a920af5ceb67cfb20d1498ef5ce47e91d8800c937639cc8ce376268"

# The source experiment's seam is not guessed from the new prose.
SOURCE_SEAM = {
    "residual": "m",
    "left_exposure": "memoherom",
    "right_exposure": "memoherom",
    "equation": "memohero + m = m + reverse(morehome)",
    "left_role": "nine memo-hero memos: object of rips",
    "right_role": "some more home men: subject of inspire",
    "nonempty": True,
}

# Added words are chosen from the repository lexicon.  The inherited control
# contains ``memo-hero`` and the source's lexicalized plural surface; those
# are explicitly scoped as inherited, never as newly introduced words.
OPERATORS = (
    {
        "id": "relative_np_roles",
        "seam": "after object NP / before subject NP",
        "left_template": "An aide rips nine memo-hero memos that {left}.",
        "right_template": "Some more home men who {right} inspire Diana.",
        "left": "Mara helps",
        "right": "stop a dog",
        "roles": {
            "left": {"subject": "Mara", "verb": "helps", "object": "memos (relative gap)"},
            "right": {"subject": "men (relative who)", "verb": "stop", "object": "a dog"},
        },
    },
    {
        "id": "relative_np_role_shift",
        "seam": "after object NP / before subject NP",
        "left_template": "An aide rips nine memo-hero memos that {left}.",
        "right_template": "Some more home men who {right} inspire Diana.",
        "left": "Nora spots",
        "right": "stop a dog",
        "roles": {
            "left": {"subject": "Nora", "verb": "spots", "object": "memos (relative gap)"},
            "right": {"subject": "men (relative who)", "verb": "stop", "object": "a dog"},
        },
    },
    {
        "id": "subordinate_cross_role",
        "seam": "object adjunct / subject adjunct",
        "left_template": "An aide rips nine memo-hero memos, while {left}.",
        "right_template": "Some more home men, while {right}, inspire Diana.",
        "left": "Diana helps",
        "right": "a dog stops a memo",
        "roles": {
            "left": {"subject": "Diana", "verb": "helps", "object": "implicit"},
            "right": {"subject": "a dog", "verb": "stops", "object": "a memo"},
        },
    },
)


def normalize(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.casefold()))


def independent_audit(text: str) -> dict[str, object]:
    letters = normalize(text)
    left, right = 0, len(letters) - 1
    while left < right and letters[left] == letters[right]:
        left += 1
        right -= 1
    forward = hashlib.sha256(letters.encode()).hexdigest()
    reverse = hashlib.sha256(letters[::-1].encode()).hexdigest()
    return {
        "normalized": letters,
        "letters": len(letters),
        "two_pointer_exact": bool(letters) and left >= right,
        "first_mismatch": None if left >= right else [left, right],
        "first_mismatch_letters": (
            None if left >= right else [letters[left], letters[right]]
        ),
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "hashes_agree": forward == reverse,
    }


def added_words(operator: dict[str, object]) -> tuple[str, ...]:
    words = (
        str(operator["left"]).replace(",", "").split()
        + str(operator["right"]).replace(",", "").split()
    )
    return tuple(normalize(word) for word in words)


def added_lexicon_gate(operator: dict[str, object]) -> dict[str, object]:
    lexicon = load_lexicon(str(ROOT / "data" / "lexicon.txt"))
    words = added_words(operator)
    checks = {
        word: {
            "project_lexicon": is_real_word(word, lexicon),
            "self_palindromic": len(word) > 1 and word == word[::-1],
        }
        for word in words
    }
    return {
        "scope": "new operator words only; inherited control is not reclassified",
        "all_project_lexicon": all(row["project_lexicon"] for row in checks.values()),
        "no_new_self_palindromic_word": not any(
            row["self_palindromic"] for row in checks.values()
        ),
        "words": checks,
    }


def online_obstruction(text: str, seam_letters: int = 27) -> dict[str, object]:
    """Record the first opposing-cursor obstruction without repairing it."""
    letters = normalize(text)
    left = right = 0
    trace: list[dict[str, object]] = []
    while left < len(letters) // 2 and right < len(letters):
        opposite = len(letters) - 1 - left
        row = {
            "left_cursor": left,
            "right_cursor": opposite,
            "left_char": letters[left],
            "right_char": letters[opposite],
            "residual_owner": None,
            "residual": "",
        }
        trace.append(row)
        if letters[left] != letters[opposite]:
            return {
                "closed": False,
                "left_cursor": left,
                "right_cursor": opposite,
                "left_char": letters[left],
                "right_char": letters[opposite],
                "residual_owner": "right_event_shift",
                "residual": letters[opposite : opposite + 8],
                "grammar_state": (
                    "added continuation changes the opposing subject/object frame "
                    "before the inherited NP seam is reached"
                ),
                "seam_cursor": seam_letters,
                "trace_prefix": trace,
            }
        left += 1
        right += 1
    return {
        "closed": True,
        "left_cursor": left,
        "right_cursor": len(letters) - 1 - left,
        "residual_owner": None,
        "residual": "",
        "grammar_state": "no mismatch in scanned prefix",
        "seam_cursor": seam_letters,
        "trace_prefix": trace,
    }


def shortcut_gates(text: str, operator: dict[str, object]) -> dict[str, object]:
    added_tape = normalize(str(operator["left"]) + str(operator["right"]))
    return {
        "finished_tape_reversal": False,
        "post_hoc_character_repair": False,
        "whole_sentence_sweep": False,
        "catalogue_text": False,
        "word_order_symmetry": False,
        "independently_left_reverse_right": added_tape == added_tape[::-1],
        "added_material_has_no_self_palindromic_word": added_lexicon_gate(operator)[
            "no_new_self_palindromic_word"
        ],
        "inherited_control_proper_spans_not_reclassified": True,
    }


def build_row(operator: dict[str, object]) -> dict[str, object]:
    rendered = str(operator["left_template"]).format(left=operator["left"]) + " " + str(
        operator["right_template"]
    ).format(right=operator["right"])
    audit = independent_audit(rendered)
    obstruction = online_obstruction(rendered)
    lexicon_gate = added_lexicon_gate(operator)
    checks = mechanical_admission_checks(rendered, min_letters=55, max_letters=180)
    return {
        "id": str(operator["id"]),
        "rendered": rendered,
        "length": audit["letters"],
        "semantic_roles": operator["roles"],
        "source_live_residual": SOURCE_SEAM,
        "event_residual_obstruction": obstruction,
        "independent_exact_audit": audit,
        "mechanical_diagnostics": checks,
        "project_lexicon_gate": lexicon_gate,
        "novelty_shortcut_gates": shortcut_gates(rendered, operator),
        "provenance": {
            "method": "bounded cross-role SVO continuation at inherited NP residual",
            "source_experiment": "experiments/seed_np_cross_role_intersection_20260922.py",
            "seam": operator["seam"],
            "added_material_is_complete_event_or_relative": True,
            "event_residual_consumed_by_surrounding_np": False,
            "finished_tape_reversal": False,
            "post_hoc_character_repair": False,
            "catalogue_text": False,
            "reader_packet_used_as_evidence": False,
            "prior_center_pair_reused": False,
        },
        "reader_status": "not_certified; no exact closure",
        "promotion_status": "rejected_obstruction",
    }


def build_payload() -> dict[str, object]:
    control_audit = independent_audit(CONTROL)
    assert control_audit["letters"] == 54
    assert control_audit["two_pointer_exact"]
    assert control_audit["sha256_forward"] == CONTROL_SHA256
    rows = [build_row(operator) for operator in OPERATORS]
    return {
        "experiment_id": ID,
        "method": "bounded cross-role SVO continuation with inherited NP residual ownership",
        "control": {
            "rendered": CONTROL,
            "audit": control_audit,
            "source_seam": SOURCE_SEAM,
            "role": "54-letter exact control; retained separately",
        },
        "preserved_frontiers": {
            "568": {
                "artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
                "id": "outer-causal-scene-568-working-incumbent",
                "letters": 568,
                "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
            },
            "666": {
                "artifact": "runs/incumbent-666-comparison-alternative-20260922.json",
                "id": "comparison-alternative-nora-sees-666",
                "letters": 666,
                "sha256": "cafd77235f82d9ff4f68814dc7e03d196bf719bf1ec5d541e172073502e12297",
            },
        },
        "rejected_evidence": {
            "commit": "a9d23763",
            "artifact": "runs/seed-np-cross-role-clause-growth-20260922.json",
            "reason": "prior rows used a complete center pair and Aram/Aron; preserved only as rejected evidence",
            "rows": [
                {"id": "spot-stop", "letters": 82, "sha256": "b54d261662a8ef3baec37176fb442e6c983e45e8db08517ffcea04a3647477c5"},
                {"id": "see-see", "letters": 80, "sha256": "f15ccc62f7f03f5b6a8419e71662d6355b6cccf92f914339d0d9c7a430904f28"},
                {"id": "nora-spot-stop", "letters": 82, "sha256": "13fafbbed28ede837867197728674d611ad8ff1b8cc73578dec75f1c9caccdf5"},
            ],
        },
        "stats": {
            "bounded_operators": len(rows),
            "exact_closures": sum(row["independent_exact_audit"]["two_pointer_exact"] for row in rows),
            "children_longer_than_54": sum(row["length"] > 54 for row in rows),
            "reader_certified": 0,
        },
        "rows": rows,
        "operator_change": {
            "first": "relative NP roles",
            "second": "subordinate cross-role adjuncts",
            "changed_within_run": True,
            "closure": False,
            "obstruction": "all bounded continuations mismatch before the inherited seam; residual remains right-owned after the frame shift",
        },
        "programmatic_metrics_are_diagnostic": True,
        "status": "no exact closure under bounded project-lexicon continuation; obstruction persisted",
        "next_operator": "retain the 54 control and widen only the typed NP-to-relative boundary grammar after a new independently authored seam, not the rejected center pair",
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_generator": "experiments/seed_np_cross_role_intersection_20260922.py",
            "reader_packet": "not used as evidence",
        },
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    for row in payload["rows"]:
        print(row["rendered"])


if __name__ == "__main__":
    main()
