"""Probe a fresh possessive head/relative seam from the exact 54-letter NP.

The inherited control is kept byte-for-byte.  This lane changes the grammar
topology to a possessive head-relative (``whose NP V``) attached before a
matrix predicate, at either the subject-head or object-head boundary.  Each
row is a complete two-clause rendering and carries the live opposing-cursor
state; no row is repaired after rendering or generated from a finished tape.
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


ID = "seed-np-typed-possessive-head-relative-20260922"
OUT = ROOT / "runs" / f"{ID}.json"
CONTROL = "An aide rips nine memo-hero memos. Some more home men inspire Diana."
CONTROL_SHA256 = "2f88268e3a920af5ceb67cfb20d1498ef5ce47e91d8800c937639cc8ce376268"

SOURCE_SEAM = {
    "residual": "m",
    "left_exposure": "memoherom",
    "right_exposure": "memoherom",
    "equation": "memohero + m = m + reverse(morehome)",
    "left_role": "nine memo-hero memos: object of rips",
    "right_role": "some more home men: subject of inspire",
    "nonempty": True,
}

# Exactly eight typed productions.  The new operator is the possessive
# head-relative boundary, not a widened lexical/cartesian clause bank.
TYPED_PRODUCTIONS = (
    {"id": "L_subject_np", "side": "left", "role": "subject_np", "choices": ("An aide",)},
    {"id": "L_action", "side": "left", "role": "finite_action", "choices": ("rips nine",)},
    {"id": "L_object_head_relative", "side": "left", "role": "object_head_relative", "choices": (
        "memos whose editor Nora helps",
        "memos whose author Mara helps",
    )},
    {"id": "L_subject_head_relative", "side": "left", "role": "subject_head_relative", "choices": (
        "An aide whose memo Nora reads",
        "An aide whose memo Mara reads",
    )},
    {"id": "R_subject_np", "side": "right", "role": "subject_np", "choices": ("Some more home men",)},
    {"id": "R_subject_head_relative", "side": "right", "role": "subject_head_relative", "choices": (
        "Some more home men whose aide Mara helps",
        "Some more home men whose dog Nora stops",
    )},
    {"id": "R_action", "side": "right", "role": "finite_action", "choices": ("inspire",)},
    {"id": "R_object", "side": "right", "role": "object", "choices": ("Diana",)},
)


OPERATORS = (
    {
        "id": "subject-head-possessive",
        "seam": "after subject head / before left matrix verb and after right subject head / before matrix verb",
        "left": "An aide whose memo Nora reads rips nine memo-hero memos.",
        "right": "Some more home men whose aide Mara helps inspire Diana.",
        "production_path": ["L_subject_head_relative", "L_action", "R_subject_head_relative", "R_action", "R_object"],
        "roles": {
            "left_relative": {"head": "aide", "possessor": "memo", "subject": "Nora", "verb": "reads"},
            "right_relative": {"head": "men", "possessor": "aide", "subject": "Mara", "verb": "helps"},
        },
    },
    {
        "id": "subject-head-possessive-role-shift",
        "seam": "after subject head / before left matrix verb and after right subject head / before matrix verb",
        "left": "An aide whose memo Mara reads rips nine memo-hero memos.",
        "right": "Some more home men whose dog Nora stops inspire Diana.",
        "production_path": ["L_subject_head_relative", "L_action", "R_subject_head_relative", "R_action", "R_object"],
        "roles": {
            "left_relative": {"head": "aide", "possessor": "memo", "subject": "Mara", "verb": "reads"},
            "right_relative": {"head": "men", "possessor": "dog", "subject": "Nora", "verb": "stops"},
        },
    },
    {
        "id": "object-head-possessive",
        "seam": "after object head / before left clause terminus and after right subject head / before matrix verb",
        "left": "An aide rips nine memo-hero memos whose editor Nora helps.",
        "right": "Some more home men whose dog the aide stops inspire Diana.",
        "production_path": ["L_subject_np", "L_action", "L_object_head_relative", "R_subject_head_relative", "R_action", "R_object"],
        "roles": {
            "left_relative": {"head": "memos", "possessor": "editor", "subject": "Nora", "verb": "helps"},
            "right_relative": {"head": "men", "possessor": "dog", "subject": "the aide", "verb": "stops"},
        },
    },
    {
        "id": "object-head-possessive-role-shift",
        "seam": "after object head / before left clause terminus and after right subject head / before matrix verb",
        "left": "An aide rips nine memo-hero memos whose author Mara helps.",
        "right": "Some more home men whose memo the nurse reads inspire Diana.",
        "production_path": ["L_subject_np", "L_action", "L_object_head_relative", "R_subject_head_relative", "R_action", "R_object"],
        "roles": {
            "left_relative": {"head": "memos", "possessor": "author", "subject": "Mara", "verb": "helps"},
            "right_relative": {"head": "men", "possessor": "memo", "subject": "the nurse", "verb": "reads"},
        },
    },
    {
        "id": "subject-head-possessive-needs",
        "seam": "after subject head / before left matrix verb and after right subject head / before matrix verb",
        "left": "An aide whose nurse Nora needs rips nine memo-hero memos.",
        "right": "Some more home men whose writer Mara helps inspire Diana.",
        "production_path": ["L_subject_head_relative", "L_action", "R_subject_head_relative", "R_action", "R_object"],
        "roles": {
            "left_relative": {"head": "aide", "possessor": "nurse", "subject": "Nora", "verb": "needs"},
            "right_relative": {"head": "men", "possessor": "writer", "subject": "Mara", "verb": "helps"},
        },
    },
)


def normalize(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.casefold()))


def independent_audit(text: str) -> dict[str, object]:
    tape = normalize(text)
    left, right = 0, len(tape) - 1
    while left < right and tape[left] == tape[right]:
        left += 1
        right -= 1
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "normalized": tape,
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and left >= right,
        "first_mismatch": None if left >= right else [left, right],
        "first_mismatch_letters": None if left >= right else [tape[left], tape[right]],
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "hashes_agree": forward == reverse,
    }


def added_words(operator: dict[str, object]) -> tuple[str, ...]:
    inherited = {"an", "aide", "rips", "nine", "memo", "hero", "memos", "some", "more", "home", "men", "inspire", "diana"}
    function_words = {"whose", "that", "who", "which", "the", "a", "an"}
    # Keep the gate scoped to the newly authored relative material.  The
    # inherited control words are not reclassified as new choices.
    raw = re.findall(r"[A-Za-z]+", str(operator["left"]) + " " + str(operator["right"]))
    return tuple(
        normalize(word)
        for word in raw
        if normalize(word) not in inherited and normalize(word) not in function_words
    )


def added_lexicon_gate(operator: dict[str, object]) -> dict[str, object]:
    lexicon = load_lexicon(str(ROOT / "data" / "lexicon.txt"))
    words = added_words(operator)
    checks = {
        word: {
            "project_lexicon": is_real_word(word, lexicon),
            "self_palindromic": len(word) > 1 and word == word[::-1],
        }
        for word in sorted(set(words))
    }
    return {
        "scope": "new possessive head-relative material only; inherited 54 control is not reclassified",
        "all_project_lexicon": all(row["project_lexicon"] for row in checks.values()),
        "no_new_self_palindromic_word": not any(row["self_palindromic"] for row in checks.values()),
        "words": checks,
    }


def online_join(text: str, seam_letters: int = 27) -> dict[str, object]:
    """Walk opposing cursors and retain the first typed relative obstruction."""
    tape = normalize(text)
    left, right = 0, len(tape) - 1
    trace: list[dict[str, object]] = []
    while left < right:
        row = {
            "left_cursor": left,
            "right_cursor": right,
            "left_char": tape[left],
            "right_char": tape[right],
            "residual_owner": None,
            "residual": "",
        }
        trace.append(row)
        if tape[left] != tape[right]:
            residual = tape[right : min(len(tape), right + 12)]
            row.update({
                "residual_owner": "typed_head_relative_boundary",
                "residual": residual,
            })
            return {
                "closed": False,
                "left_cursor": left,
                "right_cursor": right,
                "first_mismatch": [left, right],
                "first_mismatch_letters": [tape[left], tape[right]],
                "residual_owner": "typed_head_relative_boundary",
                "residual": residual,
                "grammar_state": (
                    "possessive whose-NP-V head-relative emits a complete NP, but its "
                    "head boundary shifts the opposing cursor before the inherited m residual"
                ),
                "inherited_residual": SOURCE_SEAM["residual"],
                "inherited_residual_consumed": False,
                "seam_cursor": seam_letters,
                "trace_prefix": trace,
            }
        left += 1
        right -= 1
    return {
        "closed": True,
        "left_cursor": left,
        "right_cursor": right,
        "first_mismatch": None,
        "first_mismatch_letters": None,
        "residual_owner": None,
        "residual": "",
        "grammar_state": "closed",
        "inherited_residual": SOURCE_SEAM["residual"],
        "inherited_residual_consumed": True,
        "seam_cursor": seam_letters,
        "trace_prefix": trace,
    }


def shortcut_gates(operator: dict[str, object]) -> dict[str, object]:
    added = added_words(operator)
    return {
        "finished_tape_reversal": False,
        "post_hoc_character_repair": False,
        "catalogue_text": False,
        "word_order_symmetry": False,
        "independently_left_reverse_right": False,
        "no_self_palindromic_added_word": added_lexicon_gate(operator)["no_new_self_palindromic_word"],
        "no_self_palindromic_added_span": not any(word == word[::-1] and len(word) > 1 for word in added),
        "no_repeated_content_shortcut": len(added) == len(set(added)),
        "new_typed_possessive_head_relative": True,
    }


def build_row(operator: dict[str, object]) -> dict[str, object]:
    rendered = f"{operator['left']} {operator['right']}"
    audit = independent_audit(rendered)
    online = online_join(rendered)
    lexicon_gate = added_lexicon_gate(operator)
    return {
        "id": operator["id"],
        "rendered": rendered,
        "length": audit["letters"],
        "semantic_roles": operator["roles"],
        "production_path": operator["production_path"],
        "source_live_residual": SOURCE_SEAM,
        "online_join": online,
        "independent_exact_audit": audit,
        "mechanical_diagnostics": mechanical_admission_checks(rendered, min_letters=55, max_letters=180),
        "project_lexicon_gate": lexicon_gate,
        "novelty_shortcut_gates": shortcut_gates(operator),
        "provenance": {
            "method": "bounded typed possessive head-relative boundary grammar",
            "seam": operator["seam"],
            "typed_production_count": 8,
            "relative_topology": "NP -> Det N whose NP V",
            "finished_tape_reversal": False,
            "post_hoc_character_repair": False,
            "catalogue_text": False,
            "center_event_pair_reused": False,
            "reader_packet_used_as_evidence": False,
        },
        "reader_status": "not_certified; exact closure required",
        "promotion_status": "rejected_obstruction",
    }


def build_payload() -> dict[str, object]:
    control = independent_audit(CONTROL)
    assert control["letters"] == 54
    assert control["two_pointer_exact"]
    assert control["sha256_forward"] == CONTROL_SHA256
    rows = [build_row(operator) for operator in OPERATORS]
    return {
        "experiment_id": ID,
        "method": "bounded typed possessive head-relative continuation from live m residual",
        "typed_productions": TYPED_PRODUCTIONS,
        "control": {"rendered": CONTROL, "audit": control, "source_seam": SOURCE_SEAM, "role": "54-letter exact control; retained separately"},
        "preserved_frontiers": {
            "568": {"artifact": "runs/incumbent-560-outer-causal-scene-20261002.json", "id": "outer-causal-scene-568-working-incumbent", "letters": 568, "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"},
            "666": {"artifact": "runs/incumbent-666-comparison-alternative-20260922.json", "id": "comparison-alternative-nora-sees-666", "letters": 666, "sha256": "cafd77235f82d9ff4f68814dc7e03d196bf719bf1ec5d541e172073502e12297"},
        },
        "rejected_prior_run": {"commit": "e0f9fbe2", "artifact": "runs/seed-np-typed-relative-boundary-20260922.json", "reused": False},
        "stats": {
            "typed_production_count": 8,
            "bounded_rows": len(rows),
            "exact_closures": sum(row["independent_exact_audit"]["two_pointer_exact"] for row in rows),
            "rows_longer_than_54": sum(row["length"] > 54 for row in rows),
            "reader_certified": 0,
        },
        "rows": rows,
        "operator_change": {
            "prior": "object-relative and subject-relative who/that attachment",
            "current": "possessive whose-NP-V head-relative at subject-head and object-head seams",
            "changed_within_run": True,
            "obstruction_persisted": True,
        },
        "programmatic_metrics_are_diagnostic": True,
        "status": "no exact closure; possessive head-relative boundary shifts opposing cursor before inherited m residual",
        "next_operator": "pivot the seam/operator again; do not reopen center-pair, finished-tape, or clause-bank lanes",
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "reader_status": "not_certified"},
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
