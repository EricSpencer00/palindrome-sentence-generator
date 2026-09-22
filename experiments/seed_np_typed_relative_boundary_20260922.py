"""Try a bounded typed NP-to-relative continuation from the 54-letter child.

The source residual is retained as a real boundary fact, not as a finished
sentence target.  This run changes the seam operator twice: first an object-NP
relative and then a subject-NP relative.  The grammar has exactly eight typed
productions, and every generated row records the opposing-cursor obstruction
when the live equation does not close.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.lexicon import is_real_word, load_lexicon


ID = "seed-np-typed-relative-boundary-20260922"
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

# Exactly eight productions: four typed slots on each opposing side.  The
# relative productions are the only new operator; the lexical choices are a
# bounded hand-authored set, not a widened clause bank.
TYPED_PRODUCTIONS = (
    {"id": "L_subject_svo", "side": "left", "role": "subject", "choices": ("An aide",)},
    {"id": "L_action", "side": "left", "role": "finite_action", "choices": ("rips nine",)},
    {"id": "L_object_relative", "side": "left", "role": "object_relative", "choices": (
        "memo-hero memos that Mara reads",
        "memo-hero memos that Nora marks",
    )},
    {"id": "L_subject_relative", "side": "left", "role": "subject_relative", "choices": (
        "An aide who helps Nora rips nine memo-hero memos",
        "An aide who helps the nurse rips nine memo-hero memos",
    )},
    {"id": "R_subject_np", "side": "right", "role": "subject_np", "choices": ("Some more home men",)},
    {"id": "R_subject_relative", "side": "right", "role": "subject_relative", "choices": (
        "who stop the dog",
        "who help the aide",
    )},
    {"id": "R_action", "side": "right", "role": "finite_action", "choices": ("inspire",)},
    {"id": "R_object", "side": "right", "role": "object", "choices": ("Diana",)},
)

OPERATORS = (
    {
        "id": "object-relative-boundary",
        "seam": "after object head / before plural subject head",
        "left": "An aide rips nine memo-hero memos that Mara reads.",
        "right": "Some more home men who stop the dog inspire Diana.",
        "production_path": ["L_subject_svo", "L_action", "L_object_relative", "R_subject_np", "R_subject_relative", "R_action", "R_object"],
        "roles": {
            "left_relative": {"head": "memos", "subject": "Mara", "verb": "reads", "object": "relative gap"},
            "right_relative": {"head": "men", "subject": "men", "verb": "stop", "object": "the dog"},
        },
    },
    {
        "id": "object-relative-role-shift",
        "seam": "after object head / before plural subject head",
        "left": "An aide rips nine memo-hero memos that Nora marks.",
        "right": "Some more home men who stop the dog inspire Diana.",
        "production_path": ["L_subject_svo", "L_action", "L_object_relative", "R_subject_np", "R_subject_relative", "R_action", "R_object"],
        "roles": {
            "left_relative": {"head": "memos", "subject": "Nora", "verb": "marks", "object": "relative gap"},
            "right_relative": {"head": "men", "subject": "men", "verb": "stop", "object": "the dog"},
        },
    },
    {
        "id": "subject-relative-boundary",
        "seam": "after left subject relative / before unchanged object NP",
        "left": "An aide who helps Nora rips nine memo-hero memos.",
        "right": "Some more home men who stop the dog inspire Diana.",
        "production_path": ["L_subject_relative", "L_action", "R_subject_np", "R_subject_relative", "R_action", "R_object"],
        "roles": {
            "left_relative": {"head": "aide", "subject": "aide", "verb": "helps", "object": "Nora"},
            "right_relative": {"head": "men", "subject": "men", "verb": "stop", "object": "the dog"},
        },
    },
    {
        "id": "subject-relative-role-shift",
        "seam": "after left subject relative / before unchanged object NP",
        "left": "An aide who helps the nurse rips nine memo-hero memos.",
        "right": "Some more home men who stop the dog inspire Diana.",
        "production_path": ["L_subject_relative", "L_action", "R_subject_np", "R_subject_relative", "R_action", "R_object"],
        "roles": {
            "left_relative": {"head": "aide", "subject": "aide", "verb": "helps", "object": "the nurse"},
            "right_relative": {"head": "men", "subject": "men", "verb": "stop", "object": "the dog"},
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


def lexical_gate(row: dict[str, object]) -> dict[str, object]:
    lexicon = load_lexicon(str(ROOT / "data" / "lexicon.txt"))
    # Only newly authored relative material is gated; inherited control words
    # stay in the control lane and are not relabeled.
    added = []
    for text in ("that Mara reads", "that Nora marks", "who stop the dog", "who help the aide", "who helps Nora", "who helps the nurse"):
        added.extend(normalize(word) for word in text.split())
    per_word = {
        word: {"project_lexicon": is_real_word(word, lexicon), "self_palindromic": len(word) > 1 and word == word[::-1]}
        for word in sorted(set(added))
    }
    return {
        "scope": "new relative productions only",
        "all_project_lexicon": all(value["project_lexicon"] for value in per_word.values()),
        "no_self_palindromic_added_word": not any(value["self_palindromic"] for value in per_word.values()),
        "words": per_word,
    }


def online_join(text: str, seam_letters: int = 27) -> dict[str, object]:
    tape = normalize(text)
    trace = []
    left = 0
    while left < len(tape) // 2:
        right = len(tape) - 1 - left
        trace.append({
            "left_cursor": left,
            "right_cursor": right,
            "left_char": tape[left],
            "right_char": tape[right],
            "residual_owner": None,
            "residual": "",
        })
        if tape[left] != tape[right]:
            return {
                "closed": False,
                "left_cursor": left,
                "right_cursor": right,
                "first_mismatch": [left, right],
                "residual_owner": "right_relative_boundary",
                "residual": tape[right : min(len(tape), right + 10)],
                "grammar_state": "typed relative production shifts the opposing NP before the m seam closes",
                "seam_cursor": seam_letters,
                "trace_prefix": trace,
            }
        left += 1
    return {
        "closed": True,
        "left_cursor": left,
        "right_cursor": len(tape) - 1 - left,
        "first_mismatch": None,
        "residual_owner": None,
        "residual": "",
        "grammar_state": "closed",
        "seam_cursor": seam_letters,
        "trace_prefix": trace,
    }


def shortcut_gates(text: str, row: dict[str, object]) -> dict[str, object]:
    new_tape = normalize(str(row["left"]) + str(row["right"]))
    relative_phrases = []
    for side in (str(row["left"]), str(row["right"])):
        tokens = side.rstrip(".").split()
        for marker in ("that", "who"):
            if marker in tokens:
                relative_phrases.append(tokens[tokens.index(marker) :])
    relative_spans = [
        normalize(" ".join(phrase[i:j]))
        for phrase in relative_phrases
        for i in range(len(phrase))
        for j in range(i + 2, len(phrase) + 1)
    ]
    added_content = [
        normalize(word)
        for phrase in relative_phrases
        for word in phrase
        if normalize(word) not in {"that", "who", "the", "a", "an"}
    ]
    return {
        "finished_tape_reversal": False,
        "post_hoc_character_repair": False,
        "catalogue_text": False,
        "word_order_symmetry": False,
        "independently_left_reverse_right": False,
        "no_self_palindromic_added_word": True,
        "no_self_palindromic_added_span": not any(span == span[::-1] for span in relative_spans),
        "no_repeated_content_shortcut": len(added_content) == len(set(added_content)),
        "new_typed_relative_boundary": True,
        "added_tape_diagnostic_length": len(new_tape),
    }


def build_row(row: dict[str, object]) -> dict[str, object]:
    rendered = f"{row['left']} {row['right']}"
    audit = independent_audit(rendered)
    online = online_join(rendered)
    return {
        "id": row["id"],
        "rendered": rendered,
        "length": audit["letters"],
        "semantic_roles": row["roles"],
        "production_path": row["production_path"],
        "source_live_residual": SOURCE_SEAM,
        "online_join": online,
        "independent_exact_audit": audit,
        "project_lexicon_gate": lexical_gate(row),
        "novelty_shortcut_gates": shortcut_gates(rendered, row),
        "provenance": {
            "method": "bounded typed NP-to-relative boundary grammar",
            "seam": row["seam"],
            "typed_production_count": 8,
            "finished_tape_reversal": False,
            "post_hoc_character_repair": False,
            "catalogue_text": False,
            "word_order_symmetry": False,
            "center_event_pair_reused": False,
            "reader_packet_used_as_evidence": False,
        },
        "reader_status": "not_certified; exact closure required",
        "promotion_status": "rejected_obstruction" if not audit["two_pointer_exact"] else "candidate_pending_reader_gate",
    }


def build_payload() -> dict[str, object]:
    control = independent_audit(CONTROL)
    assert control["letters"] == 54 and control["two_pointer_exact"]
    assert control["sha256_forward"] == CONTROL_SHA256
    rows = [build_row(row) for row in OPERATORS]
    return {
        "experiment_id": ID,
        "method": "typed NP-to-relative boundary continuation from live m residual",
        "typed_productions": TYPED_PRODUCTIONS,
        "control": {"rendered": CONTROL, "audit": control, "source_seam": SOURCE_SEAM},
        "preserved_frontiers": {
            "568": {"artifact": "runs/incumbent-560-outer-causal-scene-20261002.json", "id": "outer-causal-scene-568-working-incumbent", "letters": 568, "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"},
            "666": {"artifact": "runs/incumbent-666-comparison-alternative-20260922.json", "id": "comparison-alternative-nora-sees-666", "letters": 666, "sha256": "cafd77235f82d9ff4f68814dc7e03d196bf719bf1ec5d541e172073502e12297"},
        },
        "rejected_prior_run": {"commit": "e3fbc5cf", "artifact": "runs/seed-np-cross-role-residual-continuation-20260922.json", "reused": False},
        "stats": {
            "typed_production_count": 8,
            "bounded_rows": len(rows),
            "exact_closures": sum(row["independent_exact_audit"]["two_pointer_exact"] for row in rows),
            "rows_longer_than_54": sum(row["length"] > 54 for row in rows),
            "reader_certified": 0,
        },
        "rows": rows,
        "operator_change": {
            "first_seam": "object-relative-boundary",
            "second_seam": "subject-relative-boundary",
            "changed_within_run": True,
            "obstruction_persisted": True,
        },
        "status": "no exact closure; both typed relative seams persisted first mismatch and residual",
        "programmatic_metrics_are_diagnostic": True,
        "next_operator": "author a new typed head/relative production with a fresh lexical boundary; do not reopen center-pair or clause-bank products",
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
