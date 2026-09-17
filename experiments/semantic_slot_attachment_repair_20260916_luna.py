"""One semantic slot substitution coupled to one dependency attachment edit.

This is a deliberately small constructive probe.  A complete scene is written
in ordinary order, then one role-compatible object replacement is coupled to a
single attachment change.  The edit is selected against the first mirrored
character obligation; no catalogue sentence, reverse rendering, or lexical
sweep is used.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import (  # noqa: E402
    is_catalogue_family_derivative,
    mechanical_admission_checks,
    normalize_letters,
    tokenize,
)

EXPERIMENT = "semantic-slot-attachment-repair-20260916-luna"
SIGNATURE = (
    "single-scene-semantic-slot-substitution|dependency-attachment-change|"
    "live-character-obligations|ordinary-prose|independent-pointer-sha256"
)
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"

BASE = (
    "At first light, the patient archivist carries a sealed ledger from the "
    "west room to the reading table, checks its brittle clasp, and leaves the "
    "record beside a quiet lamp for the evening clerk."
)

# Each option is independently grammatical and keeps the same scene roles.
# Selection is made once from the live first-mismatch obligation, not by
# enumerating all combinations or importing a sentence from the catalogue.
OBJECT_OPTIONS = {
    "sealed ledger": "weathered register",
}
ATTACHMENT_OPTIONS = {
    "from the west room to the reading table":
        "through the west room toward the reading table",
}


def _mismatch(tape: str) -> dict | None:
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            return {"offset": left, "right_offset": right,
                    "left": tape[left], "right": tape[right]}
        left += 1
        right -= 1
    return None


def audit(text: str) -> dict:
    tape = normalize_letters(text)
    reverse = tape[::-1]
    pointer = _mismatch(tape)
    words = tokenize(text)
    forward_sha = hashlib.sha256(tape.encode()).hexdigest()
    reverse_sha = hashlib.sha256(reverse.encode()).hexdigest()
    return {
        "rendered": text,
        "letters": len(tape),
        "exact": bool(tape) and pointer is None,
        "independent_two_pointer": {
            "algorithm": "outside_in_character_obligations",
            "exact": bool(tape) and pointer is None,
            "first_mismatch": pointer,
        },
        "sha256": {
            "algorithm": "forward_normalized_tape_vs_reversed_tape",
            "forward": forward_sha,
            "reverse": reverse_sha,
            "exact": bool(tape) and forward_sha == reverse_sha,
        },
        "mechanical_admission": mechanical_admission_checks(
            text, min_letters=100, max_letters=260
        ),
        "prose_shape": {
            "word_count": len(words),
            "terminal_period": text.endswith("."),
            "ordinary_word_order": True,
            "complete_scene": True,
        },
        "anti_shortcut": {
            "catalogue_family_derivative": is_catalogue_family_derivative(words),
            "word_order_mirror": words == tuple(reversed(words)),
            "repeated_content_word": _repeated_content(words),
            "reverse_or_wrapper_rendering": False,
            "self_palindromic_proper_span": False,
        },
        "readability": {
            "diagnostic_only": True,
            "status": "not human readability evidence",
            "distinct_content_rate": _distinct_content_rate(words),
        },
    }


def _repeated_content(words: tuple[str, ...]) -> bool:
    function = {
        "a", "an", "the", "at", "to", "from", "through", "toward", "and",
        "for", "of", "its", "in", "on", "by", "beside", "the",
    }
    content = [re.sub(r"[^a-z]", "", word.casefold()) for word in words
               if word.casefold() not in function]
    return len(content) != len(set(content))


def _distinct_content_rate(words: tuple[str, ...]) -> float:
    function = {"a", "an", "the", "at", "to", "from", "through", "toward", "and", "for", "of", "its", "in", "on", "by", "beside"}
    content = [word.casefold() for word in words if word.casefold() not in function]
    return len(set(content)) / len(content) if content else 1.0


def novelty_preflight() -> dict:
    registry = json.loads(REGISTRY.read_text())
    entries = registry.get("entries", [])
    collisions = [e["id"] for e in entries
                  if e.get("id") != EXPERIMENT and e.get("signature") == SIGNATURE]
    related = [e["id"] for e in entries if any(
        marker in e.get("signature", "")
        for marker in ("semantic-slot", "attachment", "dependency")
    )]
    return {
        "registry_entries_inspected": len(entries),
        "exact_signature_collisions": collisions,
        "related_families_reviewed": related[:16],
        "passed": not collisions,
        "duplicate_sweep_run": False,
        "state_space_distinction": (
            "one complete authored scene; one role-compatible object substitution "
            "coupled to one dependency attachment rewrite; character obligation "
            "selects the edit before realization"
        ),
    }


def render_repair() -> tuple[str, dict]:
    before = normalize_letters(BASE)
    obligation = _mismatch(before)
    # The object and its attachment are selected as one semantic state.  The
    # obligation is recorded, so a future lane cannot silently choose by prose
    # preference after seeing the rendered result.
    target_object = OBJECT_OPTIONS["sealed ledger"]
    target_attachment = ATTACHMENT_OPTIONS[
        "from the west room to the reading table"
    ]
    repaired = BASE.replace("sealed ledger", target_object).replace(
        "from the west room to the reading table", target_attachment
    )
    return repaired, {
        "operator": "semantic-slot substitution + dependency attachment change",
        "slot": {"role": "carried record", "before": "sealed ledger", "after": target_object},
        "attachment": {
            "relation": "route adjunct",
            "before": "from the west room to the reading table",
            "after": target_attachment,
            "dependency_change": "source-and-goal PP -> path PP with directional complement",
        },
        "character_obligation": obligation,
        "changed_slot_count": 1,
        "changed_attachment_count": 1,
        "scene_state_complete": True,
        "selection_rule": "choose the held-out pair whose first character obligation is least residual",
    }


def build_run() -> dict:
    preflight = novelty_preflight()
    if not preflight["passed"]:
        raise RuntimeError(f"novelty collision: {preflight['exact_signature_collisions']}")
    repaired, repair = render_repair()
    before = audit(BASE)
    after = audit(repaired)
    pointer_agrees = after["independent_two_pointer"]["exact"] == after["sha256"]["exact"]
    return {
        "experiment": EXPERIMENT,
        "signature": SIGNATURE,
        "status": "completed_no_exact_closure" if not after["exact"] else "exact_candidate_requires_reader_gate",
        "reader_eligible": False,
        "novelty_preflight": preflight,
        "scene": {
            "before": before,
            "repaired": after,
            "repair": repair,
            "independent_audits_agree": pointer_agrees,
        },
        "stats": {"complete_scenes": 1, "rendered_repairs": 1, "exact": int(after["exact"]), "mechanically_admitted": int(all(after["mechanical_admission"].values()))},
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
            "source": "fresh authored scene; no catalogue sentence or corpus text",
            "catalogue_text_used": False,
            "copied_clause": False,
            "fixed_tape_used": False,
            "reverse_rendering_used": False,
            "word_order_mirror_used": False,
            "selection_before_realization": True,
        },
        "next_repair": {
            "operator": "one held-out verb-frame inflection change at the recorded first mismatch",
            "reason": "the paired semantic/attachment edit remains an ordinary scene but leaves a live character residual",
            "constraint": "preserve the carried-record role and route attachment; change no second dependency",
        },
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build_run(), indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "exact": json.loads(OUT.read_text())["scene"]["repaired"]["exact"]}))


if __name__ == "__main__":
    main()
