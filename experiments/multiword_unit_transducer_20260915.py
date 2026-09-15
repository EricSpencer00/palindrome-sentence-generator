"""Bounded multiword-unit transducer for exact sentence palindromes.

This pilot tests a construction dimension absent from the novelty registry:
two independently authored banks of idiomatic clause fragments are composed
as units, and a small reversible transducer may change one local unit by an
English affix or compound split/join.  It never decodes the reverse tape and
never copies or reverses a completed sentence.  Exactness is audited twice,
including an explicit two-pointer scan; the shared admission gate remains a
mechanical eligibility check rather than a readability certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

MIN_LETTERS = 39
MAX_LETTERS = 180
FAMILY_ID = "multiword-unit-transducer-local-repair"
STATE_SPACE_SIGNATURE = (
    "multiword-lexical-unit-transducer|independent-idiom-fragments|"
    "bounded-affix-compound-repair|one-local-unit-rewrite|"
    "boundary-resegmentation-with-syntax-preserved"
)


@dataclass(frozen=True)
class Fragment:
    """An independently authored idiomatic fragment with typed unit slots."""

    fragment_id: str
    side: str
    words: tuple[str, ...]
    slots: tuple[str, ...]
    note: str

    @property
    def tape(self) -> str:
        return "".join(self.words)


@dataclass(frozen=True)
class UnitRepair:
    """A local, reversible surface rewrite over one multiword lexical unit."""

    repair_id: str
    source: tuple[str, ...]
    target: tuple[str, ...]
    class_name: str
    syntax_note: str


# The two banks are authored independently.  They intentionally contain
# ordinary clauses, not corpus/POS/event/dialogue templates or reverse strings.
LEFT_FRAGMENTS = (
    Fragment("L01", "left", ("the", "baker", "packs", "a", "notebook"), ("det", "agent", "verb", "det", "object"), "baker packs a notebook"),
    Fragment("L02", "left", ("a", "gardener", "waters", "the", "sunflower"), ("det", "agent", "verb", "det", "object"), "gardener waters a sunflower"),
    Fragment("L03", "left", ("the", "pilot", "checks", "a", "raincoat"), ("det", "agent", "verb", "det", "object"), "pilot checks a raincoat"),
    Fragment("L04", "left", ("a", "teacher", "marks", "the", "workshop"), ("det", "agent", "verb", "det", "object"), "teacher marks a workshop"),
    Fragment("L05", "left", ("the", "sailor", "folds", "a", "seashell"), ("det", "agent", "verb", "det", "object"), "sailor folds a seashell"),
    Fragment("L06", "left", ("a", "carpenter", "paints", "the", "daylight"), ("det", "agent", "verb", "det", "object"), "carpenter paints a daylight mural"),
)

RIGHT_FRAGMENTS = (
    Fragment("R01", "right", ("the", "reader", "opens", "a", "notebook"), ("det", "agent", "verb", "det", "object"), "reader opens a notebook"),
    Fragment("R02", "right", ("a", "florist", "carries", "the", "sunflower"), ("det", "agent", "verb", "det", "object"), "florist carries a sunflower"),
    Fragment("R03", "right", ("the", "driver", "wears", "a", "raincoat"), ("det", "agent", "verb", "det", "object"), "driver wears a raincoat"),
    Fragment("R04", "right", ("a", "student", "leaves", "the", "workshop"), ("det", "agent", "verb", "det", "object"), "student leaves a workshop"),
    Fragment("R05", "right", ("the", "child", "finds", "a", "seashell"), ("det", "agent", "verb", "det", "object"), "child finds a seashell"),
    Fragment("R06", "right", ("a", "muralist", "waits", "in", "daylight"), ("det", "agent", "verb", "prep", "object"), "muralist waits in daylight"),
)

# Each entry is reversible in the sense that both surfaces are conventional
# English lexicalizations of the same unit.  Only one entry may fire per row.
REPAIRS = (
    UnitRepair("compound_notebook_split", ("notebook",), ("note", "book"), "compound_split", "object compound may be written as a noun-noun phrase"),
    UnitRepair("compound_sunflower_split", ("sunflower",), ("sun", "flower"), "compound_split", "object compound may be written as a noun-noun phrase"),
    UnitRepair("compound_raincoat_split", ("raincoat",), ("rain", "coat"), "compound_split", "object compound may be written as a noun-noun phrase"),
    UnitRepair("compound_workshop_split", ("workshop",), ("work", "shop"), "compound_split", "object compound may be written as a noun-noun phrase"),
    UnitRepair("compound_seashell_split", ("seashell",), ("sea", "shell"), "compound_split", "object compound may be written as a noun-noun phrase"),
    UnitRepair("compound_daylight_split", ("daylight",), ("day", "light"), "compound_split", "object compound may be written as a noun-noun phrase"),
    UnitRepair("plural_notebook", ("notebook",), ("notebooks",), "affix_plural", "object number changes only at the local noun unit"),
    UnitRepair("plural_sunflower", ("sunflower",), ("sunflowers",), "affix_plural", "object number changes only at the local noun unit"),
    UnitRepair("plural_raincoat", ("raincoat",), ("raincoats",), "affix_plural", "object number changes only at the local noun unit"),
    UnitRepair("plural_workshop", ("workshop",), ("workshops",), "affix_plural", "object number changes only at the local noun unit"),
)


def apply_one_local_repair(fragment: Fragment, repair: UnitRepair | None) -> tuple[tuple[str, ...], dict]:
    """Apply zero or one exact lexical-unit rewrite; never reverse-decode."""
    if repair is None:
        return fragment.words, {"repair_id": "none", "changed": False}
    for index in range(len(fragment.words) - len(repair.source) + 1):
        if fragment.words[index:index + len(repair.source)] == repair.source:
            words = fragment.words[:index] + repair.target + fragment.words[index + len(repair.source):]
            return words, {
                "repair_id": repair.repair_id,
                "changed": True,
                "unit_start": index,
                "unit_width_before": len(repair.source),
                "unit_width_after": len(repair.target),
                "class": repair.class_name,
                "syntax_preservation": repair.syntax_note,
            }
    return fragment.words, {"repair_id": repair.repair_id, "changed": False, "not_applicable": True}


def repo_palindrome_tapes(output: Path | None) -> tuple[frozenset[str], dict]:
    """Collect existing exact tapes while excluding this run's output path."""
    tapes: set[str] = set()
    scanned = 0
    skipped_output = 0
    output = output.resolve() if output else None

    def walk(value: object) -> None:
        if isinstance(value, str):
            try:
                tape = normalize_letters(value)
            except ValueError:
                return
            if len(tape) >= MIN_LETTERS and tape == tape[::-1]:
                tapes.add(tape)
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for path in sorted((ROOT / "runs").rglob("*.json")):
        if output and path.resolve() == output:
            skipped_output += 1
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        scanned += 1
        walk(payload)
    digest = hashlib.sha256("\n".join(sorted(tapes)).encode()).hexdigest()
    return frozenset(tapes), {
        "files_scanned": scanned,
        "output_files_excluded": skipped_output,
        "palindrome_tapes": len(tapes),
        "fingerprint_sha256": digest,
    }


def two_pointer_audit(tape: str) -> dict[str, object]:
    """Independently compare every mirrored pair without slicing reversal."""
    mismatches: list[dict[str, object]] = []
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            mismatches.append({"left": left, "right": right, "left_char": tape[left], "right_char": tape[right]})
        left += 1
        right -= 1
    return {"exact": not mismatches and bool(tape), "comparisons": len(tape) // 2, "mismatch_count": len(mismatches), "mismatches": mismatches[:40]}


def word_order_mirror(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    normalized_left = tuple(normalize_letters(word) for word in left)
    normalized_right = tuple(normalize_letters(word) for word in right)
    return bool(normalized_left) and normalized_right == tuple(word[::-1] for word in normalized_left[::-1])


def repeated_content_shortcut(units: tuple[str, ...]) -> bool:
    # The central gate has the authoritative hard checks; this local report
    # makes the construction-specific shortcut decision inspectable.
    content = [word for word in tokenize(" ".join(units)) if word not in {"a", "an", "the", "in", "on", "at", "by", "of", "to"}]
    return len(content) != len(set(content))


def render(left: tuple[str, ...], right: tuple[str, ...]) -> str:
    # Semicolon retains one grammatical sentence while contributing no letters.
    return " ".join(left).capitalize() + "; " + " ".join(right) + "."


def audit_row(left: Fragment, right: Fragment, right_words: tuple[str, ...], repair: dict, existing: frozenset[str]) -> dict:
    text = render(left.words, right_words)
    tape = normalize_letters(text)
    independent = bool(tape) and tape == tape[::-1]
    pointers = two_pointer_audit(tape)
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    shortcut = {
        "not_word_order_symmetry": not word_order_mirror(left.words, right_words),
        "not_repeated_content": not repeated_content_shortcut(left.words + right_words),
        "not_reverse_decoded": True,
        "single_local_repair_only": repair.get("repair_id") == "none" or repair.get("changed") is True,
    }
    exact_agreement = independent == bool(pointers["exact"])
    novel = tape not in existing
    return {
        "rendered": text,
        "letters": len(tape),
        "normalized_letters": tape,
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "independent_ascii_exact": independent,
        "independent_two_pointer": pointers,
        "independent_audits_agree": exact_agreement,
        "left_fragment": {"id": left.fragment_id, "side": left.side, "slots": left.slots, "authored_note": left.note},
        "right_fragment": {"id": right.fragment_id, "side": right.side, "slots": right.slots, "authored_note": right.note},
        "local_repair": repair,
        "shortcut_checks": shortcut,
        "existing_repository_tape_collision": not novel,
        "central_admission": checks,
        "mechanically_admitted": independent and bool(pointers["exact"]) and exact_agreement and novel and all(checks.values()) and all(shortcut.values()),
        "reader_status": "not_run; exactness and mechanical checks do not certify readability; requires blinded intact-prose study",
    }


def run(output: Path | None = None) -> dict:
    existing, fingerprint = repo_palindrome_tapes(output)
    rows: list[dict] = []
    stats = Counter({
        "left_fragments": len(LEFT_FRAGMENTS),
        "right_fragments": len(RIGHT_FRAGMENTS),
        "repair_rules": len(REPAIRS),
        "raw_cross_products": 0,
        "local_repair_attempts": 0,
        "local_repairs_applied": 0,
        "exact_rows": 0,
        "mechanically_admitted": 0,
        "repository_collision_rejections": 0,
    })
    # The baseline and repaired surfaces are both retained so a repair can be
    # evaluated as a local operator, not silently treated as a larger inventory.
    for left in LEFT_FRAGMENTS:
        for right in RIGHT_FRAGMENTS:
            for repair in (None, *REPAIRS):
                stats["raw_cross_products"] += 1
                if repair is not None:
                    stats["local_repair_attempts"] += 1
                right_words, repair_meta = apply_one_local_repair(right, repair)
                if repair_meta.get("changed"):
                    stats["local_repairs_applied"] += 1
                row = audit_row(left, right, right_words, repair_meta, existing)
                if row["independent_ascii_exact"]:
                    stats["exact_rows"] += 1
                if row["existing_repository_tape_collision"]:
                    stats["repository_collision_rejections"] += 1
                if row["mechanically_admitted"]:
                    stats["mechanically_admitted"] += 1
                rows.append(row)

    rows.sort(key=lambda row: (row["mechanically_admitted"], row["independent_ascii_exact"], -row["independent_two_pointer"]["mismatch_count"], row["letters"]), reverse=True)
    exact = [row for row in rows if row["independent_ascii_exact"]]
    probes = [row for row in rows if not row["independent_ascii_exact"]]
    probes.sort(key=lambda row: (row["independent_two_pointer"]["mismatch_count"], -row["letters"]))
    return {
        "status": "complete_multiword_unit_transducer_local_repair_pilot",
        "family_id": FAMILY_ID,
        "state_space_signature": STATE_SPACE_SIGNATURE,
        "signature": STATE_SPACE_SIGNATURE,
        "config": {
            "construction": "independent idiomatic multiword fragments joined as one semicolon sentence",
            "transducer": "at most one reversible affix or compound split/join on one local right-hand unit",
            "reverse_decoder": False,
            "center_out_replay": False,
            "brown_pos_event_discourse_dialogue_sources": False,
            "minimum_letters": MIN_LETTERS,
            "maximum_letters": MAX_LETTERS,
            "repository_tape_exclusion": True,
            "anti_shortcut_gate": True,
        },
        "stats": dict(stats),
        "repository_fingerprint": fingerprint,
        "admitted": [row for row in rows if row["mechanically_admitted"]],
        "rendered_candidates": exact[:100],
        "residual_probes": probes[:40],
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_text_copied": False,
            "left_bank": "hand-authored idiomatic clause fragments v1",
            "right_bank": "independently hand-authored idiomatic clause fragments v1",
            "repair_inventory": "hand-authored reversible affix/compound rules v1",
            "output_excluded_from_fingerprint": output is not None,
            "readability_evidence": "none; all rows require blinded reader evaluation",
        },
        "next_operator": (
            "Retain the multiword-unit transducer but add a second independently authored unit slot only "
            "after a dead-frontier analysis; permit at most one affix/compound repair per side and rerun "
            "the same exact two-pointer, repository-exclusion, and blinded-readability gates."
        ),
        "reader_gate": "No rendered candidate is promoted as readable evidence; use intact prose plus shuffled controls and independent raters.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite output: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(args.out)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "stats": result["stats"], "admitted": len(result["admitted"])}, indent=2))


if __name__ == "__main__":
    main()
